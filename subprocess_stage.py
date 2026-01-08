from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows can default to a non-UTF8 stdout encoding (e.g. cp1252), which can crash
# on printing certain unicode characters. Force UTF-8 so subprocess stages never
# fail due to log output.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# Keep env consistent with the Gradio app.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
ASSETS_DIR = APP_DIR / "assets"
O_VOXEL_SRC_DIR = APP_DIR / "o-voxel"

# Ensure TRELLIS models dir is discoverable (offline-friendly).
os.environ.setdefault("TRELLIS_MODELS_DIR", str(MODELS_DIR))


def _log_vram_usage(label: str) -> None:
    """Log current VRAM usage for debugging OOM issues."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] {label}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)
    except Exception:
        pass  # Silently ignore if torch not imported yet


def _ensure_o_voxel_available() -> None:
    """
    TRELLIS.2 depends on the CUDA extension package `o_voxel`.
    If it's not installed (common on Windows), attempt to install from bundled source.
    """
    try:
        import o_voxel  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise RuntimeError(
            "Failed to import 'o_voxel' (it may be installed but unusable).\n"
            "Try reinstalling from the bundled source:\n"
            "  python -m pip install ./o-voxel --no-build-isolation\n"
        ) from e

    if not O_VOXEL_SRC_DIR.is_dir():
        raise ModuleNotFoundError(
            "No module named 'o_voxel'. Also could not find bundled source at "
            f"{str(O_VOXEL_SRC_DIR)!r}."
        )

    import subprocess

    print(f"[setup] 'o_voxel' not found. Installing from bundled source: {O_VOXEL_SRC_DIR}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", str(O_VOXEL_SRC_DIR), "--no-build-isolation"])

    import importlib

    importlib.invalidate_caches()
    import o_voxel  # noqa: F401


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_sparse(path: Path) -> Tuple["torch.Tensor", "torch.Tensor"]:
    import numpy as np
    import torch

    data = np.load(str(path))
    feats = torch.from_numpy(data["feats"])
    coords = torch.from_numpy(data["coords"])
    return feats, coords


def _save_npz_sparse(path: Path, feats: "torch.Tensor", coords: "torch.Tensor") -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), feats=feats.detach().cpu().numpy(), coords=coords.detach().cpu().numpy())


def _filter_spatial_cache_for_save(cache: dict) -> dict:
    """Filter out SubMConv3dNeighborCache entries which can't be properly serialized/moved.
    These caches will be regenerated automatically by the convolution operations."""
    filtered = {}
    for key, value in cache.items():
        if isinstance(value, dict):
            # Recursively filter nested dicts (scale-keyed cache structure)
            filtered_sub = {}
            for k, v in value.items():
                # Skip neighbor cache entries (contain SubMConv3dNeighborCache)
                if 'neighbor' in str(k).lower():
                    continue
                # Check if value is a SubMConv3dNeighborCache object
                if hasattr(v, '__class__') and 'NeighborCache' in v.__class__.__name__:
                    continue
                filtered_sub[k] = v
            if filtered_sub:
                filtered[key] = filtered_sub
        else:
            # Skip if it's a NeighborCache object
            if hasattr(value, '__class__') and 'NeighborCache' in value.__class__.__name__:
                continue
            filtered[key] = value
    return filtered


def _save_sparse_tensor_full(path: Path, tensor: "SparseTensor") -> None:
    """Save a SparseTensor including its spatial cache using torch.save."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    # Move to CPU before saving
    tensor_cpu = tensor.cpu()
    # Filter out NeighborCache entries that can't be properly moved between devices
    filtered_cache = _filter_spatial_cache_for_save(tensor_cpu._spatial_cache)
    data = {
        "feats": tensor_cpu.feats,
        "coords": tensor_cpu.coords,
        "_spatial_cache": filtered_cache,
        "_scale": tensor_cpu._scale if hasattr(tensor_cpu, '_scale') else None,
        "_shape": tensor_cpu._shape if hasattr(tensor_cpu, '_shape') else None,
    }
    torch.save(data, str(path))


def _move_cache_to_device(cache: dict, device: str) -> dict:
    """Recursively move all tensors in a nested dict/tuple structure to device."""
    import torch

    if isinstance(cache, dict):
        return {k: _move_cache_to_device(v, device) for k, v in cache.items()}
    elif isinstance(cache, tuple):
        return tuple(_move_cache_to_device(v, device) for v in cache)
    elif isinstance(cache, list):
        return [_move_cache_to_device(v, device) for v in cache]
    elif isinstance(cache, torch.Tensor):
        return cache.to(device)
    else:
        # For custom objects like SubMConv3dNeighborCache, try to move if possible
        if hasattr(cache, 'to'):
            return cache.to(device)
        return cache


def _load_sparse_tensor_full(path: Path, device: str = "cpu") -> "SparseTensor":
    """Load a SparseTensor including its spatial cache."""
    import torch
    from trellis2.modules.sparse import SparseTensor

    # weights_only=False needed because spatial cache contains custom flex_gemm objects
    data = torch.load(str(path), map_location="cpu", weights_only=False)

    # Move spatial cache tensors to target device
    spatial_cache = data.get("_spatial_cache", {})
    if device != "cpu":
        spatial_cache = _move_cache_to_device(spatial_cache, device)

    tensor = SparseTensor(
        feats=data["feats"],
        coords=data["coords"],
        spatial_cache=spatial_cache,
    )
    if data.get("_scale") is not None:
        tensor._scale = data["_scale"]
    if data.get("_shape") is not None:
        tensor._shape = data["_shape"]
    return tensor.to(device)


def _load_cond(path: Path, device: str) -> Dict[str, "torch.Tensor"]:
    import torch

    cond_cpu = torch.load(str(path), map_location="cpu")
    if not isinstance(cond_cpu, dict):
        raise ValueError(f"Invalid cond file (expected dict): {path}")
    return {k: v.to(device) for k, v in cond_cpu.items()}


def _save_cond(path: Path, cond: Dict[str, "torch.Tensor"]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in cond.items()}, str(path))


def _pipeline_type_from_resolution(resolution: str) -> tuple[str, int]:
    """
    Convert resolution string to pipeline type and target resolution.
    
    Returns:
        (pipeline_type, target_resolution)
    
    Supports any resolution >=512 and divisible by 128.
    """
    try:
        res = int(resolution)
    except (ValueError, TypeError):
        raise ValueError(f"Resolution must be a number, got: {resolution}")
    
    if res < 512:
        raise ValueError(f"Resolution must be >= 512, got: {res}")
    
    if res % 128 != 0:
        raise ValueError(f"Resolution must be divisible by 128, got: {res}")
    
    if res == 512:
        return "512", 512
    elif res == 1024:
        # Match reference pipeline: 1024 uses the cascade path.
        return "1024_cascade", 1024
    else:
        # Any other resolution uses cascade
        return f"{res}_cascade", res


def _ss_res_from_pipeline_type(pipeline_type: str) -> int:
    """Sparse structure resolution: 64 for direct 1024, 32 for all others."""
    return 64 if pipeline_type == "1024" else 32


def _target_res_from_pipeline_type(pipeline_type: str, default_res: int) -> int:
    """Extract target resolution from cascade pipeline type."""
    if "_cascade" in pipeline_type:
        # Extract number from "1024_cascade", "1536_cascade", etc.
        return int(pipeline_type.split("_")[0])
    return default_res


def _ignore_all_image_models() -> List[str]:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    return list(getattr(Trellis2ImageTo3DPipeline, "model_names_to_load", []))


def _ignore_except_image_models(keep: List[str]) -> List[str]:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    names = list(getattr(Trellis2ImageTo3DPipeline, "model_names_to_load", []))
    return [n for n in names if n not in set(keep)]


def _ignore_except_texturing_models(keep: List[str]) -> List[str]:
    from trellis2.pipelines import Trellis2TexturingPipeline

    names = list(getattr(Trellis2TexturingPipeline, "model_names_to_load", []))
    return [n for n in names if n not in set(keep)]


def stage_preprocess_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    low_vram = payload.get("low_vram", False)
    in_path = Path(payload["input_image_path"])
    out_path = Path(payload["output_image_path"])

    print(f"[preprocess] loading image: {in_path}", flush=True)
    img = Image.open(str(in_path))

    # Preprocess uses rembg only; skip loading the heavy diffusion models.
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        model_repo,
        ignore_models=_ignore_all_image_models(),
        load_texture_models=False,
        load_image_cond_model=False,
        load_rembg_model=True,
    )
    pipe.low_vram = low_vram
    pipe.cuda()

    print("[preprocess] removing background / cropping…", flush=True)
    out = pipe.preprocess_image(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_path))
    print(f"[preprocess] saved: {out_path}", flush=True)
    return {"output_image_path": str(out_path)}


def stage_encode_cond(payload: Dict[str, Any]) -> Dict[str, Any]:
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    low_vram = payload.get("low_vram", False)
    image_path = Path(payload["image_path"])
    resolution = str(payload["resolution"])
    pipeline_type, target_res = _pipeline_type_from_resolution(resolution)
    force_high_res_conditional = payload.get("force_high_res_conditional", False)

    cond_512_path = Path(payload["cond_512_path"])
    cond_1024_path = Path(payload["cond_1024_path"]) if payload.get("cond_1024_path") else None

    img = Image.open(str(image_path))

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        model_repo,
        ignore_models=_ignore_all_image_models(),
        load_texture_models=False,
        load_image_cond_model=True,
        load_rembg_model=False,
    )
    pipe.low_vram = low_vram
    pipe.cuda()

    # Use 1024 resolution for sparse structure conditioning if force_high_res_conditional is enabled
    cond_512_res = 1024 if force_high_res_conditional else 512
    print(f"[cond] computing image embeddings ({cond_512_res}px for sparse structure)…", flush=True)
    cond_512 = pipe.get_cond([img], cond_512_res)
    _save_cond(cond_512_path, cond_512)
    print(f"[cond] saved: {cond_512_path}", flush=True)

    if pipeline_type != "512":
        if cond_1024_path is None:
            raise ValueError("cond_1024_path is required for non-512 pipeline types.")
        print("[cond] computing image embeddings (1024px)…", flush=True)
        cond_1024 = pipe.get_cond([img], 1024)
        _save_cond(cond_1024_path, cond_1024)
        print(f"[cond] saved: {cond_1024_path}", flush=True)
        return {
            "cond_512_path": str(cond_512_path),
            "cond_1024_path": str(cond_1024_path),
            "pipeline_type": pipeline_type,
        }

    return {"cond_512_path": str(cond_512_path), "cond_1024_path": None, "pipeline_type": pipeline_type}


def stage_sample_sparse_structure(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    seed = int(payload.get("seed", 42))
    resolution = str(payload["resolution"])
    pipeline_type, target_res = _pipeline_type_from_resolution(resolution)
    ss_res = _ss_res_from_pipeline_type(pipeline_type)
    low_vram = payload.get("low_vram", False)

    cond_512_path = Path(payload["cond_512_path"])
    coords_path = Path(payload["coords_path"])

    ss_params = payload["ss_params"]

    device = "cuda"
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        model_repo,
        ignore_models=_ignore_except_image_models(["sparse_structure_flow_model", "sparse_structure_decoder"]),
        load_texture_models=False,
        load_image_cond_model=False,
        load_rembg_model=False,
    )
    pipe.low_vram = low_vram
    pipe.cuda()

    print("[sparse] loading cond_512…", flush=True)
    cond = _load_cond(cond_512_path, device=device)

    # RNG handling:
    # - If an RNG state is provided, restore it so subprocess stages match the single-process
    #   reference pipeline noise sequence.
    # - Otherwise, seed once here (first sampling stage).
    rng_in = payload.get("rng_state_in_path")
    rng_out = payload.get("rng_state_out_path")
    if rng_in:
        state = torch.load(str(rng_in), map_location="cpu")
        if isinstance(state, dict) and "cpu" in state:
            torch.set_rng_state(state["cpu"])
            if torch.cuda.is_available() and state.get("cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all(state["cuda"])
                except Exception:
                    pass
        else:
            torch.set_rng_state(state)
        print(f"[sparse] restored RNG state: {rng_in}", flush=True)
    else:
        print(f"[sparse] setting random seed: {seed}", flush=True)
        torch.manual_seed(seed)
    
    print(f"[sparse] sampling sparse structure (ss_res={ss_res})…", flush=True)
    coords = pipe.sample_sparse_structure(cond, ss_res, 1, ss_params)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(coords.detach().cpu(), str(coords_path))
    print(f"[sparse] saved coords: {coords_path}", flush=True)

    # Persist RNG state after sampling so the next stage continues the same sequence.
    if rng_out:
        out_state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            try:
                out_state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                out_state["cuda"] = None
        torch.save(out_state, str(rng_out))
        print(f"[sparse] saved RNG state: {rng_out}", flush=True)
    return {"coords_path": str(coords_path)}


def stage_sample_shape_slat(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    seed = int(payload.get("seed", 42))
    resolution = str(payload["resolution"])
    pipeline_type, target_res = _pipeline_type_from_resolution(resolution)
    shape_params = payload["shape_params"]
    max_num_tokens = int(payload.get("max_num_tokens", 49152))
    low_vram = payload.get("low_vram", False)

    cond_512_path = Path(payload["cond_512_path"])
    cond_1024_path = Path(payload["cond_1024_path"]) if payload.get("cond_1024_path") else None
    coords_path = Path(payload["coords_path"])
    shape_slat_path = Path(payload["shape_slat_path"])
    out_res_path = Path(payload["out_res_path"])

    device = "cuda"

    rng_in = payload.get("rng_state_in_path")
    rng_out = payload.get("rng_state_out_path")
    if rng_in:
        state = torch.load(str(rng_in), map_location="cpu")
        if isinstance(state, dict) and "cpu" in state:
            torch.set_rng_state(state["cpu"])
            if torch.cuda.is_available() and state.get("cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all(state["cuda"])
                except Exception:
                    pass
        else:
            torch.set_rng_state(state)
        print(f"[shape] restored RNG state: {rng_in}", flush=True)
    else:
        # Backward-compatible fallback: deterministic but does NOT match single-process ordering.
        print(f"[shape] setting random seed: {seed}", flush=True)
        torch.manual_seed(seed)

    print("[shape] loading coords…", flush=True)
    coords = torch.load(str(coords_path), map_location="cpu").to(device)

    if pipeline_type == "512":
        keep = ["shape_slat_flow_model_512"]
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(
            model_repo,
            ignore_models=_ignore_except_image_models(keep),
            load_texture_models=False,
            load_image_cond_model=False,
            load_rembg_model=False,
        )
        pipe.cuda()

        cond = _load_cond(cond_512_path, device=device)
        print("[shape] sampling shape SLat (512)…", flush=True)
        slat = pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_512"], coords, shape_params)
        res = 512

    elif pipeline_type == "1024":
        if cond_1024_path is None:
            raise ValueError("cond_1024_path is required for 1024 pipeline type.")
        keep = ["shape_slat_flow_model_1024"]
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(
            model_repo,
            ignore_models=_ignore_except_image_models(keep),
            load_texture_models=False,
            load_image_cond_model=False,
            load_rembg_model=False,
        )
        pipe.cuda()

        cond = _load_cond(cond_1024_path, device=device)
        print("[shape] sampling shape SLat (1024)…", flush=True)
        slat = pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_1024"], coords, shape_params)
        res = 1024

    elif "_cascade" in pipeline_type:
        # Any cascade resolution (768, 1024, 1280, 1536, 2048, custom)
        if cond_1024_path is None:
            raise ValueError("cond_1024_path is required for cascade pipeline types.")

        keep = ["shape_slat_flow_model_512", "shape_slat_flow_model_1024", "shape_slat_decoder"]
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(
            model_repo,
            ignore_models=_ignore_except_image_models(keep),
            load_texture_models=False,
            load_image_cond_model=False,
            load_rembg_model=False,
        )
        pipe.cuda()

        lr_cond = _load_cond(cond_512_path, device=device)
        cond = _load_cond(cond_1024_path, device=device)

        print(f"[shape] sampling shape SLat (cascade → {target_res})…", flush=True)
        
        # Clear any leftover tensors before heavy operation
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        _log_vram_usage("Before cascade sampling")
        
        slat, res = pipe.sample_shape_slat_cascade(
            lr_cond,
            cond,
            pipe.models["shape_slat_flow_model_512"],
            pipe.models["shape_slat_flow_model_1024"],
            512,
            target_res,
            coords,
            shape_params,
            max_num_tokens,
        )
        
        # Immediate cleanup after cascade
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

    # Persist as npz (portable, easy to inspect).
    _save_npz_sparse(shape_slat_path, slat.feats, slat.coords)
    _write_json(out_res_path, {"res": int(res), "pipeline_type": pipeline_type})
    print(f"[shape] saved: {shape_slat_path}", flush=True)
    print(f"[shape] saved: {out_res_path} (res={res})", flush=True)

    if rng_out:
        out_state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            try:
                out_state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                out_state["cuda"] = None
        torch.save(out_state, str(rng_out))
        print(f"[shape] saved RNG state: {rng_out}", flush=True)

    return {"shape_slat_path": str(shape_slat_path), "res": int(res), "pipeline_type": pipeline_type}


def stage_sample_tex_slat(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    seed = int(payload.get("seed", 42))
    resolution = str(payload["resolution"])
    pipeline_type, target_res = _pipeline_type_from_resolution(resolution)
    low_vram = payload.get("low_vram", False)

    cond_path = Path(payload["cond_path"])
    shape_slat_path = Path(payload["shape_slat_path"])
    tex_slat_path = Path(payload["tex_slat_path"])
    tex_params = payload["tex_params"]

    device = "cuda"

    rng_in = payload.get("rng_state_in_path")
    rng_out = payload.get("rng_state_out_path")
    if rng_in:
        state = torch.load(str(rng_in), map_location="cpu")
        if isinstance(state, dict) and "cpu" in state:
            torch.set_rng_state(state["cpu"])
            if torch.cuda.is_available() and state.get("cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all(state["cuda"])
                except Exception:
                    pass
        else:
            torch.set_rng_state(state)
        print(f"[tex] restored RNG state: {rng_in}", flush=True)
    else:
        print(f"[tex] setting random seed: {seed}", flush=True)
        torch.manual_seed(seed)

    feats, coords = _load_npz_sparse(shape_slat_path)
    shape_slat = SparseTensor(feats=feats.to(device), coords=coords.to(device))

    if pipeline_type == "512":
        keep = ["tex_slat_flow_model_512"]
        flow_key = "tex_slat_flow_model_512"
    else:
        keep = ["tex_slat_flow_model_1024"]
        flow_key = "tex_slat_flow_model_1024"

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        model_repo,
        ignore_models=_ignore_except_image_models(keep),
        load_texture_models=True,
        load_image_cond_model=False,
        load_rembg_model=False,
    )
    pipe.low_vram = low_vram
    pipe.cuda()

    cond = _load_cond(cond_path, device=device)

    print(f"[tex] sampling texture SLat ({flow_key})…", flush=True)
    tex_slat = pipe.sample_tex_slat(cond, pipe.models[flow_key], shape_slat, tex_params)
    _save_npz_sparse(tex_slat_path, tex_slat.feats, tex_slat.coords)
    print(f"[tex] saved: {tex_slat_path}", flush=True)

    if rng_out:
        out_state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            try:
                out_state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                out_state["cuda"] = None
        torch.save(out_state, str(rng_out))
        print(f"[tex] saved RNG state: {rng_out}", flush=True)
    return {"tex_slat_path": str(tex_slat_path)}


def _has_nvdiffrec_render() -> bool:
    try:
        import nvdiffrec_render  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _tensor_chw01_to_uint8_hwc(img: "torch.Tensor") -> "np.ndarray":
    import numpy as np
    import torch

    if img.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img.shape)}")
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.detach().float().clamp(0, 1)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return arr


def _simple_shaded(base_color: "torch.Tensor", normal_01: "torch.Tensor", tint: "torch.Tensor") -> "torch.Tensor":
    import torch

    n = (normal_01 * 2.0 - 1.0)
    light_dir = torch.tensor([0.4, 0.2, 0.9], device=n.device, dtype=n.dtype)
    light_dir = light_dir / (light_dir.norm() + 1e-8)
    lambert = (n * light_dir.view(3, 1, 1)).sum(dim=0, keepdim=True).clamp(0.0, 1.0)
    ambient = 0.35
    shaded = base_color * (ambient + (1.0 - ambient) * lambert)
    shaded = shaded * tint.view(3, 1, 1).clamp(0.0, 2.0)
    return shaded.clamp(0.0, 1.0)


def stage_render_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    low_vram = payload.get("low_vram", False)
    shape_slat_path = Path(payload["shape_slat_path"])
    tex_slat_path = Path(payload["tex_slat_path"]) if payload.get("tex_slat_path") else None
    res = int(payload["res"])
    preview_dir = Path(payload["preview_dir"])
    manifest_path = Path(payload["preview_manifest_path"])
    use_tiled_extraction = bool(payload.get("use_tiled_extraction", False))
    use_chunked_processing = bool(payload.get("use_chunked_processing", False))

    device = "cuda"

    feats, coords = _load_npz_sparse(shape_slat_path)
    shape_slat = SparseTensor(feats=feats.to(device), coords=coords.to(device))
    if tex_slat_path is not None:
        t_feats, _ = _load_npz_sparse(tex_slat_path)
        tex_slat = shape_slat.replace(t_feats.to(device))
    else:
        tex_slat = None

    keep = ["shape_slat_decoder"]
    if tex_slat is not None:
        keep.append("tex_slat_decoder")

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        model_repo,
        ignore_models=_ignore_except_image_models(keep),
        load_texture_models=False,
        load_image_cond_model=False,
        load_rembg_model=False,
    )
    pipe.low_vram = low_vram
    pipe.cuda()

    # Clear memory before heavy decode operation
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram_usage("Before decode_latent")

    print("[preview] decoding latent to mesh…", flush=True)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res, use_tiled_extraction, use_chunked_processing)[0]
    
    # Clear memory after decode
    torch.cuda.empty_cache()

    print("[preview] simplifying mesh…", flush=True)
    try:
        mesh.simplify(16777216)
    except Exception as e:
        print(f"[preview] simplify failed: {type(e).__name__}: {e}", flush=True)

    # Render setup (match app_premium).
    MODES = [
        {"name": "Normal", "render_key": "normal"},
        {"name": "Clay render", "render_key": "clay"},
        {"name": "Base color", "render_key": "base_color"},
        {"name": "HDRI forest", "render_key": "shaded_forest"},
        {"name": "HDRI sunset", "render_key": "shaded_sunset"},
        {"name": "HDRI courtyard", "render_key": "shaded_courtyard"},
    ]
    STEPS = 8

    pbr_supported = _has_nvdiffrec_render()
    images: Dict[str, List[np.ndarray]] = {m["render_key"]: [] for m in MODES}

    # Camera extrinsics/intrinsics (8 views)
    yaw = np.linspace(0, 2 * np.pi, STEPS, endpoint=False)
    yaw = [float(y - 16 / 180 * np.pi) for y in yaw]
    pitch = [float(20 / 180 * np.pi) for _ in range(STEPS)]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2.0, 36.0)

    if pbr_supported:
        print("[preview] PBR preview enabled (nvdiffrec_render found).", flush=True)
        import cv2
        from trellis2.renderers import EnvMap, PbrMeshRenderer

        def _load_env(name: str) -> EnvMap:
            path = ASSETS_DIR / "hdri" / f"{name}.exr"
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return EnvMap(torch.tensor(img, dtype=torch.float32, device="cuda"))

        envmap = {"forest": _load_env("forest"), "sunset": _load_env("sunset"), "courtyard": _load_env("courtyard")}
        renderer = PbrMeshRenderer(
            rendering_options={"resolution": 1024, "near": 1, "far": 100, "ssaa": 2, "peel_layers": 8}
        )

        for j, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
            print(f"[preview] rendering view {j + 1}/{STEPS}…", flush=True)
            res_dict = renderer.render(mesh, extr, intr, envmap=envmap)
            for mode in MODES:
                key = mode["render_key"]
                if key not in res_dict:
                    # Fallback to base_color if a key is missing for some reason
                    fallback = res_dict.get("base_color", res_dict.get("clay"))
                    images[key].append(_tensor_chw01_to_uint8_hwc(fallback))
                else:
                    images[key].append(_tensor_chw01_to_uint8_hwc(res_dict[key]))
    else:
        print("[preview] PBR preview disabled (missing nvdiffrec_render). Using simple shading.", flush=True)
        from trellis2.renderers import MeshRenderer

        renderer = MeshRenderer(
            rendering_options={"resolution": 1024, "near": 1, "far": 100, "ssaa": 2, "chunk_size": None}
        )
        t_forest = torch.tensor([0.85, 1.05, 0.85], device="cuda")
        t_sunset = torch.tensor([1.10, 0.90, 0.75], device="cuda")
        t_court = torch.tensor([0.85, 0.95, 1.10], device="cuda")

        for j, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
            print(f"[preview] rendering view {j + 1}/{STEPS}…", flush=True)
            res_dict = renderer.render(mesh, extr, intr, return_types=["mask", "normal", "attr"])
            normal = res_dict["normal"]  # (3,H,W) in [0,1]
            base_color = res_dict.get("base_color", torch.full_like(normal, 0.8))

            clay_base = torch.full_like(base_color, 0.78)
            clay = _simple_shaded(clay_base, normal, torch.tensor([1.0, 1.0, 1.0], device=normal.device))
            shaded_forest = _simple_shaded(base_color, normal, t_forest)
            shaded_sunset = _simple_shaded(base_color, normal, t_sunset)
            shaded_courtyard = _simple_shaded(base_color, normal, t_court)

            mode_map = {
                "normal": normal,
                "clay": clay,
                "base_color": base_color,
                "shaded_forest": shaded_forest,
                "shaded_sunset": shaded_sunset,
                "shaded_courtyard": shaded_courtyard,
            }
            for mode in MODES:
                key = mode["render_key"]
                images[key].append(_tensor_chw01_to_uint8_hwc(mode_map[key]))

    # Persist images to disk (JPEG) and write a manifest.
    preview_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[str]] = {}
    for m_idx, mode in enumerate(MODES):
        key = mode["render_key"]
        manifest[key] = []
        for s_idx in range(STEPS):
            fname = f"view-m{m_idx}-s{s_idx}.jpg"
            path = preview_dir / fname
            Image.fromarray(images[key][s_idx]).save(str(path), format="JPEG", quality=85)
            manifest[key].append(str(path))

    _write_json(manifest_path, {"modes": MODES, "steps": STEPS, "files": manifest})
    print(f"[preview] saved manifest: {manifest_path}", flush=True)
    return {"preview_manifest_path": str(manifest_path), "preview_dir": str(preview_dir)}


def stage_extract_glb(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    from subprocess_utils import next_indexed_path

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    low_vram = payload.get("low_vram", False)
    shape_slat_path = Path(payload["shape_slat_path"])
    tex_slat_path = Path(payload["tex_slat_path"]) if payload.get("tex_slat_path") else None
    res = int(payload["res"])

    decimation_target = int(payload["decimation_target"])
    texture_size = int(payload["texture_size"])
    requested_remesh_method = str(payload["remesh_method"])
    remesh_method = requested_remesh_method
    simplify_method = str(payload["simplify_method"])
    prune_invisible_faces = bool(payload["prune_invisible_faces"])
    no_texture_gen = bool(payload["no_texture_gen"])
    
    # Extract GLB mesh extraction settings (user-configurable)
    extract_use_tiled_extraction = bool(payload.get("extract_use_tiled_extraction", False))
    extract_use_chunked_processing = bool(payload.get("extract_use_chunked_processing", False))

    out_dir = Path(payload["out_dir"])
    prefix = str(payload.get("prefix", "glb"))
    export_formats = payload.get("export_formats") or ["glb"]
    export_formats = [str(f).lower().strip() for f in export_formats]
    if "glb" not in export_formats:
        export_formats = ["glb"] + export_formats

    device = "cuda"

    feats, coords = _load_npz_sparse(shape_slat_path)
    shape_slat = SparseTensor(feats=feats.to(device), coords=coords.to(device))
    if tex_slat_path is not None and not no_texture_gen:
        t_feats, _ = _load_npz_sparse(tex_slat_path)
        tex_slat = shape_slat.replace(t_feats.to(device))
    else:
        tex_slat = None

    keep = ["shape_slat_decoder"]
    if tex_slat is not None:
        keep.append("tex_slat_decoder")

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        model_repo,
        ignore_models=_ignore_except_image_models(keep),
        load_texture_models=False,
        load_image_cond_model=False,
        load_rembg_model=False,
    )
    pipe.low_vram = low_vram
    pipe.cuda()

    # Clear memory before heavy decode operation
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram_usage("Before extract decode_latent")

    print("[extract] decoding latent to mesh…", flush=True)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res, extract_use_tiled_extraction, extract_use_chunked_processing)[0]
    
    # Save values needed later before unloading pipeline
    pbr_attr_layout = pipe.pbr_attr_layout
    
    # CRITICAL: Unload pipeline entirely to free VRAM for mesh operations
    # The pipeline holds GBs of decoder weights that must be freed before to_glb
    print("[extract] freeing decoder memory…", flush=True)
    del shape_slat, tex_slat
    pipe.cpu()  # Move all models to CPU
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram_usage("After pipeline unload, before to_glb")

    print("[extract] converting to GLB…", flush=True)
    # NOTE: `faithful_contouring` remeshing depends on optional FaithC packages
    # (`faithcontour` + `atom3d`). These are not installed by default on many
    # setups (especially Windows). Instead of failing the whole extraction,
    # fall back to the built-in `dual_contouring` remesher with a clear log.
    if remesh_method == "faithful_contouring":
        try:
            import importlib

            importlib.import_module("faithcontour")
            importlib.import_module("atom3d")
        except Exception as e:
            print(
                "[extract] warning: remesh_method='faithful_contouring' requested but optional "
                f"dependency is missing/unusable ({type(e).__name__}: {e}). "
                "Falling back to 'dual_contouring'.",
                flush=True,
            )
            remesh_method = "dual_contouring"

    to_glb_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pbr_attr_layout,
        "grid_size": res,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": decimation_target,
        "simplify_method": simplify_method,
        "texture_extraction": not no_texture_gen,
        "texture_size": texture_size,
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0,
        "remesh_method": remesh_method,
        "prune_invisible": prune_invisible_faces,
        "use_tqdm": True,
    }
    try:
        glb = o_voxel.postprocess.to_glb(**to_glb_kwargs)
    except ImportError as e:
        # Failsafe: if the FaithC import fails inside `o_voxel` after our check,
        # retry once with a safe remesher.
        if requested_remesh_method == "faithful_contouring" and "Faithful Contouring is not installed" in str(e):
            fallback_method = "dual_contouring"
            print(
                f"[extract] warning: {e} Falling back to remesh_method={fallback_method!r}.",
                flush=True,
            )
            to_glb_kwargs["remesh_method"] = fallback_method
            glb = o_voxel.postprocess.to_glb(**to_glb_kwargs)
        else:
            raise

    idx, glb_path = next_indexed_path(out_dir, prefix=prefix, ext="glb", digits=4, start=1)
    glb.export(str(glb_path), extension_webp=False)

    # Optional extra exports (best effort; never fail the main GLB export).
    for fmt in export_formats:
        if fmt == "glb":
            continue
        try:
            if fmt == "gltf":
                gltf_path = out_dir / f"gltf_{idx:04d}.gltf"
                glb.export(str(gltf_path))
            elif fmt == "obj":
                obj_path = out_dir / f"obj_{idx:04d}.obj"
                glb.export(str(obj_path))
            elif fmt == "ply":
                ply_path = out_dir / f"ply_{idx:04d}.ply"
                glb.export(str(ply_path))
            elif fmt == "stl":
                stl_path = out_dir / f"stl_{idx:04d}.stl"
                glb.export(str(stl_path))
        except Exception as e:
            print(f"[extract] extra export '{fmt}' failed: {type(e).__name__}: {e}", flush=True)
    torch.cuda.empty_cache()
    print(f"[extract] saved: {glb_path}", flush=True)
    return {"glb_path": str(glb_path)}


def stage_tex_encode_cond(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 1: Load image conditioning model, compute embeddings, save, exit."""
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2TexturingPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    config_file = payload.get("config_file", "texturing_pipeline.json")
    
    image_path = Path(payload["image_path"])
    preprocessed_image_path = Path(payload["preprocessed_image_path"])
    cond_path = Path(payload["cond_path"])
    resolution = int(payload["resolution"])
    seed = int(payload["seed"])

    device = "cuda"
    
    print("[tex_cond] loading texturing pipeline (image_cond_model only)...", flush=True)
    _log_vram_usage("Before loading")
    
    # Load ONLY image conditioning model
    pipe = Trellis2TexturingPipeline.from_pretrained(
        model_repo,
        config_file=config_file,
        ignore_models=_ignore_except_texturing_models([])  # Load base + image_cond_model
    )
    pipe.cuda()
    
    _log_vram_usage("After loading")
    
    print("[tex_cond] loading and preprocessing image...", flush=True)
    img = Image.open(str(image_path))
    img = pipe.preprocess_image(img)
    
    # Save preprocessed image
    preprocessed_image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(preprocessed_image_path))
    print(f"[tex_cond] saved preprocessed image: {preprocessed_image_path}", flush=True)
    
    torch.manual_seed(seed)
    cond_res = 512 if resolution == 512 else 1024
    print(f"[tex_cond] computing image embeddings ({cond_res}px)...", flush=True)
    cond = pipe.get_cond([img], cond_res)
    
    # Save conditioning
    cond_path.parent.mkdir(parents=True, exist_ok=True)
    _save_cond(cond_path, cond)
    print(f"[tex_cond] saved: {cond_path}", flush=True)
    
    torch.cuda.empty_cache()
    _log_vram_usage("After save (before exit)")
    
    return {"cond_path": str(cond_path), "preprocessed_image_path": str(preprocessed_image_path)}


def stage_tex_encode_shape(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 2: Load Texturing pipeline, encode mesh, save, exit."""
    import trimesh
    import torch
    from trellis2.pipelines import Trellis2TexturingPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    config_file = payload.get("config_file", "texturing_pipeline.json")

    mesh_path = Path(payload["mesh_path"])
    shape_slat_path = Path(payload["shape_slat_path"])
    resolution = int(payload["resolution"])

    device = "cuda"

    print("[tex_shape] loading Texturing pipeline (shape_slat_encoder only)...", flush=True)
    _log_vram_usage("Before loading")

    # Use Texturing pipeline which has encode_shape_slat and preprocess_mesh
    # Only load shape_slat_encoder to save memory
    pipe = Trellis2TexturingPipeline.from_pretrained(
        model_repo,
        config_file=config_file,
        ignore_models=_ignore_except_texturing_models(["shape_slat_encoder"])
    )
    pipe.cuda()

    _log_vram_usage("After loading")

    print("[tex_shape] loading mesh...", flush=True)
    mesh = trimesh.load(str(mesh_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    # Use the pipeline's preprocessing method
    print("[tex_shape] preprocessing mesh...", flush=True)
    mesh = pipe.preprocess_mesh(mesh)
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram_usage("Before encoding")

    print("[tex_shape] encoding mesh to shape latent...", flush=True)
    # Use inference mode to avoid gradient memory allocation
    with torch.inference_mode():
        shape_slat = pipe.encode_shape_slat(mesh, resolution)

    # Save shape latent with spatial cache (needed for decoding)
    _save_sparse_tensor_full(shape_slat_path, shape_slat)
    print(f"[tex_shape] saved: {shape_slat_path}", flush=True)
    
    torch.cuda.empty_cache()
    _log_vram_usage("After save (before exit)")
    
    return {"shape_slat_path": str(shape_slat_path)}


def stage_tex_sample_tex_slat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 3: Load texture flow model, sample texture latent, save, exit."""
    import torch
    from trellis2.pipelines import Trellis2TexturingPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    config_file = payload.get("config_file", "texturing_pipeline.json")
    
    cond_path = Path(payload["cond_path"])
    shape_slat_path = Path(payload["shape_slat_path"])
    tex_slat_path = Path(payload["tex_slat_path"])
    resolution = int(payload["resolution"])
    tex_params = payload["tex_params"]
    seed = int(payload["seed"])

    device = "cuda"
    
    # Determine which model to load
    tex_model_key = "tex_slat_flow_model_512" if resolution == 512 else "tex_slat_flow_model_1024"
    
    print(f"[tex_sample] loading texturing pipeline ({tex_model_key} only)...", flush=True)
    _log_vram_usage("Before loading")
    
    # Load ONLY texture flow model
    pipe = Trellis2TexturingPipeline.from_pretrained(
        model_repo,
        config_file=config_file,
        ignore_models=_ignore_except_texturing_models([tex_model_key])
    )
    pipe.cuda()
    
    _log_vram_usage("After loading")
    
    print("[tex_sample] loading conditioning...", flush=True)
    cond = _load_cond(cond_path, device=device)

    print("[tex_sample] loading shape latent (with spatial cache)...", flush=True)
    shape_slat = _load_sparse_tensor_full(shape_slat_path, device=device)

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram_usage("Before sampling")

    torch.manual_seed(seed)
    print(f"[tex_sample] sampling texture latent ({tex_model_key})...", flush=True)
    tex_slat = pipe.sample_tex_slat(cond, pipe.models[tex_model_key], shape_slat, tex_params)

    # Save texture latent with spatial cache (needed for decoding)
    _save_sparse_tensor_full(tex_slat_path, tex_slat)
    print(f"[tex_sample] saved: {tex_slat_path}", flush=True)
    
    torch.cuda.empty_cache()
    _log_vram_usage("After save (before exit)")
    
    return {"tex_slat_path": str(tex_slat_path)}


def stage_tex_decode_and_bake(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 4: Load decoder, decode texture, bake onto mesh, save GLB, exit."""
    import trimesh
    import torch
    from trellis2.pipelines import Trellis2TexturingPipeline
    from subprocess_utils import next_indexed_path

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    config_file = payload.get("config_file", "texturing_pipeline.json")
    
    mesh_path = Path(payload["mesh_path"])
    tex_slat_path = Path(payload["tex_slat_path"])
    resolution = int(payload["resolution"])
    texture_size = int(payload["texture_size"])
    out_dir = Path(payload["out_dir"])
    prefix = str(payload.get("prefix", "textured"))

    device = "cuda"
    
    print("[tex_bake] loading texturing pipeline (tex_slat_decoder only)...", flush=True)
    _log_vram_usage("Before loading")
    
    # Load ONLY texture decoder
    pipe = Trellis2TexturingPipeline.from_pretrained(
        model_repo,
        config_file=config_file,
        ignore_models=_ignore_except_texturing_models(["tex_slat_decoder"])
    )
    pipe.cuda()
    
    _log_vram_usage("After loading")
    
    print("[tex_bake] loading mesh...", flush=True)
    mesh = trimesh.load(str(mesh_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    mesh = pipe.preprocess_mesh(mesh)
    
    print("[tex_bake] loading texture latent (with spatial cache)...", flush=True)
    tex_slat = _load_sparse_tensor_full(tex_slat_path, device=device)

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram_usage("Before decoding")

    print("[tex_bake] decoding texture latent...", flush=True)
    # Use inference mode to reduce memory usage
    with torch.inference_mode():
        pbr_voxel = pipe.decode_tex_slat(tex_slat)

    # Free tex_slat memory before postprocessing
    del tex_slat
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram_usage("After decoding")

    print("[tex_bake] baking textures onto mesh...", flush=True)
    with torch.inference_mode():
        out_mesh = pipe.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size)
    
    _, out_path = next_indexed_path(out_dir, prefix=prefix, ext="glb", digits=4, start=1)
    out_mesh.export(str(out_path), extension_webp=False)
    
    torch.cuda.empty_cache()
    _log_vram_usage("After export (before exit)")
    
    print(f"[tex_bake] saved: {out_path}", flush=True)
    return {"glb_path": str(out_path)}



def main() -> int:
    parser = argparse.ArgumentParser(description="TRELLIS.2 subprocess stage runner")
    parser.add_argument("--stage", required=True, type=str)
    parser.add_argument("--payload", required=True, type=str)
    parser.add_argument("--result", required=True, type=str)
    args = parser.parse_args()

    try:
        _ensure_o_voxel_available()

        stage = args.stage.strip()
        payload_path = Path(args.payload)
        result_path = Path(args.result)
        payload = _read_json(payload_path)

        print(f"[stage] {stage}", flush=True)

        if stage == "preprocess_image":
            result = stage_preprocess_image(payload)
        elif stage == "encode_cond":
            result = stage_encode_cond(payload)
        elif stage == "sample_sparse_structure":
            result = stage_sample_sparse_structure(payload)
        elif stage == "sample_shape_slat":
            result = stage_sample_shape_slat(payload)
        elif stage == "sample_tex_slat":
            result = stage_sample_tex_slat(payload)
        elif stage == "render_preview":
            result = stage_render_preview(payload)
        elif stage == "extract_glb":
            result = stage_extract_glb(payload)
        elif stage == "tex_encode_cond":
            result = stage_tex_encode_cond(payload)
        elif stage == "tex_encode_shape":
            result = stage_tex_encode_shape(payload)
        elif stage == "tex_sample_tex_slat":
            result = stage_tex_sample_tex_slat(payload)
        elif stage == "tex_decode_and_bake":
            result = stage_tex_decode_and_bake(payload)
        else:
            raise ValueError(f"Unknown stage: {stage}")

        _write_json(result_path, {"ok": True, "result": result})
        return 0
    except Exception as e:
        err = {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json(Path(args.result), err)  # type: ignore[arg-type]
        except Exception:
            pass
        print(err["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


