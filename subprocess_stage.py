from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# UniRig checkpoints are trusted local files but may include custom objects.
# PyTorch 2.6+ defaults torch.load(weights_only=True), which can fail on these ckpts.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

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



# ================================ UniRig Stages ================================


_UNIRIG_PREDICT_IMPORT_TO_PACKAGE = {
    "box": "python-box",
    "bpy": "bpy",
    "torch": "torch",
    "lightning": "lightning",
    "pytorch_lightning": "pytorch-lightning",
    "omegaconf": "omegaconf",
    "yaml": "PyYAML",
    "fast_simplification": "fast-simplification",
    "trimesh": "trimesh",
    "tqdm": "tqdm",
}

_UNIRIG_MERGE_IMPORT_TO_PACKAGE = {
    "box": "python-box",
    "bpy": "bpy",
    "open3d": "open3d",
}


def _resolve_unirig_python(payload: Dict[str, Any]) -> str:
    def _validate_candidate(candidate: str, source: str) -> Optional[str]:
        c = str(candidate or "").strip().strip('"').strip("'")
        if not c:
            return None
        # If this looks like a path, require it to exist to avoid cryptic failures later.
        looks_like_path = ("/" in c) or ("\\" in c) or c.lower().endswith(".exe")
        if looks_like_path and not Path(c).exists():
            raise FileNotFoundError(f"{source} points to a missing Python executable: {c}")
        return c

    payload_candidate = _validate_candidate(payload.get("unirig_python", ""), "payload.unirig_python")
    if payload_candidate:
        return payload_candidate

    env_candidate = _validate_candidate(os.environ.get("UNIRIG_PYTHON", ""), "UNIRIG_PYTHON")
    if env_candidate:
        return env_candidate

    py_name = "python.exe" if os.name == "nt" else "python"
    py_dir = "Scripts" if os.name == "nt" else "bin"
    local_candidates = [
        APP_DIR / "UniRig" / ".venv" / py_dir / py_name,
        APP_DIR / "UniRig" / "venv" / py_dir / py_name,
        APP_DIR / ".venv" / py_dir / py_name,
        APP_DIR / "venv" / py_dir / py_name,
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


def _missing_modules_for_interpreter(python_exe: str, imports: List[str]) -> List[str]:
    import subprocess

    probe_code = (
        "import importlib.util, json; "
        f"mods={json.dumps(imports)}; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print(json.dumps(missing))"
    )

    try:
        proc = subprocess.run(
            [python_exe, "-c", probe_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Unable to execute UniRig Python interpreter: {python_exe}") from e

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Failed to verify UniRig dependencies with interpreter '{python_exe}'.\n{detail}"
        )

    raw = (proc.stdout or "").strip()
    try:
        missing = json.loads(raw or "[]")
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Dependency probe returned invalid JSON for interpreter '{python_exe}': {raw!r}"
        ) from e

    if not isinstance(missing, list):
        raise RuntimeError(
            f"Dependency probe returned unexpected payload for interpreter '{python_exe}': {raw!r}"
        )

    return [str(m) for m in missing]


def _ensure_unirig_runtime_ready(
    *,
    python_exe: str,
    import_to_package: Dict[str, str],
    stage_label: str,
) -> None:
    imports = list(import_to_package.keys())
    missing_imports = _missing_modules_for_interpreter(python_exe, imports)
    if not missing_imports:
        return

    missing_packages = [import_to_package.get(m, m) for m in missing_imports]
    missing_packages = list(dict.fromkeys(missing_packages))
    req_path = APP_DIR / "UniRig" / "requirements.txt"

    raise RuntimeError(
        f"UniRig environment is missing dependencies for stage '{stage_label}'.\n"
        f"Interpreter: {python_exe}\n"
        f"Missing imports: {', '.join(missing_imports)}\n"
        f"Suggested packages: {', '.join(missing_packages)}\n"
        f"Install with:\n  \"{python_exe}\" -m pip install -r \"{req_path}\"\n"
        "Or set UNIRIG_PYTHON to a Python executable where UniRig is already installed."
    )


def _run_logged_subprocess(cmd: List[str], *, cwd: Path, label: str) -> None:
    import subprocess

    print(f"[{label}] Running: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    tail: List[str] = []
    if proc.stdout:
        for line in proc.stdout:
            clean = line.rstrip("\n")
            print(clean, flush=True)
            if clean:
                tail.append(clean)
                if len(tail) > 40:
                    tail = tail[-40:]

    rc = proc.wait()
    if rc != 0:
        msg = f"{label} failed with exit code {rc}"
        if tail:
            msg += "\nLast UniRig log lines:\n" + "\n".join(tail[-20:])
        raise RuntimeError(msg)


def stage_unirig_skeleton(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate skeleton for a 3D mesh using UniRig.
    
    Payload:
        input_mesh_path: Path to input mesh (.obj, .fbx, .glb, .vrm)
        output_fbx_path: Path to output skeleton FBX
        npz_dir: Temporary directory for intermediate NPZ files
        seed: Random seed for skeleton generation
        skeleton_task: UniRig config path (default: configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml)
        faces_target_count: Target face count for mesh simplification (default: 50000)
    """
    input_mesh = Path(payload["input_mesh_path"])
    output_fbx = Path(payload["output_fbx_path"])
    npz_dir = Path(payload["npz_dir"])
    seed = int(payload.get("seed", 12345))
    skeleton_task = payload.get("skeleton_task", "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml")
    faces_target_count = int(payload.get("faces_target_count", 50000))
    
    unirig_dir = APP_DIR / "UniRig"
    run_py = unirig_dir / "run.py"
    
    if not run_py.exists():
        raise FileNotFoundError(f"UniRig run.py not found at: {run_py}")

    unirig_python = _resolve_unirig_python(payload)
    _ensure_unirig_runtime_ready(
        python_exe=unirig_python,
        import_to_package=_UNIRIG_PREDICT_IMPORT_TO_PACKAGE,
        stage_label="unirig_skeleton",
    )
    
    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[unirig_skeleton] Input: {input_mesh}", flush=True)
    print(f"[unirig_skeleton] Output: {output_fbx}", flush=True)
    print(f"[unirig_skeleton] Seed: {seed}", flush=True)
    print(f"[unirig_skeleton] Python: {unirig_python}", flush=True)

    # UniRig skeleton inference expects precomputed raw_data.npz under npz_dir.
    # Mirror UniRig's official launch flow: extract first, then run inference.
    extract_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    extract_cmd = [
        unirig_python,
        "-m",
        "src.data.extract",
        "--config=configs/data/quick_inference.yaml",
        "--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm",
        "--force_override=true",
        "--num_runs=1",
        "--id=0",
        f"--time={extract_stamp}",
        f"--faces_target_count={faces_target_count}",
        f"--input={input_mesh}",
        f"--output_dir={npz_dir}",
    ]
    _run_logged_subprocess(extract_cmd, cwd=unirig_dir, label="unirig_extract")

    raw_npz_files = list(npz_dir.rglob("raw_data.npz"))
    if not raw_npz_files:
        raise RuntimeError(
            f"UniRig extraction finished but produced no raw_data.npz under {npz_dir}"
        )

    # Build the command to call UniRig's run.py.
    cmd = [
        unirig_python,
        str(run_py),
        f"--task={skeleton_task}",
        f"--seed={seed}",
        f"--input={input_mesh}",
        f"--output={output_fbx}",
        f"--npz_dir={npz_dir}",
    ]

    _run_logged_subprocess(cmd, cwd=unirig_dir, label="unirig_skeleton")
    
    if not output_fbx.exists():
        raise RuntimeError(f"UniRig did not produce expected output: {output_fbx}")
    
    print(f"[unirig_skeleton] Success! Skeleton saved to: {output_fbx}", flush=True)
    
    return {
        "output_fbx_path": str(output_fbx),
        "seed": seed,
    }


def stage_unirig_skinning(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict skinning weights for a skeleton using UniRig.
    
    Payload:
        input_skeleton_path: Path to skeleton FBX from skeleton stage
        output_fbx_path: Path to output skinned FBX
        npz_dir: Temporary directory for intermediate NPZ files
        seed: Random seed
        skin_task: UniRig config path (default: configs/task/quick_inference_unirig_skin.yaml)
        data_name: NPZ data name (default: raw_data.npz)
    """
    input_skeleton = Path(payload["input_skeleton_path"])
    output_fbx = Path(payload["output_fbx_path"])
    npz_dir = Path(payload["npz_dir"])
    seed = int(payload.get("seed", 12345))
    skin_task = payload.get("skin_task", "configs/task/quick_inference_unirig_skin.yaml")
    data_name = payload.get("data_name", "raw_data.npz")
    faces_target_count = int(payload.get("faces_target_count", 50000))
    
    unirig_dir = APP_DIR / "UniRig"
    run_py = unirig_dir / "run.py"
    
    if not run_py.exists():
        raise FileNotFoundError(f"UniRig run.py not found at: {run_py}")

    unirig_python = _resolve_unirig_python(payload)
    _ensure_unirig_runtime_ready(
        python_exe=unirig_python,
        import_to_package=_UNIRIG_PREDICT_IMPORT_TO_PACKAGE,
        stage_label="unirig_skinning",
    )
    
    if not input_skeleton.exists():
        raise FileNotFoundError(f"Input skeleton not found: {input_skeleton}")
    
    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[unirig_skinning] Input skeleton: {input_skeleton}", flush=True)
    print(f"[unirig_skinning] Output: {output_fbx}", flush=True)
    print(f"[unirig_skinning] Seed: {seed}", flush=True)
    print(f"[unirig_skinning] Python: {unirig_python}", flush=True)

    # UniRig skinning inference also expects precomputed NPZ under npz_dir for this input.
    extract_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    extract_cmd = [
        unirig_python,
        "-m",
        "src.data.extract",
        "--config=configs/data/quick_inference.yaml",
        "--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm",
        "--force_override=true",
        "--num_runs=1",
        "--id=0",
        f"--time={extract_stamp}",
        f"--faces_target_count={faces_target_count}",
        f"--input={input_skeleton}",
        f"--output_dir={npz_dir}",
    ]
    _run_logged_subprocess(extract_cmd, cwd=unirig_dir, label="unirig_extract_skin")

    expected_npz = npz_dir / input_skeleton.stem / data_name
    if not expected_npz.exists():
        # Fallback check in case naming differs in get_files() output mapping.
        any_npz = list(npz_dir.rglob(data_name))
        if not any_npz:
            raise RuntimeError(
                f"UniRig skin extraction finished but produced no {data_name} under {npz_dir}"
            )
    
    cmd = [
        unirig_python,
        str(run_py),
        f"--task={skin_task}",
        f"--seed={seed}",
        f"--input={input_skeleton}",
        f"--output={output_fbx}",
        f"--npz_dir={npz_dir}",
        f"--data_name={data_name}",
    ]

    _run_logged_subprocess(cmd, cwd=unirig_dir, label="unirig_skinning")
    
    if not output_fbx.exists():
        raise RuntimeError(f"UniRig did not produce expected output: {output_fbx}")
    
    print(f"[unirig_skinning] Success! Skinned model saved to: {output_fbx}", flush=True)
    
    return {
        "output_fbx_path": str(output_fbx),
        "seed": seed,
    }


def stage_unirig_merge(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge skeleton/skinning with original mesh using UniRig.
    
    Payload:
        source_path: Path to skeleton or skinned FBX
        target_path: Path to original mesh
        output_path: Path to final rigged output
        export_format: Export format ('fbx' or 'glb')
    """
    source = Path(payload["source_path"])
    target = Path(payload["target_path"])
    output = Path(payload["output_path"])
    export_format = payload.get("export_format", "fbx")
    
    unirig_dir = APP_DIR / "UniRig"
    merge_script = unirig_dir / "src" / "inference" / "merge.py"
    
    if not merge_script.exists():
        raise FileNotFoundError(f"UniRig merge script not found at: {merge_script}")

    unirig_python = _resolve_unirig_python(payload)
    _ensure_unirig_runtime_ready(
        python_exe=unirig_python,
        import_to_package=_UNIRIG_MERGE_IMPORT_TO_PACKAGE,
        stage_label="unirig_merge",
    )
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    
    if not target.exists():
        raise FileNotFoundError(f"Target file not found: {target}")
    
    output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[unirig_merge] Source: {source}", flush=True)
    print(f"[unirig_merge] Target: {target}", flush=True)
    print(f"[unirig_merge] Output: {output}", flush=True)
    print(f"[unirig_merge] Python: {unirig_python}", flush=True)
    
    cmd = [
        unirig_python,
        "-m", "src.inference.merge",
        "--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm",
        f"--source={source}",
        f"--target={target}",
        f"--output={output}",
        "--num_runs=1",
        "--id=0",
    ]

    _run_logged_subprocess(cmd, cwd=unirig_dir, label="unirig_merge")
    
    if not output.exists():
        raise RuntimeError(f"UniRig did not produce expected output: {output}")
    
    print(f"[unirig_merge] Success! Rigged model saved to: {output}", flush=True)
    
    return {
        "output_path": str(output),
        "export_format": export_format,
    }


def stage_unirig_skeleton_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a preview-friendly GLB that includes both mesh + visible skeleton overlay.

    Payload:
        source_npz_path: Path to UniRig raw_data.npz containing vertices/faces/joints/parents
        source_fbx_path: Optional skeleton FBX path used to regenerate NPZ when missing/invalid
        npz_dir: Optional NPZ root dir used with source_fbx_path
        faces_target_count: Optional extraction simplification target when regenerating NPZ
        output_glb_path: Output GLB path
        bone_radius: Optional explicit bone radius
        mesh_alpha: Mesh opacity for preview body in [0,1] (default: 0.5)
        include_mesh: Whether to include the mesh in preview (default: True)
        visibility_boost: Add an outward duplicate skeleton for visibility (default: True)
    """
    import numpy as np
    import trimesh

    source_npz = Path(payload.get("source_npz_path", ""))
    source_fbx_path = payload.get("source_fbx_path")
    npz_dir_path = payload.get("npz_dir")
    faces_target_count = int(payload.get("faces_target_count", 50000))
    output_glb = Path(payload["output_glb_path"])
    explicit_radius = payload.get("bone_radius")
    mesh_alpha = float(payload.get("mesh_alpha", 0.5))
    mesh_alpha = max(0.05, min(0.95, mesh_alpha))
    include_mesh = bool(payload.get("include_mesh", True))
    visibility_boost = bool(payload.get("visibility_boost", True))

    def _load_preview_arrays(npz_path: Path) -> Optional[Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]]:
        if not npz_path.exists():
            return None
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            vertices_local = np.asarray(data["vertices"], dtype=np.float32)
            faces_local = np.asarray(data["faces"], dtype=np.int64)
            joints_blob = data["joints"]
            joints_raw = joints_blob[()] if getattr(joints_blob, "dtype", None) == object and joints_blob.shape == () else joints_blob
            if joints_raw is None:
                return None
            joints_local = np.asarray(joints_raw, dtype=np.float32)
            parents_blob = data["parents"]
            parents_raw = parents_blob[()] if getattr(parents_blob, "dtype", None) == object and parents_blob.shape == () else parents_blob
            if parents_raw is None:
                return None
            parents_local = np.asarray(parents_raw, dtype=object).reshape(-1)
        except Exception:
            return None

        if vertices_local.ndim != 2 or vertices_local.shape[1] != 3:
            return None
        if faces_local.ndim != 2 or faces_local.shape[1] != 3:
            return None
        if joints_local.ndim != 2 or joints_local.shape[1] != 3:
            return None
        if len(joints_local) == 0 or len(parents_local) == 0:
            return None
        return vertices_local, faces_local, joints_local, parents_local

    loaded = _load_preview_arrays(source_npz)

    # Fresh skeleton runs don't always have the skeleton NPZ yet.
    # Regenerate it from the produced skeleton FBX on demand.
    if loaded is None and source_fbx_path and npz_dir_path:
        source_fbx = Path(source_fbx_path)
        npz_dir = Path(npz_dir_path)
        if source_fbx.exists():
            print(
                f"[unirig_skeleton_preview] Source NPZ missing/invalid, regenerating from skeleton FBX: {source_fbx}",
                flush=True,
            )
            unirig_dir = APP_DIR / "UniRig"
            unirig_python = _resolve_unirig_python(payload)
            _ensure_unirig_runtime_ready(
                python_exe=unirig_python,
                import_to_package=_UNIRIG_PREDICT_IMPORT_TO_PACKAGE,
                stage_label="unirig_skeleton_preview_extract",
            )
            npz_dir.mkdir(parents=True, exist_ok=True)
            extract_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            extract_cmd = [
                unirig_python,
                "-m",
                "src.data.extract",
                "--config=configs/data/quick_inference.yaml",
                "--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm",
                "--force_override=true",
                "--num_runs=1",
                "--id=0",
                f"--time={extract_stamp}",
                f"--faces_target_count={faces_target_count}",
                f"--input={source_fbx}",
                f"--output_dir={npz_dir}",
            ]
            _run_logged_subprocess(extract_cmd, cwd=unirig_dir, label="unirig_extract_preview")

            candidates: List[Path] = [npz_dir / source_fbx.stem / "raw_data.npz"]
            try:
                dynamic = sorted(
                    npz_dir.rglob("raw_data.npz"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                candidates.extend(dynamic)
            except Exception:
                pass

            seen: Set[str] = set()
            for candidate in candidates:
                key = str(candidate.resolve()) if candidate.exists() else str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                probe = _load_preview_arrays(candidate)
                if probe is not None:
                    source_npz = candidate
                    loaded = probe
                    break

    if loaded is None:
        raise FileNotFoundError(
            f"Skeleton preview source npz not found or invalid: {source_npz}. "
            "Tried regenerating from skeleton FBX when available."
        )

    vertices, faces, joints, parents = loaded

    print(f"[unirig_skeleton_preview] Source NPZ: {source_npz}", flush=True)
    print(f"[unirig_skeleton_preview] Output GLB: {output_glb}", flush=True)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Invalid vertices in source npz.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Invalid faces in source npz.")
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("Invalid joints in source npz.")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Use explicit PBR alpha blending so the body is truly translucent in GLB viewers.
    mesh_mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0.58, 0.62, 0.72, mesh_alpha],
        metallicFactor=0.0,
        roughnessFactor=0.92,
        alphaMode="BLEND",
        doubleSided=True,
    )
    mesh_uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=mesh_uv, material=mesh_mat)

    if explicit_radius is None:
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        diag = float(np.linalg.norm(bounds_max - bounds_min))
        # Boost default radius so skeleton stays visible on dense characters.
        bone_radius = max(0.008, min(0.08, diag * 0.012))
    else:
        bone_radius = float(explicit_radius)

    center = vertices.mean(axis=0)

    def _outward_shift(point: "np.ndarray", strength: float) -> "np.ndarray":
        vec = point - center
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float32)
        return (vec / norm) * float(strength)

    scene_meshes = [mesh] if include_mesh else []
    bone_count = 0

    for idx, parent in enumerate(parents):
        if parent is None:
            continue
        try:
            parent_idx = int(parent)
        except Exception:
            continue
        if parent_idx < 0 or parent_idx >= len(joints):
            continue

        start = joints[parent_idx]
        end = joints[idx]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue

        cyl = trimesh.creation.cylinder(radius=bone_radius, height=length, sections=14)
        transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vec / length)
        if transform is None:
            transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = (start + end) * 0.5
        cyl.apply_transform(transform)
        cyl.visual.face_colors = [255, 52, 38, 255]
        scene_meshes.append(cyl)

        if visibility_boost:
            mid = (start + end) * 0.5
            shift = _outward_shift(mid, bone_radius * 2.75)
            start_b = start + shift
            end_b = end + shift
            vec_b = end_b - start_b
            len_b = float(np.linalg.norm(vec_b))
            if len_b > 1e-6:
                cyl_b = trimesh.creation.cylinder(radius=bone_radius * 0.9, height=len_b, sections=12)
                transform_b = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vec_b / len_b)
                if transform_b is None:
                    transform_b = np.eye(4, dtype=np.float32)
                transform_b[:3, 3] = (start_b + end_b) * 0.5
                cyl_b.apply_transform(transform_b)
                cyl_b.visual.face_colors = [0, 255, 255, 255]
                scene_meshes.append(cyl_b)
        bone_count += 1

    joint_radius = bone_radius * 1.6
    for joint in joints:
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=joint_radius)
        sphere.apply_translation(joint)
        sphere.visual.face_colors = [255, 210, 28, 255]
        scene_meshes.append(sphere)

        if visibility_boost:
            shift = _outward_shift(joint, bone_radius * 2.9)
            sphere_b = trimesh.creation.icosphere(subdivisions=1, radius=joint_radius * 0.95)
            sphere_b.apply_translation(joint + shift)
            sphere_b.visual.face_colors = [255, 0, 255, 255]
            scene_meshes.append(sphere_b)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene(scene_meshes)
    scene.export(str(output_glb))

    if not output_glb.exists():
        raise RuntimeError(f"Skeleton preview export failed: {output_glb}")

    print(
        f"[unirig_skeleton_preview] Success! Bones: {bone_count}, Joints: {len(joints)}, Output: {output_glb}",
        flush=True,
    )
    return {
        "preview_glb_path": str(output_glb),
        "bone_count": int(bone_count),
        "joint_count": int(len(joints)),
    }


def stage_unirig_animation_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a procedural animation preview GLB from a rigged model.

    Payload:
        input_model_path: Path to rigged model (.fbx/.glb/.gltf)
        input_fbx_path: Backward-compatible alias for FBX source
        output_glb_path: Output animated GLB path
        frame_end: End frame for looping animation (default: 120)
        animation_style: walk|dance|idle (default: dance)
        animation_strength: multiplier for animation intensity (default: 1.35)
    """
    import hashlib

    blender_user_root = APP_DIR / "tmp" / "blender_user"
    blender_user_config = blender_user_root / "config"
    blender_user_scripts = blender_user_root / "scripts"
    blender_user_data = blender_user_root / "datafiles"
    blender_user_config.mkdir(parents=True, exist_ok=True)
    blender_user_scripts.mkdir(parents=True, exist_ok=True)
    blender_user_data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BLENDER_USER_CONFIG", str(blender_user_config))
    os.environ.setdefault("BLENDER_USER_SCRIPTS", str(blender_user_scripts))
    os.environ.setdefault("BLENDER_USER_DATAFILES", str(blender_user_data))

    import bpy

    input_model_value = payload.get("input_model_path") or payload.get("input_fbx_path")
    if not input_model_value:
        raise ValueError("Animation preview requires 'input_model_path' (or legacy 'input_fbx_path').")

    input_model = Path(input_model_value)
    output_glb = Path(payload["output_glb_path"])
    frame_end = int(payload.get("frame_end", 120))
    frame_end = max(frame_end, 30)
    style = str(payload.get("animation_style", "dance")).strip().lower()
    if style not in {"walk", "dance", "idle"}:
        style = "dance"
    strength = float(payload.get("animation_strength", 1.35))
    strength = max(0.2, min(2.5, strength))

    if not input_model.exists():
        raise FileNotFoundError(f"Animation preview source model not found: {input_model}")

    print(f"[unirig_animation_preview] Input model: {input_model}", flush=True)
    print(f"[unirig_animation_preview] Output GLB: {output_glb}", flush=True)
    print(f"[unirig_animation_preview] Style: {style}", flush=True)
    print(f"[unirig_animation_preview] Strength: {strength}", flush=True)

    # Reset to an empty scene for deterministic exports.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    suffix = input_model.suffix.lower()
    if suffix == ".fbx":
        ret = bpy.ops.import_scene.fbx(filepath=str(input_model))
        if "FINISHED" not in ret:
            raise RuntimeError(f"FBX import failed for animation preview: {input_model}")
    elif suffix in {".glb", ".gltf"}:
        ret = bpy.ops.import_scene.gltf(filepath=str(input_model))
        if "FINISHED" not in ret:
            raise RuntimeError(f"GLTF import failed for animation preview: {input_model}")
    else:
        raise ValueError(f"Unsupported animation preview input format: {suffix}")

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_end

    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found in imported model; cannot build animation preview.")

    def _name_has(name: str, *tokens: str) -> bool:
        return any(tok in name for tok in tokens)

    def _bone_side(name: str) -> int:
        left_hits = (
            ".l",
            "_l",
            "left",
            " l_",
            "l_",
            "hand_l",
            "foot_l",
            "arm_l",
            "leg_l",
            "thigh_l",
        )
        right_hits = (
            ".r",
            "_r",
            "right",
            " r_",
            "r_",
            "hand_r",
            "foot_r",
            "arm_r",
            "leg_r",
            "thigh_r",
        )
        if _name_has(name, *left_hits):
            return -1
        if _name_has(name, *right_hits):
            return 1
        return 0

    import math

    # Shared built-in clip phases. This keeps behavior deterministic across machines.
    phase_keys = [i / 8.0 for i in range(9)]  # 0.00 .. 1.00
    frame_keys = [1 + int(round(ph * (frame_end - 1))) for ph in phase_keys]

    def _bone_role(name_l: str) -> str:
        if _name_has(name_l, "pelvis", "hip"):
            return "pelvis"
        if _name_has(name_l, "spine", "chest", "torso"):
            return "spine"
        if _name_has(name_l, "neck", "head"):
            return "head"
        if _name_has(name_l, "thigh", "upleg", "upperleg"):
            return "leg_upper"
        if _name_has(name_l, "shin", "calf", "lowerleg", "knee"):
            return "leg_lower"
        if _name_has(name_l, "foot", "ankle", "toe"):
            return "foot"
        if _name_has(name_l, "shoulder", "upperarm", "arm"):
            return "arm_upper"
        if _name_has(name_l, "forearm", "lowerarm", "elbow"):
            return "arm_lower"
        if _name_has(name_l, "hand", "wrist", "finger", "thumb"):
            return "hand"
        return "other"

    def _rot_for_phase(style_name: str, role: str, side: int, phase: float, seed_hash: bytes) -> Tuple[float, float, float]:
        # left/right opposite timing
        side_phase = phase + (0.5 if side > 0 else 0.0)
        w = math.sin(2.0 * math.pi * side_phase)
        w2 = math.sin(4.0 * math.pi * side_phase)
        c = math.cos(2.0 * math.pi * phase)

        # deterministic fallback sign for non-sided bones
        fallback_sign = 1.0 if (seed_hash[2] % 2) == 0 else -1.0
        side_sign = float(side) if side != 0 else fallback_sign

        # base amplitudes in radians
        if style_name == "idle":
            if role in {"pelvis", "spine", "head"}:
                return (0.03 * c, 0.02 * w, 0.015 * w2)
            if role in {"arm_upper", "arm_lower", "hand"}:
                return (0.02 * side_sign * c, 0.0, 0.03 * side_sign * w)
            if role in {"leg_upper", "leg_lower", "foot"}:
                return (0.015 * side_sign * w, 0.0, 0.0)
            return (0.008 * fallback_sign * w, 0.008 * c, 0.008 * w2)

        if style_name == "walk":
            if role == "pelvis":
                return (0.04 * c, 0.02 * w, 0.04 * c)
            if role == "spine":
                return (0.03 * c, 0.02 * w, 0.08 * c)
            if role == "head":
                return (0.02 * c, 0.03 * w, 0.02 * c)
            if role == "leg_upper":
                return (0.55 * w, 0.02 * side_sign * c, 0.0)
            if role == "leg_lower":
                knee = max(0.0, -w)
                return (0.70 * knee, 0.0, 0.0)
            if role == "foot":
                toe = max(0.0, w)
                return (-0.28 * w + 0.12 * toe, 0.0, 0.0)
            if role == "arm_upper":
                return (-0.42 * w, 0.03 * side_sign * c, 0.10 * side_sign * c)
            if role == "arm_lower":
                elbow = max(0.0, w)
                return (0.18 * elbow, 0.0, 0.0)
            if role == "hand":
                return (0.10 * w, 0.0, 0.0)
            return (0.03 * fallback_sign * w, 0.02 * c, 0.02 * w2)

        # style == "dance"
        if role == "pelvis":
            return (0.10 * c, 0.05 * w, 0.18 * w2)
        if role == "spine":
            return (0.12 * c, 0.07 * w, 0.22 * w2)
        if role == "head":
            return (0.07 * c, 0.10 * w, 0.04 * c)
        if role == "leg_upper":
            return (0.85 * w, 0.05 * side_sign * c, 0.18 * w2)
        if role == "leg_lower":
            knee = max(0.0, -w)
            return (0.95 * knee, 0.0, 0.0)
        if role == "foot":
            toe = max(0.0, w)
            return (-0.42 * w + 0.22 * toe, 0.0, 0.0)
        if role == "arm_upper":
            return (-0.95 * w, 0.15 * side_sign * c, 0.35 * side_sign * w2)
        if role == "arm_lower":
            return (0.45 * max(0.0, w), 0.05 * side_sign * c, 0.0)
        if role == "hand":
            return (0.25 * w2, 0.0, 0.0)
        return (0.10 * fallback_sign * w, 0.06 * c, 0.06 * w2)

    def _root_loc(style_name: str, phase: float) -> Tuple[float, float, float]:
        if style_name == "idle":
            return (
                0.0,
                0.004 * math.sin(2.0 * math.pi * phase + 0.35),
                0.01 * math.sin(2.0 * math.pi * phase),
            )
        if style_name == "walk":
            return (
                0.01 * math.sin(2.0 * math.pi * phase),
                0.015 * math.sin(2.0 * math.pi * phase + 0.35),
                0.02 * math.sin(4.0 * math.pi * phase),
            )
        # dance
        return (
            0.03 * math.sin(2.0 * math.pi * phase),
            0.03 * math.sin(2.0 * math.pi * phase + 0.45),
            0.04 * math.sin(4.0 * math.pi * phase),
        )

    def _root_rot(style_name: str, phase: float) -> Tuple[float, float, float]:
        w = math.sin(2.0 * math.pi * phase)
        w2 = math.sin(4.0 * math.pi * phase)
        c = math.cos(2.0 * math.pi * phase)
        if style_name == "idle":
            return (0.01 * w, 0.005 * w2, 0.02 * c)
        if style_name == "walk":
            return (0.04 * w, 0.015 * w2, 0.07 * c)
        # dance
        return (0.08 * w, 0.05 * w2, 0.20 * c)

    animated_bones = 0

    for armature in armatures:
        # Drop imported animation tracks and apply shared clip to this rig.
        try:
            armature.animation_data_clear()
        except Exception:
            pass

        bpy.context.view_layer.objects.active = armature
        try:
            bpy.ops.object.mode_set(mode="POSE")
        except Exception:
            pass

        for pose_bone in armature.pose.bones:
            if pose_bone.parent is None:
                continue

            lname = pose_bone.name.lower()
            side = _bone_side(lname)
            role = _bone_role(lname)
            seed = hashlib.sha1(pose_bone.name.encode("utf-8")).digest()
            pose_bone.rotation_mode = "XYZ"

            for frame, phase in zip(frame_keys, phase_keys):
                rx, ry, rz = _rot_for_phase(style, role, side, phase, seed)
                pose_bone.rotation_euler = (rx * strength, ry * strength, rz * strength)
                pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)
            animated_bones += 1

        root_bones = [pb for pb in armature.pose.bones if pb.parent is None]
        if root_bones:
            root = root_bones[0]
            try:
                root.rotation_mode = "XYZ"
                for frame, phase in zip(frame_keys, phase_keys):
                    lx, ly, lz = _root_loc(style, phase)
                    rx, ry, rz = _root_rot(style, phase)
                    root.location = (lx * strength, ly, lz * strength)
                    root.rotation_euler = (rx * strength, ry * strength, rz * strength)
                    root.keyframe_insert(data_path="location", frame=frame)
                    root.keyframe_insert(data_path="rotation_euler", frame=frame)
                animated_bones += 1
            except Exception:
                pass

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    if animated_bones == 0:
        raise RuntimeError("No animatable bones were found for preview animation.")

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    ret = bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        use_selection=False,
        export_animations=True,
        export_force_sampling=True,
        export_frame_range=True,
    )
    if "FINISHED" not in ret:
        raise RuntimeError(f"GLB export failed for animation preview: {output_glb}")

    if not output_glb.exists():
        raise RuntimeError(f"Animation preview export failed: {output_glb}")

    print(
        f"[unirig_animation_preview] Success! Armatures: {len(armatures)}, Animated bones: {animated_bones}, Output: {output_glb}",
        flush=True,
    )
    return {
        "animation_preview_glb_path": str(output_glb),
        "armature_count": int(len(armatures)),
        "animated_bones": int(animated_bones),
        "frame_end": int(frame_end),
        "animation_style": style,
        "animation_strength": float(strength),
    }


# ================================ Main ================================


def main() -> int:
    parser = argparse.ArgumentParser(description="TRELLIS.2 subprocess stage runner")
    parser.add_argument("--stage", required=True, type=str)
    parser.add_argument("--payload", required=True, type=str)
    parser.add_argument("--result", required=True, type=str)
    args = parser.parse_args()

    try:
        stage = args.stage.strip()
        if stage not in {
            "unirig_skeleton",
            "unirig_skinning",
            "unirig_merge",
            "unirig_skeleton_preview",
            "unirig_animation_preview",
        }:
            _ensure_o_voxel_available()

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
        elif stage == "unirig_skeleton":
            result = stage_unirig_skeleton(payload)
        elif stage == "unirig_skinning":
            result = stage_unirig_skinning(payload)
        elif stage == "unirig_merge":
            result = stage_unirig_merge(payload)
        elif stage == "unirig_skeleton_preview":
            result = stage_unirig_skeleton_preview(payload)
        elif stage == "unirig_animation_preview":
            result = stage_unirig_animation_preview(payload)
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


