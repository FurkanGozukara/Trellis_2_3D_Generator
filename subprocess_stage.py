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


def _pipeline_type_from_resolution(resolution: str) -> str:
    # Match app_premium mapping.
    return {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
        "2048": "2048_cascade",
    }[resolution]


def _ss_res_from_pipeline_type(pipeline_type: str) -> int:
    return {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32, "2048_cascade": 32}[pipeline_type]


def _target_res_from_pipeline_type(pipeline_type: str) -> int:
    return {"1024_cascade": 1024, "1536_cascade": 1536, "2048_cascade": 2048}[pipeline_type]


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
    image_path = Path(payload["image_path"])
    resolution = str(payload["resolution"])
    pipeline_type = _pipeline_type_from_resolution(resolution)

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
    pipe.cuda()

    print("[cond] computing image embeddings (512px)…", flush=True)
    cond_512 = pipe.get_cond([img], 512)
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
    resolution = str(payload["resolution"])
    pipeline_type = _pipeline_type_from_resolution(resolution)
    ss_res = _ss_res_from_pipeline_type(pipeline_type)

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
    pipe.cuda()

    print("[sparse] loading cond_512…", flush=True)
    cond = _load_cond(cond_512_path, device=device)

    print(f"[sparse] sampling sparse structure (ss_res={ss_res})…", flush=True)
    coords = pipe.sample_sparse_structure(cond, ss_res, 1, ss_params)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(coords.detach().cpu(), str(coords_path))
    print(f"[sparse] saved coords: {coords_path}", flush=True)
    return {"coords_path": str(coords_path)}


def stage_sample_shape_slat(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    resolution = str(payload["resolution"])
    pipeline_type = _pipeline_type_from_resolution(resolution)
    shape_params = payload["shape_params"]
    max_num_tokens = int(payload.get("max_num_tokens", 49152))

    cond_512_path = Path(payload["cond_512_path"])
    cond_1024_path = Path(payload["cond_1024_path"]) if payload.get("cond_1024_path") else None
    coords_path = Path(payload["coords_path"])
    shape_slat_path = Path(payload["shape_slat_path"])
    out_res_path = Path(payload["out_res_path"])

    device = "cuda"

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

    elif pipeline_type in {"1024_cascade", "1536_cascade", "2048_cascade"}:
        if cond_1024_path is None:
            raise ValueError("cond_1024_path is required for cascade pipeline types.")
        target_res = _target_res_from_pipeline_type(pipeline_type)

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
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

    # Persist as npz (portable, easy to inspect).
    _save_npz_sparse(shape_slat_path, slat.feats, slat.coords)
    _write_json(out_res_path, {"res": int(res), "pipeline_type": pipeline_type})
    print(f"[shape] saved: {shape_slat_path}", flush=True)
    print(f"[shape] saved: {out_res_path} (res={res})", flush=True)

    return {"shape_slat_path": str(shape_slat_path), "res": int(res), "pipeline_type": pipeline_type}


def stage_sample_tex_slat(payload: Dict[str, Any]) -> Dict[str, Any]:
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    resolution = str(payload["resolution"])
    pipeline_type = _pipeline_type_from_resolution(resolution)

    cond_path = Path(payload["cond_path"])
    shape_slat_path = Path(payload["shape_slat_path"])
    tex_slat_path = Path(payload["tex_slat_path"])
    tex_params = payload["tex_params"]

    device = "cuda"

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
    pipe.cuda()

    cond = _load_cond(cond_path, device=device)

    print(f"[tex] sampling texture SLat ({flow_key})…", flush=True)
    tex_slat = pipe.sample_tex_slat(cond, pipe.models[flow_key], shape_slat, tex_params)
    _save_npz_sparse(tex_slat_path, tex_slat.feats, tex_slat.coords)
    print(f"[tex] saved: {tex_slat_path}", flush=True)
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
    shape_slat_path = Path(payload["shape_slat_path"])
    tex_slat_path = Path(payload["tex_slat_path"]) if payload.get("tex_slat_path") else None
    res = int(payload["res"])
    preview_dir = Path(payload["preview_dir"])
    manifest_path = Path(payload["preview_manifest_path"])

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
    pipe.cuda()

    print("[preview] decoding latent to mesh…", flush=True)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]

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
    shape_slat_path = Path(payload["shape_slat_path"])
    tex_slat_path = Path(payload["tex_slat_path"]) if payload.get("tex_slat_path") else None
    res = int(payload["res"])

    decimation_target = int(payload["decimation_target"])
    texture_size = int(payload["texture_size"])
    remesh_method = str(payload["remesh_method"])
    simplify_method = str(payload["simplify_method"])
    prune_invisible_faces = bool(payload["prune_invisible_faces"])
    no_texture_gen = bool(payload["no_texture_gen"])

    out_dir = Path(payload["out_dir"])
    prefix = str(payload.get("prefix", "glb"))

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
    pipe.cuda()

    print("[extract] decoding latent to mesh…", flush=True)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]

    print("[extract] converting to GLB…", flush=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipe.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        simplify_method=simplify_method,
        texture_extraction=not no_texture_gen,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        remesh_method=remesh_method,
        prune_invisible=prune_invisible_faces,
        use_tqdm=True,
    )

    _, glb_path = next_indexed_path(out_dir, prefix=prefix, ext="glb", digits=4, start=1)
    glb.export(str(glb_path), extension_webp=True)
    torch.cuda.empty_cache()
    print(f"[extract] saved: {glb_path}", flush=True)
    return {"glb_path": str(glb_path)}


def stage_texture_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    import trimesh
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2TexturingPipeline

    from subprocess_utils import next_indexed_path

    model_repo = payload.get("model_repo", "microsoft/TRELLIS.2-4B")
    config_file = payload.get("config_file", "texturing_pipeline.json")

    mesh_path = Path(payload["mesh_path"])
    image_path = Path(payload["image_path"])
    seed = int(payload["seed"])
    resolution = int(payload["resolution"])
    texture_size = int(payload["texture_size"])
    tex_params = payload["tex_params"]

    out_dir = Path(payload["out_dir"])
    prefix = str(payload.get("prefix", "textured"))

    print("[texturing] loading texturing pipeline…", flush=True)
    pipe = Trellis2TexturingPipeline.from_pretrained(model_repo, config_file=config_file)
    pipe.cuda()

    print("[texturing] loading mesh…", flush=True)
    mesh = trimesh.load(str(mesh_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    print("[texturing] loading reference image…", flush=True)
    img = Image.open(str(image_path))

    print("[texturing] preprocessing image…", flush=True)
    img = pipe.preprocess_image(img)

    print("[texturing] preprocessing mesh…", flush=True)
    mesh = pipe.preprocess_mesh(mesh)

    torch.manual_seed(seed)
    print(f"[texturing] computing image embeddings ({512 if resolution == 512 else 1024}px)…", flush=True)
    cond = pipe.get_cond([img], 512) if resolution == 512 else pipe.get_cond([img], 1024)

    print("[texturing] encoding mesh to shape latent…", flush=True)
    shape_slat = pipe.encode_shape_slat(mesh, resolution)

    tex_model_key = "tex_slat_flow_model_512" if resolution == 512 else "tex_slat_flow_model_1024"
    print("[texturing] sampling texture latent…", flush=True)
    tex_slat = pipe.sample_tex_slat(cond, pipe.models[tex_model_key], shape_slat, tex_params)

    print("[texturing] decoding texture latent…", flush=True)
    pbr_voxel = pipe.decode_tex_slat(tex_slat)

    print("[texturing] baking textures onto mesh…", flush=True)
    out_mesh = pipe.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size)

    _, out_path = next_indexed_path(out_dir, prefix=prefix, ext="glb", digits=4, start=1)
    out_mesh.export(str(out_path), extension_webp=True)
    torch.cuda.empty_cache()
    print(f"[texturing] saved: {out_path}", flush=True)
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
        elif stage == "texture_generate":
            result = stage_texture_generate(payload)
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


