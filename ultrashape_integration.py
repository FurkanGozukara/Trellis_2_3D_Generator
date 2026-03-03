from __future__ import annotations

import inspect
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import trimesh
from PIL import Image


def _torch_or_numpy_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _as_trimesh(mesh_obj) -> trimesh.Trimesh:
    """
    Convert TRELLIS mesh-like objects (e.g. MeshWithVoxel) into a plain trimesh.
    UltraShape surface loaders operate on trimesh geometry APIs.
    """
    if isinstance(mesh_obj, trimesh.Scene):
        return mesh_obj.to_mesh()
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj

    if not hasattr(mesh_obj, "vertices") or not hasattr(mesh_obj, "faces"):
        raise TypeError(f"Unsupported mesh type for UltraShape: {type(mesh_obj).__name__}")

    v_np = _torch_or_numpy_to_numpy(mesh_obj.vertices)
    f_np = _torch_or_numpy_to_numpy(mesh_obj.faces)
    return trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)


def _candidate_ultrashape_roots(app_dir: Path) -> list[Path]:
    return [
        app_dir / "UltraShape-1.0",
        app_dir / "ComfyUI-UltraShape1" / "UltraShape-1.0",
        app_dir / "UltraShape_v2",
    ]


def resolve_ultrashape_root(app_dir: Path) -> Path:
    for root in _candidate_ultrashape_roots(app_dir):
        if (root / "ultrashape").is_dir() and (root / "configs").is_dir():
            return root
    raise FileNotFoundError(
        "UltraShape source not found. Expected one of: "
        + ", ".join(str(p) for p in _candidate_ultrashape_roots(app_dir))
    )


def resolve_ultrashape_checkpoint(models_dir: Path, checkpoint: str = "") -> Path:
    base = models_dir / "UltraShape"
    if checkpoint:
        ckpt = Path(checkpoint)
        if ckpt.is_file():
            return ckpt
        ckpt2 = base / checkpoint
        if ckpt2.is_file():
            return ckpt2
        raise FileNotFoundError(f"UltraShape checkpoint not found: {checkpoint}")

    if not base.is_dir():
        raise FileNotFoundError(f"UltraShape model dir not found: {base}")

    candidates = []
    for ext in ("*.pt", "*.ckpt", "*.safetensors"):
        candidates.extend(sorted(base.glob(ext)))
    if not candidates:
        raise FileNotFoundError(
            f"No UltraShape checkpoints found in {base}. "
            "Put your .pt/.ckpt/.safetensors files there."
        )
    return candidates[0]


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_torch_ckpt(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def refine_mesh_with_ultrashape(
    mesh: trimesh.Trimesh,
    image_path: str,
    app_dir: str,
    models_dir: str,
    *,
    checkpoint: str = "",
    config_name: str = "infer_dit_refine.yaml",
    dtype: str = "bfloat16",
    low_vram: bool = True,
    steps: int = 50,
    guidance_scale: float = 5.0,
    octree_resolution: int = 384,
    num_chunks: int = 8000,
    mc_level: float = 0.0,
    box_v: float = 1.0,
    seed: int = 42,
    remove_bg: bool = False,
    normalize_scale: float = 0.99,
    num_sharp_points: int = 204800,
    num_uniform_points: int = 204800,
    num_latents: int = 0,
    target_face_count: int = 500000,
    enable_pbar: bool = False,
) -> trimesh.Trimesh:
    def _ulog(msg: str) -> None:
        print(f"[ultrashape] {msg}", flush=True)

    app_root = Path(app_dir).resolve()
    models_root = Path(models_dir).resolve()
    _ulog("Resolving UltraShape paths/config...")

    ultrashape_root = resolve_ultrashape_root(app_root)
    ckpt_path = resolve_ultrashape_checkpoint(models_root, checkpoint=checkpoint)
    config_path = ultrashape_root / "configs" / config_name
    if not config_path.is_file():
        raise FileNotFoundError(f"UltraShape config not found: {config_path}")
    _ulog(f"Using source: {ultrashape_root}")
    _ulog(f"Using checkpoint: {ckpt_path.name}")

    # Ensure the local UltraShape package is importable.
    ul_root_str = str(ultrashape_root)
    if ul_root_str not in sys.path:
        sys.path.insert(0, ul_root_str)

    try:
        from omegaconf import OmegaConf
        from ultrashape.pipelines import UltraShapePipeline
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils.misc import instantiate_from_config
        from ultrashape.utils import voxelize_from_point
    except ModuleNotFoundError as e:
        missing_mod = getattr(e, "name", None) or "unknown"
        pip_name = "scikit-image" if missing_mod == "skimage" else missing_mod
        req_path = ultrashape_root / "requirements.txt"
        raise RuntimeError(
            f"UltraShape dependency import failed (missing module: {missing_mod}). "
            f"Install in this venv (example: `python -m pip install {pip_name}`), then run "
            f"`python -m pip install -r \"{req_path}\"`."
        ) from e

    cfg = OmegaConf.load(str(config_path))
    params = cfg.model.params
    _ulog(f"Loaded config: {config_name}")

    # Prefer local DINOv2 weights if available.
    local_dino = models_root / "dinov2-large"
    try:
        if local_dino.is_dir():
            params.conditioner_config.params.main_image_encoder.kwargs.version = str(local_dino)
    except Exception:
        pass

    vae = instantiate_from_config(params.vae_config)
    dit = instantiate_from_config(params.dit_cfg)
    conditioner = instantiate_from_config(params.conditioner_config)
    scheduler = instantiate_from_config(params.scheduler_cfg)
    image_processor = instantiate_from_config(params.image_processor_cfg)
    _ulog("Model modules instantiated.")

    _ulog("Loading checkpoint weights...")
    weights = _load_torch_ckpt(ckpt_path)
    vae.load_state_dict(weights["vae"], strict=True)
    dit.load_state_dict(weights["dit"], strict=True)
    conditioner.load_state_dict(weights["conditioner"], strict=True)
    _ulog("Checkpoint weights loaded.")

    torch_dtype = _dtype_from_name(dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae.eval().to(device, dtype=torch_dtype)
    dit.eval().to(device, dtype=torch_dtype)
    conditioner.eval().to(device, dtype=torch_dtype)
    _ulog(f"Models moved to {device} ({torch_dtype}).")

    if hasattr(vae, "enable_flashvdm_decoder"):
        try:
            vae.enable_flashvdm_decoder()
        except Exception:
            pass

    pipeline = UltraShapePipeline(
        vae=vae,
        model=dit,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
    )

    if low_vram and hasattr(pipeline, "enable_model_cpu_offload") and device.startswith("cuda"):
        try:
            pipeline.enable_model_cpu_offload()
            _ulog("CPU offload enabled for low-VRAM mode.")
        except Exception:
            # Fallback to regular in-device mode if accelerate offload is unavailable.
            pipeline.to(device=device, dtype=torch_dtype)
            _ulog("CPU offload unavailable; using standard device mode.")

    # Preserve voxel fields when input is TRELLIS MeshWithVoxel, so callers can
    # still pass the refined mesh directly into o_voxel.postprocess.to_glb().
    has_voxel_fields = (
        hasattr(mesh, "attrs")
        and hasattr(mesh, "coords")
        and hasattr(mesh, "origin")
        and hasattr(mesh, "voxel_size")
        and hasattr(mesh, "voxel_shape")
    )
    voxel_payload = None
    if has_voxel_fields:
        voxel_payload = {
            "attrs": mesh.attrs,
            "coords": mesh.coords,
            "origin": mesh.origin,
            "voxel_size": mesh.voxel_size,
            "voxel_shape": mesh.voxel_shape,
            "layout": getattr(mesh, "layout", {}),
        }

    mesh_tri = _as_trimesh(mesh)
    _ulog(f"Input mesh prepared (v={len(mesh_tri.vertices)}, f={len(mesh_tri.faces)}).")

    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=int(num_sharp_points),
        num_uniform_points=int(num_uniform_points),
    )
    _ulog(
        f"Sampling mesh surface points (uniform={int(num_uniform_points)}, sharp={int(num_sharp_points)})..."
    )
    surface = loader(mesh_tri.copy(), normalize_scale=float(normalize_scale))
    surface = surface.to(device=device, dtype=torch_dtype)
    pc = surface[:, :, :3]
    _ulog("Surface sampling complete.")

    token_num_cfg = int(params.vae_config.params.num_latents)
    token_num = int(num_latents) if int(num_latents) > 0 else token_num_cfg
    voxel_res = int(params.vae_config.params.voxel_query_res)
    _ulog(f"Voxelizing points (target_tokens={token_num}, resolution={voxel_res})...")
    _, voxel_idx = voxelize_from_point(pc, token_num, resolution=voxel_res)
    try:
        unique_voxels = int(torch.unique(voxel_idx[0], dim=0).shape[0])
        total_tokens = int(voxel_idx.shape[1])
        if unique_voxels < total_tokens:
            _ulog(
                f"Voxel tokens prepared: {unique_voxels}/{total_tokens} unique; repeated tokens were used to fill target length."
            )
        else:
            _ulog(f"Voxel tokens prepared: {total_tokens}/{total_tokens} unique.")
    except Exception:
        pass

    ref = Image.open(str(image_path))
    _ulog("Reference image loaded.")
    if remove_bg:
        try:
            # Reuse TRELLIS rembg backend for consistency with Image -> 3D preprocessing.
            from trellis2.pipelines.rembg import BiRefNet

            rembg_model = BiRefNet()
            rembg_model.to(device)
            ref = rembg_model(ref.convert("RGB"))
            _ulog("Background removal applied to reference image.")
        except Exception:
            pass
    if ref.mode != "RGBA":
        ref = ref.convert("RGBA")

    call_kwargs = {
        "image": ref,
        "voxel_cond": voxel_idx,
        "generator": torch.Generator(device=device).manual_seed(int(seed)),
        "box_v": float(box_v),
        "mc_level": float(mc_level),
        "octree_resolution": int(octree_resolution),
        "num_inference_steps": int(steps),
        "num_chunks": int(num_chunks),
        "guidance_scale": float(guidance_scale),
        "enable_pbar": bool(enable_pbar),
    }
    sig = inspect.signature(pipeline.__call__)
    if "target_face_count" in sig.parameters:
        call_kwargs["target_face_count"] = int(target_face_count)

    autocast_enabled = device.startswith("cuda") and torch_dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch_dtype) if autocast_enabled else nullcontext()
    _ulog(
        f"Starting UltraShape diffusion sampling (steps={int(steps)}, guidance={float(guidance_scale):.2f})..."
    )
    sample_start = time.time()
    hb_stop = threading.Event()

    def _heartbeat() -> None:
        while not hb_stop.wait(6.0):
            elapsed = int(time.time() - sample_start)
            _ulog(f"Sampling in progress... {elapsed}s elapsed.")

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()
    with autocast_ctx:
        try:
            result = pipeline(**call_kwargs)
        finally:
            hb_stop.set()
            hb_thread.join(timeout=0.2)
    _ulog(f"Sampling finished in {time.time() - sample_start:.1f}s.")

    if isinstance(result, tuple):
        result = result[0]
    refined = result[0] if isinstance(result, list) else result
    if isinstance(refined, trimesh.Scene):
        refined = refined.to_mesh()

    if voxel_payload is not None:
        from trellis2.representations import MeshWithVoxel

        attrs = voxel_payload["attrs"]
        coords = voxel_payload["coords"]
        target_device = attrs.device if torch.is_tensor(attrs) else torch.device("cpu")

        refined_vertices = torch.as_tensor(
            refined.vertices,
            dtype=torch.float32,
            device=target_device,
        )
        refined_faces = torch.as_tensor(
            refined.faces,
            dtype=torch.int32,
            device=target_device,
        )
        origin = voxel_payload["origin"]
        if torch.is_tensor(origin):
            origin = origin.detach().cpu().tolist()

        refined = MeshWithVoxel(
            vertices=refined_vertices,
            faces=refined_faces,
            origin=origin,
            voxel_size=float(voxel_payload["voxel_size"]),
            coords=coords,
            attrs=attrs,
            voxel_shape=voxel_payload["voxel_shape"],
            layout=voxel_payload.get("layout", {}),
        )
        _ulog("Refined mesh converted back to TRELLIS MeshWithVoxel.")
    else:
        _ulog("Refined mesh ready (trimesh).")

    if hasattr(pipeline, "maybe_free_model_hooks"):
        try:
            pipeline.maybe_free_model_hooks()
        except Exception:
            pass
    _ulog("UltraShape refinement stage complete.")
    return refined
