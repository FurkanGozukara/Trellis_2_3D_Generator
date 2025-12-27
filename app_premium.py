import gradio as gr

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import subprocess
import importlib

APP_DIR = os.path.dirname(os.path.abspath(__file__))
_O_VOXEL_SRC_DIR = os.path.join(APP_DIR, "o-voxel")


def _ensure_o_voxel_available() -> None:
    """
    TRELLIS.2 depends on the CUDA extension package `o_voxel`.

    On some installs (especially Windows), users may have the source present at
    `./o-voxel` but not actually installed into the current environment yet.
    This helper attempts a local install to avoid a hard crash on import.
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

    if not os.path.isdir(_O_VOXEL_SRC_DIR):
        raise ModuleNotFoundError(
            "No module named 'o_voxel'. Also could not find bundled source at "
            f"{_O_VOXEL_SRC_DIR!r}."
        )

    print(f"[setup] 'o_voxel' not found. Installing from bundled source: {_O_VOXEL_SRC_DIR}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", _O_VOXEL_SRC_DIR, "--no-build-isolation"]
        )
    except Exception as e:
        raise RuntimeError(
            "Could not install the required CUDA extension 'o_voxel'.\n"
            f"Tried: {sys.executable} -m pip install {_O_VOXEL_SRC_DIR} --no-build-isolation\n"
            "Make sure you're running in the project's venv and have a working CUDA + C++ "
            "build toolchain (NVCC + MSVC Build Tools on Windows)."
        ) from e

    importlib.invalidate_caches()
    import o_voxel  # noqa: F401


_ensure_o_voxel_available()

from datetime import datetime
import shutil
import base64
import io
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


# ------------------------------- Capability Checks ---------------------------

def _has_nvdiffrec_render() -> bool:
    try:
        import nvdiffrec_render  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


# ------------------------------- Paths / Config ------------------------------

MODELS_DIR = os.path.join(APP_DIR, "models")
TMP_DIR = os.path.join(APP_DIR, "tmp")

# Ensure TRELLIS_MODELS_DIR is set (trellis2 code also falls back to ../models).
os.environ.setdefault("TRELLIS_MODELS_DIR", MODELS_DIR)

MAX_SEED = np.iinfo(np.int32).max

MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3


# ------------------------------- UI Styling ---------------------------------

css = """
/* Slightly tightened layout & a cleaner preview panel */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: 722px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 320px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.25;
    margin: 6px 0;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.55;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.08); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.08);
}

.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 400px;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible { display: block; }

.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 420px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.15);
}

/* Remove padding around the HTML preview block */
.gradio-container .padded:has(.previewer-container) { padding: 0 !important; }
"""

head = """
<script>
    function refreshView(mode, step) {
        const allImgs = document.querySelectorAll('.previewer-main-image');
        for (let i = 0; i < allImgs.length; i++) {
            const img = allImgs[i];
            if (img.classList.contains('visible')) {
                const id = img.id;
                const [_, m, s] = id.split('-');
                if (mode === -1) mode = parseInt(m.slice(1));
                if (step === -1) step = parseInt(s.slice(1));
                break;
            }
        }

        allImgs.forEach(img => img.classList.remove('visible'));

        const targetId = 'view-m' + mode + '-s' + step;
        const targetImg = document.getElementById(targetId);
        if (targetImg) targetImg.classList.add('visible');

        const allBtns = document.querySelectorAll('.mode-btn');
        allBtns.forEach((btn, idx) => {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }
    function selectMode(mode) { refreshView(mode, -1); }
    function onSliderChange(val) { refreshView(-1, parseInt(val)); }
</script>
"""

empty_html = """
<div class="previewer-container">
    <svg style="opacity: .55; height: var(--size-5); color: var(--body-text-color);"
        xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <polyline points="21 15 16 10 5 21"></polyline>
    </svg>
</div>
"""


# ------------------------------- Model Loading ------------------------------

_image_pipeline = None
_texturing_pipeline = None
_envmap = None
_mode_icons_ready = False


def _image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def _ensure_mode_icons():
    global _mode_icons_ready
    if _mode_icons_ready:
        return
    for i in range(len(MODES)):
        icon = Image.open(os.path.join(APP_DIR, MODES[i]["icon"]))
        MODES[i]["icon_base64"] = _image_to_base64(icon)
    _mode_icons_ready = True


def _get_envmap():
    global _envmap
    if _envmap is not None:
        return _envmap

    def load_exr(name: str) -> EnvMap:
        path = os.path.join(APP_DIR, "assets", "hdri", f"{name}.exr")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return EnvMap(torch.tensor(img, dtype=torch.float32, device="cuda"))

    _envmap = {
        "forest": load_exr("forest"),
        "sunset": load_exr("sunset"),
        "courtyard": load_exr("courtyard"),
    }
    return _envmap


def get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        _image_pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        _image_pipeline.cuda()
    return _image_pipeline


def get_texturing_pipeline():
    global _texturing_pipeline
    if _texturing_pipeline is None:
        _texturing_pipeline = Trellis2TexturingPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B",
            config_file="texturing_pipeline.json",
        )
        _texturing_pipeline.cuda()
    return _texturing_pipeline


# ------------------------------- Session Utils ------------------------------

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


# ------------------------------- State Packing ------------------------------

def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        "shape_slat_feats": shape_slat.feats.cpu().numpy(),
        "tex_slat_feats": tex_slat.feats.cpu().numpy() if tex_slat is not None else None,
        "coords": shape_slat.coords.cpu().numpy(),
        "res": res,
    }


def unpack_state(state: Optional[dict]) -> Tuple[SparseTensor, Optional[SparseTensor], int]:
    if state is None:
        raise ValueError("No generation state found.")
    if not isinstance(state, dict):
        raise ValueError(f"Invalid generation state type: {type(state)!r}")
    for k in ("shape_slat_feats", "coords", "res"):
        if k not in state:
            raise ValueError(f"Missing key in generation state: {k!r}")
        if state[k] is None:
            raise ValueError(f"Missing value in generation state: {k!r}")

    shape_slat = SparseTensor(
        feats=torch.from_numpy(state["shape_slat_feats"]).cuda(),
        coords=torch.from_numpy(state["coords"]).cuda(),
    )
    if state.get("tex_slat_feats") is not None:
        tex_slat = shape_slat.replace(torch.from_numpy(state["tex_slat_feats"]).cuda())
    else:
        tex_slat = None
    return shape_slat, tex_slat, state["res"]


# ------------------------------- Shared Helpers -----------------------------

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def preprocess_image(
    image: Image.Image,
    progress=gr.Progress(track_tqdm=True),
) -> Image.Image:
    # Used by Upload and Examples. On first run it may load the full pipeline.
    progress(0.05, desc="Loading Image→3D pipeline (TRELLIS.2-4B)…")
    pipe = get_image_pipeline()
    progress(0.2, desc="Preprocessing image (background removal / crop)…")
    out = pipe.preprocess_image(image)
    progress(1.0, desc="Image ready.")
    return out


# ------------------------------- Preview Rendering ---------------------------

def _tensor_to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """
    Convert a (C,H,W) float tensor in [0,1] to (H,W,3) uint8 numpy.
    """
    if img.dim() != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(img.shape)}")
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.detach().float().clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _simple_shaded(base_color: torch.Tensor, normal_01: torch.Tensor, tint: torch.Tensor) -> torch.Tensor:
    """
    Very simple lambert-ish shading from normals and base color (no PBR / no envmap).
    Inputs are (3,H,W) in [0,1].
    """
    n = (normal_01 * 2.0 - 1.0)
    # Fixed light direction in camera space (roughly top-right-front)
    light_dir = torch.tensor([0.4, 0.2, 0.9], device=n.device, dtype=n.dtype)
    light_dir = light_dir / (light_dir.norm() + 1e-8)
    lambert = (n * light_dir.view(3, 1, 1)).sum(dim=0, keepdim=True).clamp(0.0, 1.0)
    ambient = 0.35
    shaded = base_color * (ambient + (1.0 - ambient) * lambert)
    shaded = shaded * tint.view(3, 1, 1).clamp(0.0, 2.0)
    return shaded.clamp(0.0, 1.0)


def _render_preview_snapshots_incremental(
    mesh,
    *,
    resolution: int,
    r: float,
    fov: float,
    nviews: int,
    envmap: Optional[Dict[str, EnvMap]],
    pbr_supported: bool,
    progress: gr.Progress,
    log_fn,
) -> Dict[str, List[np.ndarray]]:
    """
    Render preview images with per-view progress updates.
    Returns dict mapping render_key -> list[H,W,3 uint8] of length nviews.
    """
    # Camera setup
    yaw = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
    yaw_offset = -16 / 180 * np.pi
    yaw = [float(y + yaw_offset) for y in yaw]
    pitch = [20 / 180 * np.pi for _ in range(nviews)]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)

    images: Dict[str, List[np.ndarray]] = {m["render_key"]: [] for m in MODES}

    if pbr_supported:
        # Full PBR renderer (requires nvdiffrec_render)
        from trellis2.renderers import PbrMeshRenderer

        if envmap is None:
            raise RuntimeError("PBR rendering requested but envmap is None.")

        renderer = PbrMeshRenderer(
            rendering_options={
                "resolution": resolution,
                "near": 1,
                "far": 100,
                "ssaa": 2,
                "peel_layers": 8,
            }
        )

        for j, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
            p = 0.88 + 0.08 * (j / max(1, nviews - 1))
            log_fn(f"Rendering preview view {j + 1}/{nviews}…", p)
            res = renderer.render(mesh, extr, intr, envmap=envmap)
            for mode in MODES:
                key = mode["render_key"]
                if key not in res:
                    # If a key is missing, just fall back to base_color.
                    fallback = res.get("base_color", res.get("clay"))
                    images[key].append(_tensor_to_uint8_hwc(fallback))
                else:
                    images[key].append(_tensor_to_uint8_hwc(res[key]))
        return images

    # Fallback renderer (no nvdiffrec_render): use MeshRenderer and synthesize shaded modes.
    from trellis2.renderers import MeshRenderer

    log_fn(
        "HDRI/PBR preview disabled (missing 'nvdiffrec_render'). Using simple preview shading.",
        0.88,
    )
    renderer = MeshRenderer(
        rendering_options={
            "resolution": resolution,
            "near": 1,
            "far": 100,
            "ssaa": 2,
            "chunk_size": None,
        }
    )

    t_forest = torch.tensor([0.85, 1.05, 0.85], device="cuda")
    t_sunset = torch.tensor([1.10, 0.90, 0.75], device="cuda")
    t_court = torch.tensor([0.85, 0.95, 1.10], device="cuda")

    for j, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
        p = 0.88 + 0.08 * (j / max(1, nviews - 1))
        log_fn(f"Rendering preview view {j + 1}/{nviews}…", p)
        res = renderer.render(mesh, extr, intr, return_types=["mask", "normal", "attr"])

        normal = res["normal"]  # (3,H,W) in [0,1]
        base_color = res.get("base_color", torch.full_like(normal, 0.8))

        # Clay: simple AO-less clay from normal lighting
        clay_base = torch.full_like(base_color, 0.78)
        clay = _simple_shaded(clay_base, normal, torch.tensor([1.0, 1.0, 1.0], device=normal.device))

        shaded_forest = _simple_shaded(base_color, normal, t_forest)
        shaded_sunset = _simple_shaded(base_color, normal, t_sunset)
        shaded_courtyard = _simple_shaded(base_color, normal, t_court)

        # Fill all modes (keep existing UI keys)
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
            images[key].append(_tensor_to_uint8_hwc(mode_map[key]))

    return images


# ------------------------------- Image -> 3D --------------------------------

def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    no_texture_gen: bool,
    max_num_tokens: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[dict], str, str]:
    # Stream step-by-step status so users aren't "in the dark" during long runs.
    status = ""

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        status = (status + "\n" if status else "") + line
        print(line, flush=True)
        if p is not None:
            progress(p, desc=msg)
        return status

    if image is None:
        raise gr.Error("Please provide an image (upload or pick an example).")

    _log("Starting Image → 3D generation…", 0.0)
    yield None, empty_html, status

    _log("Loading TRELLIS.2 pipeline (first run can take a while)…", 0.01)
    pipe = get_image_pipeline()
    yield None, empty_html, status

    pbr_supported = _has_nvdiffrec_render()
    envmap = None
    if pbr_supported:
        _log("Loading HDRI environment maps…", 0.03)
        envmap = _get_envmap()
        yield None, empty_html, status
    else:
        _log("PBR preview not available (missing 'nvdiffrec_render'); will use fallback preview.", 0.03)
        yield None, empty_html, status

    _log("Preparing UI render-mode icons…", 0.05)
    _ensure_mode_icons()
    yield None, empty_html, status

    pipeline_type = {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
        "2048": "2048_cascade",
    }[resolution]

    ss_params = {
        "steps": ss_sampling_steps,
        "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale,
        "rescale_t": ss_rescale_t,
    }
    shape_params = {
        "steps": shape_slat_sampling_steps,
        "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale,
        "rescale_t": shape_slat_rescale_t,
    }
    tex_params = {
        "steps": tex_slat_sampling_steps,
        "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale,
        "rescale_t": tex_slat_rescale_t,
    }

    # Run the pipeline step-by-step so we can update status between stages.
    images = [image]
    torch.manual_seed(seed)

    _log("Computing image embeddings (512px)…", 0.08)
    cond_512 = pipe.get_cond(images, 512)
    yield None, empty_html, status

    cond_1024 = None
    if pipeline_type != "512":
        _log("Computing image embeddings (1024px)…", 0.12)
        cond_1024 = pipe.get_cond(images, 1024)
        yield None, empty_html, status

    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32, "2048_cascade": 32}[pipeline_type]
    _log("Stage 1/3: Sampling sparse structure…", 0.18)
    coords = pipe.sample_sparse_structure(cond_512, ss_res, 1, ss_params)
    yield None, empty_html, status

    if pipeline_type == "512":
        _log("Stage 2/3: Sampling shape latent (512)…", 0.35)
        shape_slat = pipe.sample_shape_slat(cond_512, pipe.models["shape_slat_flow_model_512"], coords, shape_params)
        yield None, empty_html, status

        if not no_texture_gen:
            _log("Stage 3/3: Sampling texture latent (512)…", 0.55)
            tex_slat = pipe.sample_tex_slat(cond_512, pipe.models["tex_slat_flow_model_512"], shape_slat, tex_params)
            yield None, empty_html, status
        else:
            _log("Stage 3/3: Skipping texture generation.", 0.55)
            tex_slat = None
            yield None, empty_html, status
        res = 512
    elif pipeline_type == "1024":
        _log("Stage 2/3: Sampling shape latent (1024)…", 0.35)
        shape_slat = pipe.sample_shape_slat(cond_1024, pipe.models["shape_slat_flow_model_1024"], coords, shape_params)
        yield None, empty_html, status

        if not no_texture_gen:
            _log("Stage 3/3: Sampling texture latent (1024)…", 0.55)
            tex_slat = pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, tex_params)
            yield None, empty_html, status
        else:
            _log("Stage 3/3: Skipping texture generation.", 0.55)
            tex_slat = None
            yield None, empty_html, status
        res = 1024
    elif pipeline_type in {"1024_cascade", "1536_cascade", "2048_cascade"}:
        target_res = {"1024_cascade": 1024, "1536_cascade": 1536, "2048_cascade": 2048}[pipeline_type]
        _log(f"Stage 2/3: Sampling shape latent (cascade → {target_res})…", 0.35)
        shape_slat, res = pipe.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            pipe.models["shape_slat_flow_model_512"],
            pipe.models["shape_slat_flow_model_1024"],
            512,
            target_res,
            coords,
            shape_params,
            max_num_tokens,
        )
        yield None, empty_html, status

        if not no_texture_gen:
            _log("Stage 3/3: Sampling texture latent (1024)…", 0.55)
            tex_slat = pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, tex_params)
            yield None, empty_html, status
        else:
            _log("Stage 3/3: Skipping texture generation.", 0.55)
            tex_slat = None
            yield None, empty_html, status
    else:
        raise gr.Error(f"Unsupported pipeline type: {pipeline_type}")

    _log("Decoding latent to mesh…", 0.75)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]
    yield None, empty_html, status

    _log("Simplifying mesh…", 0.82)
    mesh.simplify(16777216)  # nvdiffrast limit
    yield None, empty_html, status

    _log("Rendering preview snapshots…", 0.88)
    try:
        images = _render_preview_snapshots_incremental(
            mesh,
            resolution=1024,
            r=2,
            fov=36,
            nviews=STEPS,
            envmap=envmap,
            pbr_supported=pbr_supported,
            progress=progress,
            log_fn=_log,
        )
    except Exception as e:
        _log(f"Preview rendering failed ({type(e).__name__}: {e}). Continuing without preview.", 0.92)
        # Still continue so state is produced and Extract works.
        images = {m["render_key"]: [np.zeros((1024, 1024, 3), dtype=np.uint8) for _ in range(STEPS)] for m in MODES}
    yield None, empty_html, status

    _log("Packing generation state (for GLB extraction)…", 0.93)
    state = pack_state((shape_slat, tex_slat, res))
    torch.cuda.empty_cache()
    yield None, empty_html, status

    _log("Building preview UI…", 0.97)
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            # Small progress ticks while we convert images to base64 and build HTML.
            # (48 images total)
            p = 0.97 + 0.02 * ((m_idx * STEPS + s_idx) / max(1, (len(MODES) * STEPS - 1)))
            progress(p, desc="Building preview UI…")
            unique_id = f"view-m{m_idx}-s{s_idx}"
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            img_base64 = _image_to_base64(Image.fromarray(images[mode["render_key"]][s_idx]))
            images_html += f"""
                <img id="{unique_id}"
                     class="previewer-main-image {vis_class}"
                     src="{img_base64}"
                     loading="eager">
            """

    btns_html = ""
    for idx, mode in enumerate(MODES):
        active_class = "active" if idx == DEFAULT_MODE else ""
        btns_html += f"""
            <img src="{mode['icon_base64']}"
                 class="mode-btn {active_class}"
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """

    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> — Click a circular button to switch render modes.</p>
                <p>● <b>View Angle</b> — Drag the slider to change view angle.</p>
            </div>
        </div>

        <div class="display-row">
            {images_html}
        </div>

        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """

    _log("Done. You can now click “Extract GLB”.", 1.0)
    yield state, full_html, status


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    remesh_method: str,
    simplify_method: str,
    no_texture_gen: bool,
    prune_invisible_faces: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    if state is None:
        # This happens when users click "Extract GLB" before clicking "Generate"
        # (or right after changing the image / clicking an example).
        raise gr.Error("Nothing to extract yet. Click **Generate** first.")

    texture_extraction = not no_texture_gen

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    try:
        shape_slat, tex_slat, res = unpack_state(state)
    except Exception:
        raise gr.Error("Invalid/empty generation state. Please click **Generate** again.")

    status = ""

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        status = (status + "\n" if status else "") + line
        print(line, flush=True)
        if p is not None:
            progress(p, desc=msg)
        return status

    _log("Starting GLB extraction…", 0.0)
    yield None, None, status

    _log("Loading TRELLIS.2 pipeline…", 0.05)
    pipe = get_image_pipeline()
    yield None, None, status

    _log("Decoding latent to mesh…", 0.15)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]
    yield None, None, status

    _log("Post-processing + baking GLB (this can take a while)…", 0.3)
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
        texture_extraction=texture_extraction,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        remesh_method=remesh_method,
        prune_invisible=prune_invisible_faces,
        use_tqdm=True,
    )
    yield None, None, status

    _log("Saving GLB…", 0.9)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f"sample_{timestamp}.glb")
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    _log("Done.", 1.0)
    yield glb_path, glb_path, status


# ------------------------------- Texturing ----------------------------------

def shapeimage_to_tex(
    mesh_file: str,
    image: Image.Image,
    seed: int,
    resolution: str,
    texture_size: int,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    status = ""

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        status = (status + "\n" if status else "") + line
        print(line, flush=True)
        if p is not None:
            progress(p, desc=msg)
        return status

    if mesh_file is None:
        raise gr.Error("Please upload a mesh file (or use the example).")
    if image is None:
        raise gr.Error("Please provide a reference image (or use the example).")

    _log("Starting Texturing…", 0.0)
    yield None, None, status

    _log("Loading texturing pipeline (first run can take a while)…", 0.05)
    pipe = get_texturing_pipeline()
    yield None, None, status

    _log("Loading mesh…", 0.1)

    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    yield None, None, status

    res_int = int(resolution)
    tex_params = {
        "steps": tex_slat_sampling_steps,
        "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale,
        "rescale_t": tex_slat_rescale_t,
    }

    _log("Preprocessing reference image…", 0.18)
    image = pipe.preprocess_image(image)
    yield None, None, status

    _log("Preprocessing mesh…", 0.22)
    mesh = pipe.preprocess_mesh(mesh)
    yield None, None, status

    torch.manual_seed(seed)
    _log(f"Computing image embeddings ({512 if res_int == 512 else 1024}px)…", 0.3)
    cond = pipe.get_cond([image], 512) if res_int == 512 else pipe.get_cond([image], 1024)
    yield None, None, status

    _log("Encoding mesh to shape latent…", 0.4)
    shape_slat = pipe.encode_shape_slat(mesh, res_int)
    yield None, None, status

    tex_model = pipe.models["tex_slat_flow_model_512"] if res_int == 512 else pipe.models["tex_slat_flow_model_1024"]
    _log("Sampling texture latent…", 0.55)
    tex_slat = pipe.sample_tex_slat(cond, tex_model, shape_slat, tex_params)
    yield None, None, status

    _log("Decoding texture latent…", 0.72)
    pbr_voxel = pipe.decode_tex_slat(tex_slat)
    yield None, None, status

    _log("Baking textures onto mesh…", 0.84)
    output = pipe.postprocess_mesh(mesh, pbr_voxel, res_int, texture_size)
    yield None, None, status

    _log("Saving textured GLB…", 0.9)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f"textured_{timestamp}.glb")
    output.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    _log("Done.", 1.0)
    yield glb_path, glb_path, status


# --------------------------------- App UI -----------------------------------

with gr.Blocks(
    title="TRELLIS.2 Premium",
    theme=gr.themes.Soft(),
    css=css,
    head=head,
    delete_cache=(600, 600),
) as demo:
    gr.Markdown(
        """
## TRELLIS.2 Premium (Image → 3D + Texturing)
- **Offline-ready**: download models into `Trellis_2_3D_Generator/models/` via `HF_model_downloader.py`.
- **Two tools in one app**: generate a mesh from an image, then (optionally) texture any mesh with a reference image.
"""
    )

    demo.load(start_session)
    demo.unload(end_session)

    with gr.Tabs():
        # ---------------------------- Tab 1: Image -> 3D ----------------------------
        with gr.Tab("Image → 3D"):
            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)

                    resolution = gr.Radio(["512", "1024", "1536", "2048"], label="Resolution", value="1024")
                    seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    decimation_target = gr.Slider(100000, 1000000, label="Decimation Target", value=500000, step=10000)
                    remesh_method = gr.Dropdown(["dual_contouring", "faithful_contouring"], label="Remesh Method", value="dual_contouring")
                    simplify_method = gr.Dropdown(["cumesh", "meshlib"], label="Simplify Method", value="cumesh")
                    prune_invisible_faces = gr.Checkbox(label="Prune Invisible Faces", value=True)
                    no_texture_gen = gr.Checkbox(label="Skip Texture Generation", value=False)
                    texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

                    status_box = gr.Textbox(
                        label="Progress",
                        value="Select an image (upload or example), then click Generate.",
                        lines=10,
                        interactive=False,
                    )

                    with gr.Accordion(label="Advanced Settings", open=False):
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                            ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                            ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)

                        gr.Markdown("Stage 2: Shape Generation")
                        with gr.Row():
                            shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                            shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                            shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                            max_num_tokens = gr.Slider(10000, 200000, label="Max Number of Tokens", value=49152, step=1000)

                        gr.Markdown("Stage 3: Material Generation")
                        with gr.Row():
                            tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                            tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                            tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                            tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)

                with gr.Column(scale=10):
                    with gr.Walkthrough(selected=0) as walkthrough:
                        with gr.Step("Preview", id=0):
                            preview_output = gr.HTML(empty_html, label="3D Asset Preview", show_label=True, container=True)
                            generate_btn = gr.Button("Generate", variant="primary")
                            extract_btn = gr.Button("Extract GLB", interactive=False)
                        with gr.Step("Extract", id=1):
                            glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                            download_btn = gr.DownloadButton(label="Download GLB")

                with gr.Column(scale=1, min_width=172):
                    examples = gr.Examples(
                        examples=[
                            os.path.join(APP_DIR, "assets", "example_image", image)
                            for image in os.listdir(os.path.join(APP_DIR, "assets", "example_image"))
                        ],
                        inputs=[image_prompt],
                        fn=preprocess_image,
                        outputs=[image_prompt],
                        run_on_click=True,
                        examples_per_page=18,
                    )

            output_buf = gr.State()

            def _reset_image_to_3d_ui():
                return (
                    None,  # output_buf
                    empty_html,  # preview_output
                    gr.update(interactive=False),  # extract_btn
                    gr.Walkthrough(selected=0),  # walkthrough
                    None,  # glb_output
                    None,  # download_btn
                    "Select an image (upload or example), then click Generate.",  # status_box
                )

            image_prompt.upload(
                preprocess_image,
                inputs=[image_prompt],
                outputs=[image_prompt],
            )

            # Any time the input image changes (upload, example click, clear), invalidate previous results.
            image_prompt.change(
                _reset_image_to_3d_ui,
                inputs=[],
                outputs=[output_buf, preview_output, extract_btn, walkthrough, glb_output, download_btn, status_box],
            )

            generate_btn.click(
                get_seed,
                inputs=[randomize_seed, seed],
                outputs=[seed],
            ).then(
                lambda: gr.Walkthrough(selected=0), outputs=walkthrough
            ).then(
                _reset_image_to_3d_ui,
                inputs=[],
                outputs=[output_buf, preview_output, extract_btn, walkthrough, glb_output, download_btn, status_box],
            ).then(
                image_to_3d,
                inputs=[
                    image_prompt,
                    seed,
                    resolution,
                    ss_guidance_strength,
                    ss_guidance_rescale,
                    ss_sampling_steps,
                    ss_rescale_t,
                    shape_slat_guidance_strength,
                    shape_slat_guidance_rescale,
                    shape_slat_sampling_steps,
                    shape_slat_rescale_t,
                    tex_slat_guidance_strength,
                    tex_slat_guidance_rescale,
                    tex_slat_sampling_steps,
                    tex_slat_rescale_t,
                    no_texture_gen,
                    max_num_tokens,
                ],
                outputs=[output_buf, preview_output, status_box],
            ).then(
                lambda: gr.update(interactive=True), outputs=extract_btn
            )

            extract_btn.click(
                lambda: gr.Walkthrough(selected=1), outputs=walkthrough
            ).then(
                extract_glb,
                inputs=[output_buf, decimation_target, texture_size, remesh_method, simplify_method, no_texture_gen, prune_invisible_faces],
                outputs=[glb_output, download_btn, status_box],
            )

        # ---------------------------- Tab 2: Texturing -------------------------------
        with gr.Tab("Texturing"):
            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    mesh_file = gr.File(label="Upload Mesh", file_types=[".ply", ".obj", ".glb", ".gltf"], file_count="single")
                    tex_image = gr.Image(label="Reference Image", format="png", image_mode="RGBA", type="pil", height=400)

                    tex_resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
                    tex_seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                    tex_randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    tex_texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

                    tex_generate_btn = gr.Button("Generate Textured GLB", variant="primary")
                    tex_status_box = gr.Textbox(
                        label="Progress",
                        value="Upload a mesh + reference image (or use the example), then click Generate.",
                        lines=10,
                        interactive=False,
                    )

                    with gr.Accordion(label="Advanced Settings", open=False):
                        with gr.Row():
                            t_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                            t_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                            t_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                            t_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)

                with gr.Column(scale=10):
                    textured_glb_output = gr.Model3D(label="Textured GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    textured_download_btn = gr.DownloadButton(label="Download Textured GLB")

                with gr.Column(scale=1, min_width=172):
                    tex_examples = gr.Examples(
                        examples=[
                            [
                                os.path.join(APP_DIR, "assets", "example_texturing", "the_forgotten_knight.ply"),
                                os.path.join(APP_DIR, "assets", "example_texturing", "image.webp"),
                            ]
                        ],
                        inputs=[mesh_file, tex_image],
                        examples_per_page=6,
                    )

            tex_generate_btn.click(
                get_seed,
                inputs=[tex_randomize_seed, tex_seed],
                outputs=[tex_seed],
            ).then(
                shapeimage_to_tex,
                inputs=[
                    mesh_file,
                    tex_image,
                    tex_seed,
                    tex_resolution,
                    tex_texture_size,
                    t_guidance_strength,
                    t_guidance_rescale,
                    t_sampling_steps,
                    t_rescale_t,
                ],
                outputs=[textured_glb_output, textured_download_btn, tex_status_box],
            )


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


