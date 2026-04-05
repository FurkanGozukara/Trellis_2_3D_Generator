import gradio as gr

import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
import subprocess
import signal
import importlib
import json
import time
import threading
import inspect
import string
from pathlib import Path

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FAVICON_PATH = os.path.join(APP_DIR, "assets", "app", "trellis2-favicon.svg")
_O_VOXEL_SRC_DIR = os.path.join(APP_DIR, "o-voxel")


def _example_image_sort_key(name: str) -> tuple[int, int, str]:
    stem = os.path.splitext(name)[0].lower()
    if stem.startswith("image") and stem[5:].isdigit():
        return (0, int(stem[5:]), stem)
    return (1, 0, stem)


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


# _ensure_o_voxel_available()  # Disabled - use pre-installed o_voxel from pip

from datetime import datetime
import shutil
import base64
import io
from typing import Tuple, Optional, Dict, List, Any, Set, Callable

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import (
    Trellis2ImageTo3DPipeline,
    Trellis2MultiViewPipeline,
    Trellis2TexturingPipeline,
)
from trellis2.projection import resolve_projection_views
from trellis2.runtime_options import (
    ATTENTION_BACKEND_CHOICES,
    DEFAULT_ATTENTION_BACKEND,
    DEFAULT_MODEL_VARIANT,
    DEFAULT_SAMPLER_TYPE,
    MODEL_VARIANT_CHOICES,
    SAMPLER_TYPE_CHOICES,
    apply_runtime_backends,
    normalize_model_variant,
    normalize_sampler_type,
    resolve_model_variant,
)
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


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _is_faithful_contouring_available() -> bool:
    """
    `faithful_contouring` remeshing in `o_voxel.postprocess.to_glb()` depends on optional
    FaithC packages plus their runtime imports. A shallow `find_spec()` check is not
    enough because mismatched `atom3d` installs or missing `torch_scatter` can still
    make the remesher unusable at import time.
    """
    try:
        from faithcontour import FCTDecoder, FCTEncoder, normalize_mesh  # noqa: F401
        from atom3d import MeshBVH  # noqa: F401
        from atom3d.grid import OctreeIndexer  # noqa: F401
        import torch_scatter  # noqa: F401
        return True
    except Exception:
        return False


def _is_meshlib_available() -> bool:
    try:
        import meshlib.mrmeshpy  # noqa: F401
        import meshlib.mrmeshnumpy  # noqa: F401
        return True
    except Exception:
        return False


def _is_pymeshfix_available() -> bool:
    try:
        import pymeshfix  # noqa: F401
        import pyvista  # noqa: F401
        return True
    except Exception:
        return False


def _has_dual_contouring_vb() -> bool:
    try:
        import cumesh

        return hasattr(cumesh.remeshing, "reconstruct_mesh_dc")
    except Exception:
        return False


REMESH_METHOD_CHOICES = ["dual_contouring"]
if _has_dual_contouring_vb():
    REMESH_METHOD_CHOICES.append("dual_contouring_vb")
if _is_faithful_contouring_available():
    REMESH_METHOD_CHOICES.append("faithful_contouring")

SIMPLIFY_METHOD_CHOICES = ["cumesh"]
if _is_meshlib_available():
    SIMPLIFY_METHOD_CHOICES.append("meshlib")
SIMPLIFY_METHOD_CHOICES.append("none")

REPAIR_METHOD_CHOICES = ["disabled", "cumesh"]
if _is_meshlib_available():
    REPAIR_METHOD_CHOICES.append("meshlib")
if _is_pymeshfix_available():
    REPAIR_METHOD_CHOICES.append("pymeshfix")

PIPELINE_STRATEGY_CHOICES = ["reference_auto", "direct_1024", "hybrid_512g_1024t"]


def _normalize_simplify_method(method: Optional[str]) -> str:
    method_norm = str(method or "cumesh").strip().lower()
    if method_norm not in SIMPLIFY_METHOD_CHOICES:
        return "meshlib" if "meshlib" in SIMPLIFY_METHOD_CHOICES else "cumesh"
    return method_norm


def _normalize_repair_method(method: Optional[str]) -> str:
    method_norm = str(method or "disabled").strip().lower()
    if method_norm in {"", "none"}:
        method_norm = "disabled"
    if method_norm not in REPAIR_METHOD_CHOICES:
        return "disabled"
    return method_norm


def _normalize_extract_methods(
    *,
    remesh_method: Optional[str],
    simplify_method: Optional[str],
    repair_method: Optional[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[str, str, str]:
    remesh_norm = str(remesh_method or "dual_contouring").strip().lower()
    simplify_norm = _normalize_simplify_method(simplify_method)
    repair_norm = _normalize_repair_method(repair_method)

    if remesh_norm not in REMESH_METHOD_CHOICES:
        if log_fn is not None:
            log_fn(
                f"Requested remesh method '{remesh_norm}' is unavailable in this environment. "
                "Falling back to 'dual_contouring'."
            )
        remesh_norm = "dual_contouring"

    if remesh_norm == "dual_contouring_vb" and not _has_dual_contouring_vb():
        if log_fn is not None:
            log_fn(
                "Requested remesh_method='dual_contouring_vb' but the installed CuMesh build "
                "does not expose reconstruct_mesh_dc. Falling back to 'dual_contouring'."
            )
        remesh_norm = "dual_contouring"

    if simplify_norm == "meshlib" and not _is_meshlib_available():
        if log_fn is not None:
            log_fn("MeshLib is not installed. Falling back to simplify_method='cumesh'.")
        simplify_norm = "cumesh"

    if repair_norm == "meshlib" and not _is_meshlib_available():
        if log_fn is not None:
            log_fn("MeshLib is not installed. Disabling Extract GLB hole repair.")
        repair_norm = "disabled"
    if repair_norm == "pymeshfix" and not _is_pymeshfix_available():
        if log_fn is not None:
            log_fn("PyMeshFix is not installed. Disabling Extract GLB hole repair.")
        repair_norm = "disabled"

    return remesh_norm, simplify_norm, repair_norm


# ------------------------------- Paths / Config ------------------------------

MODELS_DIR = os.path.join(APP_DIR, "models")
TMP_DIR = os.path.join(APP_DIR, "tmp")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
PRESETS_DIR = os.path.join(APP_DIR, "presets")
SUBPROCESS_STAGE_SCRIPT = os.path.join(APP_DIR, "subprocess_stage.py")
EXTRACT_FINAL_DIRNAME = "08_final_exports"
RETEXTURE_WORK_DIRNAME = "09_retexture_work"

# UniRig paths
UNIRIG_DIR = os.path.join(APP_DIR, "UniRig")
UNIRIG_RUN_PY = os.path.join(UNIRIG_DIR, "run.py")
LEGACY_RIGGING_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "rigged_models")
# Rigging runs now share the same numbered outputs root (outputs/0001, outputs/0002, ...).
RIGGING_OUTPUTS_DIR = OUTPUTS_DIR


# Ensure TRELLIS_MODELS_DIR is set (trellis2 code also falls back to ../models).
os.environ.setdefault("TRELLIS_MODELS_DIR", MODELS_DIR)

DEFAULT_SUBPROCESS_PYTHON = r"G:\Trellis2_v3\Trellis_2_3D_Generator\venv\Scripts\python.exe"
LOCAL_SUBPROCESS_PYTHON = os.path.join(APP_DIR, "venv", "Scripts", "python.exe")


def _resolve_subprocess_python() -> str:
    candidates = [
        os.environ.get("TRELLIS_SUBPROCESS_PYTHON"),
        LOCAL_SUBPROCESS_PYTHON,
        sys.executable,
        DEFAULT_SUBPROCESS_PYTHON,
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return str(candidate)
    return sys.executable


def _configure_image_pipeline_runtime(
    pipe: Any,
    *,
    attention_backend: Optional[str],
    sampler_type: Optional[str],
) -> dict:
    runtime = apply_runtime_backends(attention_backend)
    normalized_sampler = normalize_sampler_type(sampler_type)
    if hasattr(pipe, "switch_samplers"):
        pipe.switch_samplers(normalized_sampler)
    runtime["sampler_type"] = normalized_sampler
    return runtime


def _configure_texturing_pipeline_runtime(
    pipe: Any,
    *,
    attention_backend: Optional[str],
    sampler_type: Optional[str],
) -> dict:
    runtime = apply_runtime_backends(attention_backend)
    normalized_sampler = normalize_sampler_type(sampler_type)
    if hasattr(pipe, "switch_sampler"):
        pipe.switch_sampler(normalized_sampler)
    runtime["sampler_type"] = normalized_sampler
    return runtime


def _discover_allowed_paths_all_drives() -> List[str]:
    """
    Build a permissive allow-list for Gradio file serving.
    - Windows: include all detected drive roots (A:\\..Z:\\)
    - Linux/macOS: include '/' and discovered mount points
    """
    allowed: List[str] = []
    seen: Set[str] = set()

    def _add(path: str) -> None:
        if not path:
            return
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = os.path.abspath(path)
        if resolved in seen:
            return
        if os.path.isdir(resolved):
            seen.add(resolved)
            allowed.append(resolved)

    # Always include app-specific paths first.
    for base in [
        APP_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        RIGGING_OUTPUTS_DIR,
        LEGACY_RIGGING_OUTPUTS_DIR,
        TMP_DIR,
        PRESETS_DIR,
    ]:
        _add(base)

    if os.name == "nt":
        for letter in string.ascii_uppercase:
            _add(f"{letter}:\\")
    else:
        # Root covers all mounted filesystems on Unix-like systems.
        _add("/")

        for parent in ("/mnt", "/media", "/run/media", "/Volumes"):
            _add(parent)
            try:
                parent_path = Path(parent)
                if parent_path.is_dir():
                    for child in parent_path.iterdir():
                        if child.is_dir():
                            _add(str(child))
            except Exception:
                pass

        proc_mounts = Path("/proc/mounts")
        if proc_mounts.exists():
            try:
                for line in proc_mounts.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        mount_point = parts[1].replace("\\040", " ")
                        _add(mount_point)
            except Exception:
                pass

    return allowed

# Local helpers (not the stdlib `subprocess` module)
from subprocess_utils import allocate_run_dir, next_indexed_path, ensure_dir, safe_relpath  # noqa: E402

# Trellis UI modules
from trellis_ui.rigging_tab import rigging_tab
from trellis_ui.animation_player_tab import animation_player_tab

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
    /* Bigger main preview while staying responsive */
    height: min(820px, 72vh);
    min-height: 680px;
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

/* ----------------------------- Stage Action Buttons ------------------------ */
@keyframes warm-stage-button-glow {
    0%, 100% {
        box-shadow:
            0 14px 34px rgba(127, 16, 35, 0.38),
            0 0 18px rgba(239, 68, 68, 0.32),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    50% {
        box-shadow:
            0 18px 40px rgba(127, 16, 35, 0.46),
            0 0 28px rgba(248, 113, 113, 0.42),
            inset 0 1px 0 rgba(255, 255, 255, 0.26);
    }
}

@keyframes purple-stage-button-glow {
    0%, 100% {
        box-shadow:
            0 14px 34px rgba(76, 29, 149, 0.34),
            0 0 18px rgba(168, 85, 247, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }
    50% {
        box-shadow:
            0 18px 42px rgba(76, 29, 149, 0.42),
            0 0 30px rgba(196, 181, 253, 0.34),
            inset 0 1px 0 rgba(255, 255, 255, 0.24);
    }
}

:where(
    #generate_step_btn button,
    button#generate_step_btn,
    #extract_step_btn button,
    button#extract_step_btn,
    .stage-action-btn button,
    button.stage-action-btn
) {
    min-height: 58px;
    padding: 0 1.4rem !important;
    border: 1px solid transparent !important;
    border-radius: 18px !important;
    color: #fff !important;
    font-size: 0.98rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em;
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.08);
    background-size: 180% 180% !important;
    transition:
        transform 0.18s ease,
        filter 0.18s ease,
        box-shadow 0.18s ease,
        background-position 0.28s ease;
}

:where(
    #generate_step_btn button,
    button#generate_step_btn,
    #extract_step_btn button,
    button#extract_step_btn,
    .stage-action-btn button,
    button.stage-action-btn
):hover:not(:disabled) {
    transform: translateY(-2px) scale(1.01);
    filter: saturate(1.08) brightness(1.04);
    background-position: 100% 50% !important;
}

:where(
    #generate_step_btn button,
    button#generate_step_btn,
    #extract_step_btn button,
    button#extract_step_btn,
    .stage-action-btn button,
    button.stage-action-btn
):active:not(:disabled) {
    transform: translateY(0) scale(0.99);
}

:where(
    #generate_step_btn button,
    button#generate_step_btn,
    #extract_step_btn button,
    button#extract_step_btn,
    .stage-action-btn button,
    button.stage-action-btn
):focus-visible {
    outline: none;
}

:where(
    #generate_step_btn button,
    button#generate_step_btn,
    #extract_step_btn button,
    button#extract_step_btn,
    .stage-action-btn button,
    button.stage-action-btn
):disabled {
    cursor: not-allowed;
    opacity: 0.72 !important;
    filter: saturate(0.72) brightness(0.86);
    animation: none !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.08),
        0 8px 18px rgba(15, 23, 42, 0.14) !important;
}

:where(
    #generate_step_btn button,
    button#generate_step_btn,
    .generate-stage-btn button,
    button.generate-stage-btn
) {
    background: linear-gradient(135deg, #7f1023 0%, #b91c1c 36%, #ef4444 72%, #f97316 100%) !important;
    border-color: rgba(255, 208, 208, 0.34) !important;
    box-shadow:
        0 14px 34px rgba(127, 16, 35, 0.38),
        0 0 18px rgba(239, 68, 68, 0.32),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    animation: warm-stage-button-glow 2.7s ease-in-out infinite;
}

:where(
    #generate_step_btn button,
    button#generate_step_btn,
    .generate-stage-btn button,
    button.generate-stage-btn
):hover:not(:disabled),
:where(
    #generate_step_btn button,
    button#generate_step_btn,
    .generate-stage-btn button,
    button.generate-stage-btn
):focus-visible:not(:disabled) {
    box-shadow:
        0 18px 40px rgba(127, 16, 35, 0.46),
        0 0 28px rgba(248, 113, 113, 0.42),
        inset 0 1px 0 rgba(255, 255, 255, 0.24),
        0 0 0 3px rgba(248, 113, 113, 0.16) !important;
}

:where(
    #extract_step_btn button,
    button#extract_step_btn,
    .extract-stage-btn button,
    button.extract-stage-btn
) {
    background: linear-gradient(135deg, #312e81 0%, #5b21b6 35%, #7c3aed 72%, #a855f7 100%) !important;
    border-color: rgba(224, 215, 255, 0.34) !important;
    box-shadow:
        0 14px 34px rgba(76, 29, 149, 0.34),
        0 0 18px rgba(168, 85, 247, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
    animation: purple-stage-button-glow 2.9s ease-in-out infinite;
}

:where(
    #extract_step_btn button,
    button#extract_step_btn,
    .extract-stage-btn button,
    button.extract-stage-btn
):hover:not(:disabled),
:where(
    #extract_step_btn button,
    button#extract_step_btn,
    .extract-stage-btn button,
    button.extract-stage-btn
):focus-visible:not(:disabled) {
    box-shadow:
        0 18px 42px rgba(76, 29, 149, 0.42),
        0 0 30px rgba(196, 181, 253, 0.34),
        inset 0 1px 0 rgba(255, 255, 255, 0.22),
        0 0 0 3px rgba(196, 181, 253, 0.16) !important;
}

/* ---------------------------- Preview Progress Overlay --------------------- */
/* Replaces the old left-side Progress panel: keep progress on top of the preview. */
#preview_stack { position: relative; }
#preview_status_overlay {
    position: absolute;
    inset: 0;
    padding: 12px;
    z-index: 50;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}
#preview_status_overlay > label {
    width: 100%;
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
#preview_status_overlay > label > .input-container {
    flex: 1 1 auto;
    height: 100%;
    min-height: 0;
    /* Gradio Textbox uses a row flex container; we need cross-axis stretching for full height */
    align-items: stretch;
}
#preview_status_overlay textarea {
    background: rgba(0, 0, 0, 0.78) !important;
    color: #fff !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
    font-size: 12px !important;
    line-height: 1.25 !important;
    height: 100% !important;
    min-height: 0 !important;
    overflow-y: auto !important;
}

/* ----------------------------- Model3D Fullscreen -------------------------- */
/* Gradio's Model3D root uses data-testid="model3d" (see gradio/js/model3D/shared/Model3D.svelte) */
[data-testid="model3d"]:fullscreen {
    width: 100vw !important;
    height: 100vh !important;
    background: #000;
}
[data-testid="model3d"]:fullscreen canvas {
    width: 100% !important;
    height: 100% !important;
}
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

APP_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
    font=(
        "'Segoe UI Variable Display'",
        "'Segoe UI Variable'",
        "'Segoe UI'",
        "'Inter'",
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ),
    font_mono=(
        "'Cascadia Mono'",
        "'Cascadia Code'",
        "'JetBrains Mono'",
        "'Consolas'",
        "ui-monospace",
        "monospace",
    ),
)


# ------------------------------- Model Loading ------------------------------

_image_pipeline = None
_image_pipeline_key = None
_multiview_pipeline = None
_multiview_pipeline_key = None
_texturing_pipeline = None
_envmap = None
_mode_icons_ready = False


def _image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def _jpeg_file_to_data_uri(path: str) -> str:
    """
    Encode an existing JPEG file to a data URI without re-encoding.
    """
    data = Path(path).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def _write_json(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_indexed_stem(stem: str) -> Tuple[str, Optional[int]]:
    base, sep, tail = str(stem).rpartition("_")
    if sep and tail.isdigit():
        try:
            return base, int(tail)
        except Exception:
            pass
    return str(stem), None


def _export_prefix_from_glb_prefix(glb_prefix: str, fmt: str) -> str:
    fmt_l = str(fmt).lower().strip()
    glb_prefix_s = str(glb_prefix).strip()
    glb_prefix_l = glb_prefix_s.lower()
    if glb_prefix_l == "glb":
        return fmt_l
    if glb_prefix_l.startswith("glb_"):
        return f"{fmt_l}_{glb_prefix_s[4:]}"
    return f"{fmt_l}_{glb_prefix_s}"


def _export_path_for_format(out_dir: Path, fmt: str, glb_prefix: str, idx: int) -> Path:
    fmt_l = str(fmt).lower().strip()
    ext = "gltf" if fmt_l == "gltf" else fmt_l
    return out_dir / f"{_export_prefix_from_glb_prefix(glb_prefix, fmt_l)}_{idx:04d}.{ext}"


def _round_float(value: Any, digits: int = 6) -> Optional[float]:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _detect_visual_kind(mesh: trimesh.Trimesh) -> str:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return "none"
    kind = type(visual).__name__.lower()
    if "texture" in kind:
        return "texture"
    if "color" in kind:
        return "color"
    return kind


def _collect_mesh_audit(model_path: Path) -> Optional[Dict[str, Any]]:
    try:
        asset = trimesh.load(str(model_path), force="mesh", process=False)
        if isinstance(asset, trimesh.Scene):
            asset = asset.to_mesh()
        if not isinstance(asset, trimesh.Trimesh):
            return None

        material = getattr(getattr(asset, "visual", None), "material", None)
        bounds = getattr(asset, "bounds", None)
        extents = getattr(asset, "extents", None)
        body_count = getattr(asset, "body_count", None)
        vertex_normals = getattr(asset, "vertex_normals", None)
        uv = getattr(getattr(asset, "visual", None), "uv", None)

        audit: Dict[str, Any] = {
            "path": model_path.name,
            "vertex_count": int(len(asset.vertices)),
            "face_count": int(len(asset.faces)),
            "is_empty": bool(asset.is_empty),
            "is_watertight": bool(asset.is_watertight),
            "is_winding_consistent": bool(asset.is_winding_consistent),
            "is_volume": bool(getattr(asset, "is_volume", False)),
            "body_count": int(body_count) if body_count is not None else None,
            "euler_number": int(asset.euler_number) if getattr(asset, "euler_number", None) is not None else None,
            "has_uv": bool(uv is not None and len(uv) == len(asset.vertices)),
            "has_vertex_normals": bool(vertex_normals is not None and len(vertex_normals) == len(asset.vertices)),
            "visual_kind": _detect_visual_kind(asset),
            "has_base_color_texture": bool(getattr(material, "baseColorTexture", None) is not None),
            "has_metallic_roughness_texture": bool(getattr(material, "metallicRoughnessTexture", None) is not None),
            "surface_area": _round_float(getattr(asset, "area", None)),
            "volume": _round_float(getattr(asset, "volume", None)) if bool(getattr(asset, "is_volume", False)) else None,
            "bounds_min": [float(x) for x in bounds[0].tolist()] if bounds is not None else None,
            "bounds_max": [float(x) for x in bounds[1].tolist()] if bounds is not None else None,
            "extents": [float(x) for x in extents.tolist()] if extents is not None else None,
            "bounding_box_diagonal": (
                _round_float(np.linalg.norm(extents)) if extents is not None else None
            ),
        }
        return audit
    except Exception:
        return None


def _write_extract_artifacts_manifest(
    *,
    out_dir: Path,
    stage3_glb_path: Optional[str],
    stage3_has_textures: Optional[bool],
    final_glb_path: str,
    final_has_textures: bool,
    final_texture_source: str,
    export_formats: List[str],
    retexture_requested: bool,
    requested_remesh_method: Optional[str] = None,
    effective_remesh_method: Optional[str] = None,
    remesh_fallback_reason: Optional[str] = None,
) -> None:
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _rel(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(out_dir.resolve()))
            except Exception:
                return p.name

        requested_formats: List[str] = []
        for fmt in export_formats or []:
            fmt_l = str(fmt).lower().strip()
            if fmt_l in {"glb", "gltf", "obj", "ply", "stl"} and fmt_l not in requested_formats:
                requested_formats.append(fmt_l)
        if "glb" not in requested_formats:
            requested_formats.insert(0, "glb")

        stage3_path_obj = Path(stage3_glb_path) if stage3_glb_path else None
        final_path_obj = Path(final_glb_path)
        artifacts: List[Dict[str, Any]] = []

        if stage3_path_obj is not None and stage3_path_obj.is_file():
            artifacts.append(
                {
                    "path": _rel(stage3_path_obj),
                    "role": ("intermediate_shape_only" if retexture_requested else "final_glb"),
                    "stage": "extract_to_glb",
                    "is_final": (stage3_path_obj.resolve() == final_path_obj.resolve()),
                    "contains_geometry": True,
                    "contains_pbr_textures": bool(stage3_has_textures),
                }
            )

        if final_path_obj.is_file() and (
            stage3_path_obj is None or stage3_path_obj.resolve() != final_path_obj.resolve()
        ):
            artifacts.append(
                {
                    "path": _rel(final_path_obj),
                    "role": "final_glb",
                    "stage": ("tex_decode_and_bake" if retexture_requested else "extract_to_glb"),
                    "is_final": True,
                    "contains_geometry": True,
                    "contains_pbr_textures": bool(final_has_textures),
                }
            )

        final_prefix, final_idx = _split_indexed_stem(final_path_obj.stem)
        if final_idx is not None:
            for fmt in requested_formats:
                if fmt == "glb":
                    continue
                sidecar = _export_path_for_format(out_dir, fmt, final_prefix, final_idx)
                if sidecar.is_file():
                    artifacts.append(
                        {
                            "path": _rel(sidecar),
                            "role": "final_export",
                            "stage": "final_export_formats",
                            "is_final": True,
                            "format": fmt,
                            "contains_geometry": True,
                            "contains_pbr_textures": bool(final_has_textures),
                        }
                    )

        mesh_audit: Dict[str, Any] = {
            "schema": "trellis2.mesh_audit.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": {},
        }
        final_audit = _collect_mesh_audit(final_path_obj) if final_path_obj.is_file() else None
        if final_audit is not None:
            mesh_audit["artifacts"]["final_output"] = final_audit
        if (
            stage3_path_obj is not None
            and stage3_path_obj.is_file()
            and (not final_path_obj.is_file() or stage3_path_obj.resolve() != final_path_obj.resolve())
        ):
            stage3_audit = _collect_mesh_audit(stage3_path_obj)
            if stage3_audit is not None:
                mesh_audit["artifacts"]["stage3_output"] = stage3_audit
        mesh_audit_report_path = out_dir / "mesh_audit.json"
        if mesh_audit["artifacts"]:
            _write_json(str(mesh_audit_report_path), mesh_audit)

        manifest = {
            "schema": "trellis2.extract_artifacts.v2",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "stage_meanings": {
                "extract_to_glb": "Converts decoded mesh voxel into remeshed/simplified GLB.",
                "tex_decode_and_bake": "Optional re-texture pass that bakes new PBR textures onto extracted mesh.",
                "final_export_formats": "Extra exports generated from the final GLB (gltf/obj/ply/stl).",
            },
            "mesh_audit_report": (
                str(mesh_audit_report_path.name) if mesh_audit["artifacts"] else None
            ),
            "final_output": {
                "path": _rel(final_path_obj),
                "contains_geometry": True,
                "contains_pbr_textures": bool(final_has_textures),
                "texture_source": str(final_texture_source),
            },
            "extract_settings": {
                "requested_remesh_method": (
                    str(requested_remesh_method) if requested_remesh_method is not None else None
                ),
                "effective_remesh_method": (
                    str(effective_remesh_method) if effective_remesh_method is not None else None
                ),
                "remesh_fallback_applied": (
                    bool(
                        requested_remesh_method is not None
                        and effective_remesh_method is not None
                        and str(requested_remesh_method) != str(effective_remesh_method)
                    )
                ),
                "remesh_fallback_reason": (
                    str(remesh_fallback_reason) if remesh_fallback_reason else None
                ),
            },
            "artifacts": artifacts,
        }
        _write_json(str(out_dir / "extract_artifacts.json"), manifest)
    except Exception:
        pass


# ------------------------------- Presets / Config ----------------------------

UI_PRESET_VERSION = "1.0"
UI_PRESET_FORMAT = "trellis2_premium_ui"
_LAST_USED_UI_PRESET_FILE = ".last_used_ui_preset.txt"


def _sanitize_preset_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "default"


def _ui_preset_path(preset_name: str) -> Path:
    safe = _sanitize_preset_name(preset_name)
    return Path(PRESETS_DIR) / f"{safe}.json"


def _list_ui_presets() -> List[str]:
    builtin = list(_builtin_ui_presets().keys())
    root = Path(PRESETS_DIR)
    if not root.exists():
        return builtin
    saved = sorted([p.stem for p in root.glob("*.json") if p.is_file() and p.stem not in builtin])
    return builtin + saved


def _set_last_used_ui_preset(preset_name: str) -> None:
    try:
        root = Path(PRESETS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        (root / _LAST_USED_UI_PRESET_FILE).write_text(_sanitize_preset_name(preset_name), encoding="utf-8")
    except Exception:
        pass


def _get_last_used_ui_preset() -> Optional[str]:
    root = Path(PRESETS_DIR)
    path = root / _LAST_USED_UI_PRESET_FILE
    if not path.exists():
        return None
    try:
        name = path.read_text(encoding="utf-8").strip()
        if not name:
            return None
        return name if name in _list_ui_presets() else None
    except Exception:
        return None


def _default_ui_config() -> dict:
    """
    Must match ALL user-settable UI defaults.
    (Inputs like uploaded images/files are intentionally not saved in presets.)
    """
    return {
        "_meta": {
            "version": UI_PRESET_VERSION,
            "format": UI_PRESET_FORMAT,
        },
        "global": {
            "subprocess_mode": True,
        },
        "image_to_3d": {
            "resolution": "1024",
            "pipeline_strategy": "reference_auto",
            # Keep deterministic defaults (users can enable randomize for exploration).
            "seed": 99,
            "randomize_seed": False,
            "decimation_target": 1000000,
            "remesh_method": ("dual_contouring_vb" if "dual_contouring_vb" in REMESH_METHOD_CHOICES else "dual_contouring"),
            "fill_holes_max_perimeter": 0.03,
            "repair_method": "meshlib" if "meshlib" in REPAIR_METHOD_CHOICES else "disabled",
            "simplify_method": "cumesh",
            "prune_invisible_faces": False,
            "merge_vertices_dist": 0.0,
            "shade_smooth": False,
            "shade_smooth_angle": 35.0,
            "force_double_sided": True,
            "no_pbr_export": False,
            "no_texture_gen": False,
            "deferred_texture_after_cleanup": True,
            "texture_size": 4096,
            "export_formats": ["glb"],
            "low_vram": False,  # Keep models in VRAM for best quality and speed
            "model_variant": "standard",
            "attention_backend": "auto",
            "sampler_type": "heun",
            "ss_guidance_strength": 7.5,
            "ss_guidance_rescale": 0.7,
            "ss_guidance_interval_start": 0.6,  # Model default: CFG only in last 40% of sampling
            "ss_guidance_interval_end": 1.0,
            "ss_sampling_steps": 12,
            "ss_rescale_t": 5.0,
            "force_high_res_conditional": False,
            "use_chunked_processing": False,
            "use_tiled_extraction": False,
            "extract_use_chunked_processing": False,
            "extract_use_tiled_extraction": False,
            "shape_slat_guidance_strength": 7.5,
            "shape_slat_guidance_rescale": 0.5,
            "shape_slat_guidance_interval_start": 0.6,  # Model default: CFG only in last 40% of sampling
            "shape_slat_guidance_interval_end": 1.0,
            "shape_slat_sampling_steps": 12,
            "shape_slat_rescale_t": 3.0,
            "max_num_tokens": 400000,
            "multiview_mode": "multidiffusion",
            "tex_slat_guidance_strength": 1.0,
            "tex_slat_guidance_rescale": 0.0,
            "tex_slat_guidance_interval_start": 0.6,  # Model default: CFG in middle 30% range
            "tex_slat_guidance_interval_end": 0.9,
            "tex_slat_sampling_steps": 12,
            "tex_slat_rescale_t": 3.0,
            "projection_texture_refine": False,
            "projection_view_azimuths": "",
            "projection_view_elevations": "",
            "projection_blend_exponent": 2.0,
            "projection_ortho_scale": 1.1,
            "projection_fill_holes": True,
            "projection_max_hole_size": 20,
            "ultrashape_enabled": False,
            "ultrashape_retexture_after_refine": False,
            "ultrashape_conservative_mode": False,
            "ultrashape_checkpoint": "",
            "ultrashape_config_name": "infer_dit_refine.yaml",
            "ultrashape_dtype": "bfloat16",
            "ultrashape_low_vram": True,
            "ultrashape_remove_bg": False,
            "ultrashape_steps": 50,
            "ultrashape_guidance_scale": 5.0,
            "ultrashape_octree_resolution": 384,
            "ultrashape_num_chunks": 8000,
            "ultrashape_target_face_count": 500000,
            "ultrashape_num_latents": 0,
            "ultrashape_box_v": 1.0,
            "ultrashape_mc_level": 0.0,
            "ultrashape_normalize_scale": 0.99,
            "ultrashape_num_sharp_points": 204800,
            "ultrashape_num_uniform_points": 204800,
        },
        "texturing": {
            "resolution": "1024",
            "seed": 99,
            "randomize_seed": False,
            "texture_size": 2048,
            "low_vram": True,  # Default True for memory safety
            "attention_backend": "auto",
            "sampler_type": "heun",
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "guidance_interval_start": 0.6,
            "guidance_interval_end": 0.9,
            "sampling_steps": 12,
            "rescale_t": 3.0,
        },
        "ultrashape_refine": {
            "seed": 99,
            "randomize_seed": False,
            "output_format": "glb",
            "export_formats": ["glb"],
            "checkpoint": "",
            "config_name": "infer_dit_refine.yaml",
            "dtype": "bfloat16",
            "low_vram": True,
            "remove_bg": False,
            "steps": 50,
            "guidance_scale": 5.0,
            "octree_resolution": 384,
            "num_chunks": 8000,
            "target_face_count": 500000,
            "num_latents": 0,
            "box_v": 1.0,
            "mc_level": 0.0,
            "normalize_scale": 0.99,
            "num_sharp_points": 204800,
            "num_uniform_points": 204800,
        },
        "rigging": {
            "seed": 12345,
            "randomize_seed": False,
            "skeleton_task": "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml",
            "skin_task": "configs/task/quick_inference_unirig_skin.yaml",
            "export_format": "fbx",  # or "glb"
            "export_both_formats": True,
            "faces_target_count": 50000,
            "enable_skinning": True,
            "auto_merge": True,
        },
    }


def _builtin_ui_presets() -> Dict[str, dict]:
    best_cfg = _default_ui_config()
    best_cfg["_meta"].update(
        {
            "builtin": True,
            "preset_name": "best",
            "description": "Best tested default preset.",
        }
    )

    low_vram_cfg = _default_ui_config()
    low_vram_cfg["_meta"].update(
        {
            "builtin": True,
            "preset_name": "low_vram",
            "description": "Lower-VRAM preset using the tested lower-memory export path plus chunked/tiled memory-saving switches.",
        }
    )
    low_vram_cfg["image_to_3d"].update(
        {
            "low_vram": True,
            "use_chunked_processing": True,
            "use_tiled_extraction": True,
            "extract_use_chunked_processing": True,
            "extract_use_tiled_extraction": True,
            "texture_size": 2048,
        }
    )
    low_vram_cfg["texturing"].update(
        {
            "texture_size": 1024,
            "low_vram": True,
        }
    )

    return {
        "best": best_cfg,
        "low_vram": low_vram_cfg,
    }


def _merge_ui_config(cfg: Optional[dict]) -> dict:
    """
    Merge a loaded config with defaults so older presets still work after adding new params.
    """
    base = _default_ui_config()
    if not isinstance(cfg, dict):
        return base

    meta = cfg.get("_meta")
    if isinstance(meta, dict):
        base["_meta"].update(meta)

    for section in ("global", "image_to_3d", "texturing", "ultrashape_refine", "rigging"):
        section_data = cfg.get(section)
        if isinstance(section_data, dict):
            base[section].update(section_data)

    return base


def _save_ui_preset(preset_name: str, config: dict) -> str:
    if not preset_name or not str(preset_name).strip():
        raise ValueError("Preset name cannot be empty.")

    safe_name = _sanitize_preset_name(str(preset_name).strip())
    if safe_name in _builtin_ui_presets():
        raise ValueError(f"Preset name '{safe_name}' is reserved for a built-in preset.")
    root = Path(PRESETS_DIR)
    root.mkdir(parents=True, exist_ok=True)

    cfg = _merge_ui_config(config)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["version"] = UI_PRESET_VERSION
    cfg["_meta"]["format"] = UI_PRESET_FORMAT
    cfg["_meta"]["last_modified"] = datetime.now().isoformat()
    if "created_at" not in cfg["_meta"]:
        cfg["_meta"]["created_at"] = cfg["_meta"]["last_modified"]

    out_path = _ui_preset_path(safe_name)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    _set_last_used_ui_preset(safe_name)
    return safe_name


def _load_ui_preset(preset_name: str) -> Optional[dict]:
    if not preset_name:
        return None
    builtin = _builtin_ui_presets().get(str(preset_name).strip())
    if builtin is not None:
        _set_last_used_ui_preset(str(preset_name).strip())
        return _merge_ui_config(builtin)
    path = _ui_preset_path(preset_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    _set_last_used_ui_preset(preset_name)
    return _merge_ui_config(data)


def _delete_ui_preset(preset_name: str) -> bool:
    if not preset_name:
        return False
    if str(preset_name).strip() in _builtin_ui_presets():
        return False
    path = _ui_preset_path(preset_name)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


_STATUS_TEXT_REPLACEMENTS = {
    "Ã¢â‚¬Â¦": "...",
    "â€¦": "...",
    "â†’": "->",
    "âž¡": "->",
    "â€œ": '"',
    "â€": '"',
    "â€˜": "'",
    "â€™": "'",
    "â€“": "-",
    "â€”": "-",
    "â€‘": "-",
    "â—": "*",
    "â‰¥": ">=",
    "â‰¤": "<=",
    "âœ…": "[OK]",
    "âŒ": "[ERROR]",
    "âš ï¸": "[WARNING]",
    "â„¹ï¸": "[INFO]",
    "ðŸ›‘": "[STOP]",
    "…": "...",
    "→": "->",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "‑": "-",
    "●": "*",
    "⚠️": "[WARNING]",
    "✅": "[OK]",
    "❌": "[ERROR]",
    "ℹ️": "[INFO]",
    "🛑": "[STOP]",
}


def _clean_status_text(text: Any) -> str:
    s = "" if text is None else str(text)

    # Best-effort repair for UTF-8 text that was decoded as Latin-1/CP1252.
    for _ in range(2):
        if not any(token in s for token in ("â", "Ã", "ð", "Â")):
            break
        try:
            repaired = s.encode("latin-1").decode("utf-8")
        except Exception:
            break
        if repaired == s:
            break
        s = repaired

    for bad, good in _STATUS_TEXT_REPLACEMENTS.items():
        s = s.replace(bad, good)

    return s.replace("\ufffd", "")


def _append_status(current: str, msg: str) -> str:
    current = current or ""
    msg = _clean_status_text(msg)
    if not current:
        return msg
    if not msg:
        return current
    return current + "\n" + msg


# Keep streamed UI logs bounded so Gradio + the browser never get overwhelmed.
_UI_STATUS_MAX_LINES = 200
_UI_STATUS_MAX_CHARS = 20000


def _trim_status(
    status: str,
    *,
    max_lines: int = _UI_STATUS_MAX_LINES,
    max_chars: int = _UI_STATUS_MAX_CHARS,
) -> str:
    status = status or ""
    if not status:
        return status

    # Char-bound first (avoid huge splitlines() cost if something goes wild).
    if max_chars and len(status) > max_chars * 2:
        status = status[-max_chars * 2 :]

    if max_lines:
        lines = status.splitlines()
        if len(lines) > max_lines:
            lines = ["... (truncated) ..."] + lines[-max_lines:]
            status = "\n".join(lines)

    if max_chars and len(status) > max_chars:
        status = "... (truncated) ...\n" + status[-max_chars:]

    return status


def _open_folder(path: str) -> None:
    path = os.path.abspath(path)
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
        return
    # Linux / WSL / others
    for cmd in (["xdg-open", path], ["gio", "open", path]):
        try:
            subprocess.Popen(cmd)
            return
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No folder opener found (tried: xdg-open, gio).")


def _iter_subprocess_stage(stage: str, payload: dict, work_dir: Path, log_path: Path, *, session: str):
    """
    Run one stage in a fresh Python subprocess and stream its stdout.
    Yields dict events:
      - {"type":"log","text": "..."}
      - {"type":"result","result": {...}}
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload_path = work_dir / f"{stage}.payload.json"
    result_path = work_dir / f"{stage}.result.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    python_exe = _resolve_subprocess_python()
    cmd = [
        python_exe,
        "-u",
        SUBPROCESS_STAGE_SCRIPT,
        "--stage",
        stage,
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
    ]

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    popen_kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd,
        cwd=APP_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **popen_kwargs,
    )
    _register_active_subproc(session, proc, stage)

    try:
        # Stream output to both UI and a per-run log file.
        with log_path.open("a", encoding="utf-8") as lf:
            assert proc.stdout is not None
            for line in proc.stdout:
                clean_line = _clean_status_text(line.rstrip("\n"))
                lf.write(clean_line + "\n")
                lf.flush()
                yield {"type": "log", "text": clean_line}
    finally:
        _unregister_active_subproc(session, proc)
        # If the Gradio request is cancelled, this generator can be closed mid-stream.
        # Ensure we don't leave any worker subprocess running.
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

    rc = proc.wait()
    if not result_path.exists():
        # If a user cancellation killed the worker, treat it as a clean cancel instead of an error.
        if _is_cancel_all(session):
            raise UserCancelled(f"Cancelled during stage {stage!r}.")
        raise RuntimeError(f"Stage {stage!r} failed (rc={rc}) and produced no result file.")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not data.get("ok", False):
        tb = data.get("traceback") or ""
        if _is_cancel_all(session):
            raise UserCancelled(f"Cancelled during stage {stage!r}.")
        raise RuntimeError(f"Stage {stage!r} failed: {data.get('error_type')}: {data.get('error')}\n{tb}")

    yield {"type": "result", "result": data.get("result", {})}


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


def get_image_pipeline(model_variant: str = DEFAULT_MODEL_VARIANT):
    global _image_pipeline, _image_pipeline_key, _multiview_pipeline, _multiview_pipeline_key
    variant_cfg = resolve_model_variant(model_variant)
    key = (variant_cfg["model_repo"], variant_cfg["config_file"])
    if _image_pipeline is None or _image_pipeline_key != key:
        if _image_pipeline is not None:
            _image_pipeline.cpu()
            del _image_pipeline
            _image_pipeline = None
            _image_pipeline_key = None
            torch.cuda.empty_cache()
        if _multiview_pipeline is not None:
            _multiview_pipeline.cpu()
            del _multiview_pipeline
            _multiview_pipeline = None
            _multiview_pipeline_key = None
            torch.cuda.empty_cache()
        _image_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            variant_cfg["model_repo"],
            config_file=variant_cfg["config_file"],
        )
        _image_pipeline_key = key
        _image_pipeline.low_vram = False  # Keep models in VRAM for best quality and speed
        _image_pipeline.cuda()
    return _image_pipeline


def get_multiview_pipeline(model_variant: str = DEFAULT_MODEL_VARIANT):
    global _image_pipeline, _image_pipeline_key, _multiview_pipeline, _multiview_pipeline_key
    variant_cfg = resolve_model_variant(model_variant)
    key = (variant_cfg["model_repo"], variant_cfg["config_file"])
    if _multiview_pipeline is None or _multiview_pipeline_key != key:
        if _multiview_pipeline is not None:
            _multiview_pipeline.cpu()
            del _multiview_pipeline
            _multiview_pipeline = None
            _multiview_pipeline_key = None
            torch.cuda.empty_cache()
        if _image_pipeline is not None:
            _image_pipeline.cpu()
            del _image_pipeline
            _image_pipeline = None
            _image_pipeline_key = None
            torch.cuda.empty_cache()
        _multiview_pipeline = Trellis2MultiViewPipeline.from_pretrained(
            variant_cfg["model_repo"],
            config_file=variant_cfg["config_file"],
        )
        _multiview_pipeline_key = key
        _multiview_pipeline.low_vram = False  # Keep models in VRAM for best quality and speed
        _multiview_pipeline.cuda()
    return _multiview_pipeline


def get_texturing_pipeline():
    global _texturing_pipeline
    if _texturing_pipeline is None:
        _texturing_pipeline = Trellis2TexturingPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B",
            config_file="texturing_pipeline.json",
        )
        _texturing_pipeline.cuda()
    return _texturing_pipeline


def _collect_preprocessed_view_paths(run_dir: Path) -> List[str]:
    paths: List[Path] = []
    first = run_dir / "01_preprocessed.png"
    if first.is_file():
        paths.append(first)
    paths.extend(sorted(run_dir.glob("01_preprocessed_view*.png")))
    return [str(path) for path in paths if path.is_file()]


def unload_global_pipelines():
    """Unload any global pipelines to free VRAM before subprocess mode."""
    global _image_pipeline, _image_pipeline_key, _multiview_pipeline, _multiview_pipeline_key, _texturing_pipeline
    import gc

    if _image_pipeline is not None:
        print("[main] Unloading global image pipeline to free VRAM...", flush=True)
        _image_pipeline.cpu()
        del _image_pipeline
        _image_pipeline = None
        _image_pipeline_key = None

    if _multiview_pipeline is not None:
        print("[main] Unloading global multi-view pipeline to free VRAM...", flush=True)
        _multiview_pipeline.cpu()
        del _multiview_pipeline
        _multiview_pipeline = None
        _multiview_pipeline_key = None

    if _texturing_pipeline is not None:
        print("[main] Unloading global texturing pipeline to free VRAM...", flush=True)
        _texturing_pipeline.cpu()
        del _texturing_pipeline
        _texturing_pipeline = None

    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------- Session Utils ------------------------------

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    # Defensive: clear any stale flags if a session is re-used.
    session = _session_key(req)
    _clear_cancel_all(session)
    _clear_cancel_batch(session)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    session = _session_key(req)
    # Best-effort cleanup so we don't keep stale state around.
    proc, _stage = _get_active_subproc(session)
    if proc is not None:
        _terminate_process(proc)
    with _CANCEL_LOCK:
        _RUNNING_TASKS.pop(session, None)
        _CANCEL_ALL.pop(session, None)
        _CANCEL_BATCH.pop(session, None)
        _ACTIVE_SUBPROCS.pop(session, None)
        _ACTIVE_SUBPROCS_STAGE.pop(session, None)
    shutil.rmtree(user_dir, ignore_errors=True)


# ------------------------------- UniRig Helpers ------------------------------


def _list_rigged_models() -> List[str]:
    """List previously rigged models from outputs directory."""
    rigged_dir = Path(RIGGING_OUTPUTS_DIR)
    if not rigged_dir.exists():
        return []

    supported_exts = {".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"}
    rig_name_tokens = (
        "_skeleton",
        "_skinned",
        "_rigged",
        "_anim_preview",
        "_animation_preview",
        "_skeleton_preview",
    )
    models = []
    for item in rigged_dir.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in supported_exts:
            continue
        stem_l = item.stem.lower()
        if not any(tok in stem_l for tok in rig_name_tokens):
            continue
        # Store relative paths from rigging outputs dir.
        rel_path = item.relative_to(rigged_dir)
        models.append(str(rel_path))

    return sorted(set(models))


def _safe_rig_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    safe = safe.strip("._")
    return safe or "mesh"


def _ensure_rig_workspace_dirs(work_dir: Path) -> None:
    (work_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    (work_dir / "tmp_npz").mkdir(parents=True, exist_ok=True)
    (work_dir / "preview").mkdir(parents=True, exist_ok=True)


def _rig_metadata_path(work_dir: Path) -> Path:
    return work_dir / "run_metadata.json"


def _load_rig_metadata(work_dir: Path) -> Dict[str, Any]:
    path = _rig_metadata_path(work_dir)
    if path.exists():
        try:
            data = _read_json(str(path))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "work_dir": str(work_dir),
        "input": {},
        "paths": {
            "logs_dir": str(work_dir / "logs"),
            "outputs_dir": str(work_dir),
            "tmp_npz_dir": str(work_dir / "tmp_npz"),
            "full_log_path": str(work_dir / "logs" / "run_full.log"),
        },
        "stages": {},
    }


def _save_rig_metadata(work_dir: Path, metadata: Dict[str, Any]) -> None:
    metadata["last_updated_at"] = datetime.now().isoformat()
    _write_json(str(_rig_metadata_path(work_dir)), metadata)


def _append_rig_full_log(work_dir: Path, stage: str, line: str) -> None:
    log_path = work_dir / "logs" / "run_full.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{stage}] {line}\n")


def _find_rig_work_dir(path: Path) -> Path:
    for candidate in [path.parent, *path.parents]:
        if (_rig_metadata_path(candidate)).exists():
            return candidate
    if path.parent.name in {"generated", "inputs", "logs", "tmp_npz", "preview"}:
        return path.parent.parent
    return path.parent


def _prepare_rig_input(input_mesh_path: str, upload_run_dir: Optional[str]) -> Tuple[Path, Path]:
    input_src = Path(input_mesh_path)
    if upload_run_dir:
        work_dir = Path(upload_run_dir)
    else:
        run = allocate_run_dir(OUTPUTS_DIR, digits=4)
        work_dir = run.run_dir
    _ensure_rig_workspace_dirs(work_dir)

    dst = work_dir / "inputs" / _safe_rig_filename(input_src.name)
    try:
        same_file = input_src.resolve() == dst.resolve()
    except Exception:
        same_file = False
    if not same_file:
        shutil.copy2(input_src, dst)

    metadata = _load_rig_metadata(work_dir)
    metadata["input"] = {
        "original_upload_path": str(input_src),
        "copied_input_path": str(dst),
        "filename": input_src.name,
    }
    _save_rig_metadata(work_dir, metadata)
    return work_dir, dst


def _normalize_model3d_preview_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    try:
        path = Path(path_value).resolve()
        if not path.exists() or not path.is_file():
            return None
        # Gradio Model3D does not render FBX directly.
        previewable_exts = {".glb", ".gltf", ".obj", ".ply", ".stl", ".splat"}
        if path.suffix.lower() not in previewable_exts:
            return None
        return path.as_posix()
    except Exception:
        return None


def _select_rig_preview_source(metadata: Dict[str, Any], fallback_path: Optional[str] = None) -> Optional[str]:
    input_meta = metadata.get("input", {}) if isinstance(metadata, dict) else {}
    candidates = [
        input_meta.get("preview_path"),
        input_meta.get("copied_input_path"),
        fallback_path,
    ]
    for candidate in candidates:
        normalized = _normalize_model3d_preview_path(candidate)
        if normalized:
            return normalized
    return None


def _load_rig_metadata_for_path(path_value: str) -> Dict[str, Any]:
    """Best-effort metadata lookup by walking parent directories."""
    try:
        path = Path(path_value).resolve()
    except Exception:
        return {}
    for parent in [path.parent, *path.parents]:
        meta_path = parent / "run_metadata.json"
        if meta_path.exists():
            try:
                data = _read_json(str(meta_path))
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        if parent == Path(OUTPUTS_DIR):
            break
    return {}


def _preferred_animation_preview_path(path_value: str) -> Optional[str]:
    """Prefer a generated animated GLB preview next to or referenced by a rig output."""
    try:
        path = Path(path_value).resolve()
    except Exception:
        return None

    # 1) Check metadata reference first.
    metadata = _load_rig_metadata_for_path(str(path))
    paths_meta = metadata.get("paths", {}) if isinstance(metadata, dict) else {}
    preview_from_meta = (
        paths_meta.get("textured_animation_preview")
        or paths_meta.get("animation_preview")
        or paths_meta.get("textured_preview")
        or paths_meta.get("skeleton_preview")
    )
    normalized = _normalize_model3d_preview_path(preview_from_meta)
    if normalized:
        return normalized

    # 2) Check common sibling naming.
    sibling_candidates = [
        path.with_name(f"{path.stem}_textured_anim_preview.glb"),
        path.with_name(f"{path.stem}_anim_preview.glb"),
        path.with_name(f"{path.stem}_animation_preview.glb"),
        path.with_name(f"{path.stem}_textured_preview.glb"),
        path.with_name(f"{path.stem}_preview.glb"),
    ]
    for candidate in sibling_candidates:
        normalized = _normalize_model3d_preview_path(str(candidate))
        if normalized:
            return normalized
    return None


def _run_unirig_skeleton(input_mesh_path: str, seed: int, upload_run_dir: Optional[str], req: gr.Request):
    """
    Generate skeleton for uploaded mesh using UniRig via subprocess.
    Yields status updates as processing proceeds.
    """
    session = str(req.session_hash)
    work_dir: Optional[Path] = None

    try:
        if not input_mesh_path or not os.path.exists(input_mesh_path):
            yield (None, None, "[ERROR] Please upload a valid mesh file.")
            return

        work_dir, input_path = _prepare_rig_input(input_mesh_path, upload_run_dir)
        logs_dir = work_dir / "logs"

        skeleton_fbx = work_dir / f"{input_path.stem}_skeleton.fbx"
        npz_dir = work_dir / "tmp_npz"
        log_path = logs_dir / "skeleton_log.txt"
        preview_log_path = logs_dir / "skeleton_preview_log.txt"

        metadata = _load_rig_metadata(work_dir)
        metadata.setdefault("stages", {})["skeleton"] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "log_path": str(log_path),
            "payload_path": str(work_dir / "unirig_skeleton.payload.json"),
            "result_path": str(work_dir / "unirig_skeleton.result.json"),
            "input_mesh_path": str(input_path),
            "output_path": str(skeleton_fbx),
            "seed": int(seed),
        }
        _save_rig_metadata(work_dir, metadata)

        yield (
            None,
            None,
            f"Starting skeleton generation...\nInput: {input_path.name}\nSeed: {seed}\nWorkspace: {work_dir}",
        )

        payload = {
            "input_mesh_path": str(input_path),
            "output_fbx_path": str(skeleton_fbx),
            "npz_dir": str(npz_dir),
            "seed": seed,
        }

        status = ""
        result = None
        for event in _iter_subprocess_stage("unirig_skeleton", payload, work_dir, log_path, session=session):
            if event["type"] == "log":
                _append_rig_full_log(work_dir, "skeleton", event["text"])
                status = _append_status(status, event["text"])
                status = _trim_status(status)
                yield (None, None, status)
            elif event["type"] == "result":
                result = event["result"]

        metadata = _load_rig_metadata(work_dir)
        if result and skeleton_fbx.exists():
            skeleton_preview_glb: Optional[str] = None
            skeleton_npz = npz_dir / skeleton_fbx.stem / "raw_data.npz"
            preview_payload = {
                "source_npz_path": str(skeleton_npz),
                "source_fbx_path": str(skeleton_fbx),
                "npz_dir": str(npz_dir),
                "faces_target_count": 50000,
                "mesh_alpha": 0.5,
                "include_mesh": True,
                "visibility_boost": True,
                "output_glb_path": str(work_dir / f"{input_path.stem}_skeleton_preview.glb"),
            }
            try:
                for event in _iter_subprocess_stage(
                    "unirig_skeleton_preview",
                    preview_payload,
                    work_dir,
                    preview_log_path,
                    session=session,
                ):
                    if event["type"] == "log":
                        _append_rig_full_log(work_dir, "skeleton_preview", event["text"])
                        status = _append_status(status, event["text"])
                        status = _trim_status(status)
                        yield (None, None, status)
                    elif event["type"] == "result":
                        skeleton_preview_glb = event["result"].get("preview_glb_path")
            except Exception as preview_err:
                status = _append_status(
                    status,
                    f"[skeleton_preview] Warning: {type(preview_err).__name__}: {preview_err}",
                )
                status = _trim_status(status)
                yield (None, None, status)
            metadata.setdefault("stages", {}).setdefault("skeleton", {}).update(
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": result,
                }
            )
            metadata.setdefault("paths", {})["skeleton_output"] = str(skeleton_fbx)
            if skeleton_preview_glb and os.path.exists(skeleton_preview_glb):
                metadata.setdefault("paths", {})["skeleton_preview"] = str(Path(skeleton_preview_glb).resolve())
            _save_rig_metadata(work_dir, metadata)
            preview_source = _normalize_model3d_preview_path(skeleton_preview_glb)
            note = "\nPreview: mesh + generated skeleton overlay."
            if not preview_source:
                preview_source = _select_rig_preview_source(metadata, str(input_path))
                note = "\nPreview fallback: could not build skeleton overlay, showing uploaded mesh."
            final_status = _append_status(status, f"\n✅ Skeleton generated successfully!\nOutput: {skeleton_fbx}{note}")
            yield (str(skeleton_fbx), preview_source, final_status)
        else:
            metadata.setdefault("stages", {}).setdefault("skeleton", {}).update(
                {
                    "status": "failed",
                    "completed_at": datetime.now().isoformat(),
                }
            )
            _save_rig_metadata(work_dir, metadata)
            yield (None, None, _append_status(status, "\n[ERROR] Skeleton generation failed."))

    except Exception as e:
        if work_dir is not None and work_dir.exists():
            try:
                metadata = _load_rig_metadata(work_dir)
                metadata.setdefault("stages", {}).setdefault("skeleton", {}).update(
                    {
                        "status": "failed",
                        "completed_at": datetime.now().isoformat(),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                _save_rig_metadata(work_dir, metadata)
                _append_rig_full_log(work_dir, "skeleton", f"ERROR: {type(e).__name__}: {e}")
            except Exception:
                pass
        yield (None, None, f"[ERROR] Error: {type(e).__name__}: {e}")


def _run_unirig_skinning(skeleton_fbx_path: str, seed: int, req: gr.Request):
    """
    Add skinning weights to skeleton using UniRig via subprocess.
    """
    session = str(req.session_hash)
    work_dir: Optional[Path] = None

    try:
        if not skeleton_fbx_path or not os.path.exists(skeleton_fbx_path):
            yield (None, None, "[ERROR] Please generate or upload a skeleton first.")
            return

        skeleton_path = Path(skeleton_fbx_path)
        work_dir = _find_rig_work_dir(skeleton_path)
        _ensure_rig_workspace_dirs(work_dir)
        skinned_fbx = work_dir / f"{skeleton_path.stem.replace('_skeleton', '')}_skinned.fbx"
        log_path = work_dir / "logs" / "skinning_log.txt"
        animation_log_path = work_dir / "logs" / "animation_preview_log.txt"
        npz_dir = work_dir / "tmp_npz"

        metadata = _load_rig_metadata(work_dir)
        metadata.setdefault("stages", {})["skinning"] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "log_path": str(log_path),
            "payload_path": str(work_dir / "unirig_skinning.payload.json"),
            "result_path": str(work_dir / "unirig_skinning.result.json"),
            "input_skeleton_path": str(skeleton_path),
            "output_path": str(skinned_fbx),
            "seed": int(seed),
        }
        _save_rig_metadata(work_dir, metadata)

        yield (None, None, f"Starting skinning prediction...\nSkeleton: {skeleton_path.name}\nSeed: {seed}")

        payload = {
            "input_skeleton_path": str(skeleton_path),
            "output_fbx_path": str(skinned_fbx),
            "npz_dir": str(npz_dir),
            "seed": seed,
            "data_name": "raw_data.npz",
        }

        status = ""
        result = None
        for event in _iter_subprocess_stage("unirig_skinning", payload, work_dir, log_path, session=session):
            if event["type"] == "log":
                _append_rig_full_log(work_dir, "skinning", event["text"])
                status = _append_status(status, event["text"])
                status = _trim_status(status)
                yield (None, None, status)
            elif event["type"] == "result":
                result = event["result"]

        metadata = _load_rig_metadata(work_dir)
        if result and skinned_fbx.exists():
            animation_preview_glb: Optional[str] = None
            textured_preview_glb: Optional[str] = None
            animation_source_path: Path = skinned_fbx

            input_meta = metadata.get("input", {}) if isinstance(metadata, dict) else {}
            target_mesh_path = input_meta.get("copied_input_path") or input_meta.get("original_upload_path")
            merge_preview_log_path = work_dir / "logs" / "animation_preview_merge_log.txt"

            # Build a textured GLB preview first, then animate that asset.
            if target_mesh_path and os.path.exists(target_mesh_path):
                textured_preview_output = work_dir / f"{skinned_fbx.stem}_textured_preview.glb"
                merge_preview_payload = {
                    "source_path": str(skinned_fbx),
                    "target_path": str(target_mesh_path),
                    "output_path": str(textured_preview_output),
                    "export_format": "glb",
                }
                try:
                    for event in _iter_subprocess_stage(
                        "unirig_merge",
                        merge_preview_payload,
                        work_dir,
                        merge_preview_log_path,
                        session=session,
                    ):
                        if event["type"] == "log":
                            _append_rig_full_log(work_dir, "animation_preview_merge", event["text"])
                            status = _append_status(status, event["text"])
                            status = _trim_status(status)
                            yield (None, None, status)
                        elif event["type"] == "result":
                            merged_path = event["result"].get("output_path")
                            merged_candidate = merged_path or str(textured_preview_output)
                            normalized_merged = _normalize_model3d_preview_path(merged_candidate)
                            if normalized_merged:
                                textured_preview_glb = normalized_merged
                except Exception as merge_err:
                    status = _append_status(
                        status,
                        f"[animation_preview_merge] Warning: {type(merge_err).__name__}: {merge_err}",
                    )
                    status = _trim_status(status)
                    yield (None, None, status)

            if textured_preview_glb:
                animation_source_path = Path(textured_preview_glb)

            animation_payload = {
                "input_model_path": str(animation_source_path),
                "output_glb_path": str(work_dir / f"{skinned_fbx.stem}_anim_preview.glb"),
                "frame_end": 140,
                "animation_style": "dance",
                "animation_strength": 1.5,
            }
            try:
                for event in _iter_subprocess_stage(
                    "unirig_animation_preview",
                    animation_payload,
                    work_dir,
                    animation_log_path,
                    session=session,
                ):
                    if event["type"] == "log":
                        _append_rig_full_log(work_dir, "animation_preview", event["text"])
                        status = _append_status(status, event["text"])
                        status = _trim_status(status)
                        yield (None, None, status)
                    elif event["type"] == "result":
                        animation_preview_glb = event["result"].get("animation_preview_glb_path")
            except Exception as anim_err:
                status = _append_status(
                    status,
                    f"[animation_preview] Warning: {type(anim_err).__name__}: {anim_err}",
                )
                status = _trim_status(status)
                yield (None, None, status)

            metadata.setdefault("stages", {}).setdefault("skinning", {}).update(
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": result,
                }
            )
            metadata.setdefault("paths", {})["skinning_output"] = str(skinned_fbx)
            if textured_preview_glb:
                metadata.setdefault("paths", {})["textured_preview"] = str(Path(textured_preview_glb).resolve())
            if animation_preview_glb and os.path.exists(animation_preview_glb):
                resolved_anim_preview = str(Path(animation_preview_glb).resolve())
                metadata.setdefault("paths", {})["animation_preview"] = resolved_anim_preview
                if textured_preview_glb:
                    metadata.setdefault("paths", {})["textured_animation_preview"] = resolved_anim_preview
            _save_rig_metadata(work_dir, metadata)
            paths_meta = metadata.get("paths", {}) if isinstance(metadata, dict) else {}
            preview_source: Optional[str] = _normalize_model3d_preview_path(
                paths_meta.get("textured_animation_preview")
                or paths_meta.get("animation_preview")
                or paths_meta.get("textured_preview")
                or paths_meta.get("skeleton_preview")
            )
            note = "\nPreview: generated animation preview."
            if not preview_source:
                preview_source = _normalize_model3d_preview_path(animation_preview_glb)
            if not preview_source and skinned_fbx.suffix.lower() == ".fbx":
                preview_source = _select_rig_preview_source(metadata, None)
                if preview_source:
                    note = (
                        "\nPreview fallback: skinned FBX is not directly previewable in Model3D; "
                        "showing uploaded mesh."
                    )
                else:
                    preview_source = None
                    note = "\nNote: FBX preview is not supported by Gradio Model3D in this view."
            if textured_preview_glb:
                note += f"\nTextured preview ready: {Path(textured_preview_glb).name}"
            if animation_preview_glb and os.path.exists(animation_preview_glb):
                note += f"\nAnimation preview ready: {Path(animation_preview_glb).name}"
            final_status = _append_status(status, f"\n✅ Skinning completed!\nOutput: {skinned_fbx}{note}")
            yield (str(skinned_fbx), preview_source, final_status)
        else:
            metadata.setdefault("stages", {}).setdefault("skinning", {}).update(
                {
                    "status": "failed",
                    "completed_at": datetime.now().isoformat(),
                }
            )
            _save_rig_metadata(work_dir, metadata)
            yield (None, None, _append_status(status, "\n[ERROR] Skinning failed."))

    except Exception as e:
        if work_dir is not None and work_dir.exists():
            try:
                metadata = _load_rig_metadata(work_dir)
                metadata.setdefault("stages", {}).setdefault("skinning", {}).update(
                    {
                        "status": "failed",
                        "completed_at": datetime.now().isoformat(),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                _save_rig_metadata(work_dir, metadata)
                _append_rig_full_log(work_dir, "skinning", f"ERROR: {type(e).__name__}: {e}")
            except Exception:
                pass
        yield (None, None, f"[ERROR] Error: {type(e).__name__}: {e}")


def _run_unirig_merge(source_fbx_path: str, target_mesh_path: str, export_format: str, req: gr.Request):
    """
    Merge rigged skeleton with original mesh using UniRig via subprocess.
    """
    session = str(req.session_hash)
    work_dir: Optional[Path] = None

    try:
        if not source_fbx_path or not os.path.exists(source_fbx_path):
            yield (None, None, "[ERROR] Please generate skeleton/skinning first.")
            return

        if not target_mesh_path or not os.path.exists(target_mesh_path):
            yield (None, None, "[ERROR] Please upload target mesh.")
            return

        source_path = Path(source_fbx_path)
        target_path = Path(target_mesh_path)
        work_dir = _find_rig_work_dir(source_path)
        _ensure_rig_workspace_dirs(work_dir)

        ext = ".fbx" if export_format == "fbx" else ".glb"
        final_output = work_dir / f"{target_path.stem}_rigged{ext}"
        log_path = work_dir / "logs" / "merge_log.txt"

        metadata = _load_rig_metadata(work_dir)
        metadata.setdefault("stages", {})["merge"] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "log_path": str(log_path),
            "payload_path": str(work_dir / "unirig_merge.payload.json"),
            "result_path": str(work_dir / "unirig_merge.result.json"),
            "source_path": str(source_path),
            "target_path": str(target_path),
            "output_path": str(final_output),
            "export_format": export_format,
        }
        _save_rig_metadata(work_dir, metadata)

        yield (None, None, f"Merging rig with mesh...\nSource: {source_path.name}\nTarget: {target_path.name}")

        payload = {
            "source_path": str(source_path),
            "target_path": str(target_path),
            "output_path": str(final_output),
            "export_format": export_format,
        }

        status = ""
        result = None
        for event in _iter_subprocess_stage("unirig_merge", payload, work_dir, log_path, session=session):
            if event["type"] == "log":
                _append_rig_full_log(work_dir, "merge", event["text"])
                status = _append_status(status, event["text"])
                status = _trim_status(status)
                yield (None, None, status)
            elif event["type"] == "result":
                result = event["result"]

        metadata = _load_rig_metadata(work_dir)
        if result and final_output.exists():
            metadata.setdefault("stages", {}).setdefault("merge", {}).update(
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": result,
                }
            )
            metadata.setdefault("paths", {})["final_output"] = str(final_output)
            _save_rig_metadata(work_dir, metadata)
            preview_source: Optional[str] = _preferred_animation_preview_path(str(final_output))
            if not preview_source:
                preview_source = _normalize_model3d_preview_path(str(final_output))
            note = ""
            if final_output.suffix.lower() == ".fbx":
                if not preview_source:
                    preview_source = _select_rig_preview_source(metadata, None)
                if preview_source:
                    note = (
                        "\nNote: FBX preview may be blank in Gradio Model3D. "
                        "Showing uploaded mesh as preview fallback."
                    )
                else:
                    preview_source = None
                    note = "\nNote: FBX preview is not supported by Gradio Model3D in this view."
            final_status = _append_status(status, f"\n✅ Rigged model created!\nOutput: {final_output}{note}")
            yield (str(final_output), preview_source, final_status)
        else:
            metadata.setdefault("stages", {}).setdefault("merge", {}).update(
                {
                    "status": "failed",
                    "completed_at": datetime.now().isoformat(),
                }
            )
            _save_rig_metadata(work_dir, metadata)
            yield (None, None, _append_status(status, "\n[ERROR] Merge failed."))

    except Exception as e:
        if work_dir is not None and work_dir.exists():
            try:
                metadata = _load_rig_metadata(work_dir)
                metadata.setdefault("stages", {}).setdefault("merge", {}).update(
                    {
                        "status": "failed",
                        "completed_at": datetime.now().isoformat(),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                _save_rig_metadata(work_dir, metadata)
                _append_rig_full_log(work_dir, "merge", f"ERROR: {type(e).__name__}: {e}")
            except Exception:
                pass
        yield (None, None, f"[ERROR] Error: {type(e).__name__}: {e}")


def _send_rig_output_to_animation(
    final_output_path: Optional[str],
    skinned_path: Optional[str],
    skeleton_path: Optional[str],
    current_status: str,
):
    def _norm_path(path: str) -> str:
        try:
            return Path(path).resolve().as_posix()
        except Exception:
            return path

    previewable_exts = {".glb", ".gltf", ".obj", ".ply", ".stl", ".splat"}

    # Prefer merged output first (usually textured), then skinned, then skeleton.
    candidate = final_output_path or skinned_path or skeleton_path
    if not candidate:
        status = _append_status(current_status or "", "[ERROR] No rig output to send. Generate skeleton/skinning/export first.")
        return "", _trim_status(status), gr.update()
    if not os.path.exists(candidate):
        status = _append_status(current_status or "", f"[ERROR] Output file not found: {candidate}")
        return "", _trim_status(status), gr.update()

    candidate_path = Path(candidate)
    metadata = _load_rig_metadata_for_path(str(candidate_path))
    paths_meta = metadata.get("paths", {}) if isinstance(metadata, dict) else {}

    # Prefer textured animated previews when available.
    viewer_candidate = _normalize_model3d_preview_path(paths_meta.get("textured_animation_preview")) or candidate
    if viewer_candidate == candidate and candidate_path.suffix.lower() not in previewable_exts:
        viewer_candidate = _preferred_animation_preview_path(str(candidate_path)) or candidate

    if viewer_candidate == candidate and candidate_path.suffix.lower() == ".fbx":
        same_stem_glb = candidate_path.with_suffix(".glb")
        same_stem_gltf = candidate_path.with_suffix(".gltf")
        normalized_same_stem_glb = _normalize_model3d_preview_path(str(same_stem_glb))
        normalized_same_stem_gltf = _normalize_model3d_preview_path(str(same_stem_gltf))
        if normalized_same_stem_glb:
            viewer_candidate = normalized_same_stem_glb
        elif normalized_same_stem_gltf:
            viewer_candidate = normalized_same_stem_gltf
        elif final_output_path:
            normalized_final_output = _normalize_model3d_preview_path(final_output_path)
            if normalized_final_output:
                viewer_candidate = normalized_final_output
        if viewer_candidate == candidate:
            fallback_preview = _select_rig_preview_source(metadata, None)
            if fallback_preview:
                viewer_candidate = fallback_preview
    elif viewer_candidate == candidate and candidate_path.suffix.lower() in previewable_exts:
        normalized_candidate = _normalize_model3d_preview_path(str(candidate_path))
        if normalized_candidate:
            viewer_candidate = normalized_candidate

    viewer_candidate = _norm_path(viewer_candidate)
    status = _append_status(current_status or "", f"➡ Sent to Animation Browser: {Path(viewer_candidate).name}")
    return viewer_candidate, _trim_status(status), gr.Tabs(selected="animation_tab")


def _regenerate_animation_preview_for_browser(
    model_path: Optional[str],
    animation_style: str,
    animation_strength: float,
    frame_end: int,
    req: gr.Request,
) -> Tuple[str, str]:
    """Regenerate animation preview for Animation Browser and persist metadata links."""
    session = str(req.session_hash)

    if not model_path:
        return "", "ERROR: No model selected."

    selected_path = Path(model_path)
    if not selected_path.exists():
        return "", f"ERROR: Selected model file not found: {model_path}"

    work_dir = _find_rig_work_dir(selected_path)
    _ensure_rig_workspace_dirs(work_dir)
    log_path = work_dir / "logs" / "animation_preview_log.txt"

    metadata = _load_rig_metadata(work_dir)
    paths_meta = metadata.get("paths", {}) if isinstance(metadata, dict) else {}

    source_candidates = [
        paths_meta.get("textured_preview"),
        paths_meta.get("final_output"),
        paths_meta.get("animation_preview"),
        paths_meta.get("skinning_output"),
        str(selected_path),
    ]
    source_path: Optional[Path] = None
    for candidate in source_candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate).resolve()
        except Exception:
            continue
        if path.exists() and path.suffix.lower() in {".fbx", ".glb", ".gltf"}:
            source_path = path
            break

    if source_path is None:
        return "", "ERROR: No valid source model found for animation preview (.fbx/.glb/.gltf)."

    base_stem = source_path.stem
    skinning_output = paths_meta.get("skinning_output")
    if skinning_output:
        try:
            base_stem = Path(skinning_output).stem
        except Exception:
            base_stem = source_path.stem
    output_glb = work_dir / f"{base_stem}_anim_preview.glb"

    style = str(animation_style or "dance").strip().lower()
    if style not in {"walk", "dance", "idle"}:
        style = "dance"
    strength = float(animation_strength)
    strength = max(0.4, min(2.5, strength))
    frames = int(frame_end)
    frames = max(30, min(240, frames))

    metadata = _load_rig_metadata(work_dir)
    metadata.setdefault("stages", {})["animation_preview"] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "log_path": str(log_path),
        "payload_path": str(work_dir / "unirig_animation_preview.payload.json"),
        "result_path": str(work_dir / "unirig_animation_preview.result.json"),
        "input_model_path": str(source_path),
        "output_path": str(output_glb),
        "animation_style": style,
        "animation_strength": strength,
        "frame_end": frames,
    }
    _save_rig_metadata(work_dir, metadata)

    payload = {
        "input_model_path": str(source_path),
        "output_glb_path": str(output_glb),
        "frame_end": frames,
        "animation_style": style,
        "animation_strength": strength,
    }

    result = None
    try:
        for event in _iter_subprocess_stage("unirig_animation_preview", payload, work_dir, log_path, session=session):
            if event["type"] == "log":
                _append_rig_full_log(work_dir, "animation_preview", event["text"])
            elif event["type"] == "result":
                result = event["result"]
    except Exception as e:
        metadata = _load_rig_metadata(work_dir)
        metadata.setdefault("stages", {}).setdefault("animation_preview", {}).update(
            {
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": f"{type(e).__name__}: {e}",
            }
        )
        _save_rig_metadata(work_dir, metadata)
        return "", f"ERROR: Animation preview failed: {type(e).__name__}: {e}"

    if not output_glb.exists() or not result:
        metadata = _load_rig_metadata(work_dir)
        metadata.setdefault("stages", {}).setdefault("animation_preview", {}).update(
            {
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
            }
        )
        _save_rig_metadata(work_dir, metadata)
        return "", "ERROR: Animation preview stage completed but no output was found."

    resolved_output = str(output_glb.resolve())
    metadata = _load_rig_metadata(work_dir)
    metadata.setdefault("stages", {}).setdefault("animation_preview", {}).update(
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result": result,
        }
    )
    paths_meta = metadata.setdefault("paths", {})
    paths_meta["animation_preview"] = resolved_output
    if source_path.suffix.lower() in {".glb", ".gltf"} or paths_meta.get("textured_preview"):
        paths_meta["textured_animation_preview"] = resolved_output
    _save_rig_metadata(work_dir, metadata)

    status = (
        f"OK: Animation preview regenerated: {Path(resolved_output).name}\n"
        f"Style: {style} | Strength: {strength:.1f} | Frames: {frames}"
    )
    return Path(resolved_output).as_posix(), status


# ------------------------------- Cancellation -------------------------------

class UserCancelled(RuntimeError):
    """Raised when a user explicitly cancels a running operation."""


_CANCEL_LOCK = threading.Lock()

# Per-session running tasks (used to avoid "sticky" cancels when nothing is running).
_RUNNING_TASKS: Dict[str, set[str]] = {}

# Per-session cancellation flags.
_CANCEL_ALL: Dict[str, threading.Event] = {}
_CANCEL_BATCH: Dict[str, threading.Event] = {}

# Per-session active subprocess (subprocess-mode stages).
_ACTIVE_SUBPROCS: Dict[str, subprocess.Popen] = {}
_ACTIVE_SUBPROCS_STAGE: Dict[str, str] = {}


def _session_key(req: gr.Request) -> str:
    return str(req.session_hash)


def _get_or_create_event(store: Dict[str, threading.Event], session: str) -> threading.Event:
    with _CANCEL_LOCK:
        ev = store.get(session)
        if ev is None:
            ev = threading.Event()
            store[session] = ev
        return ev


def _is_cancel_all(session: str) -> bool:
    return _get_or_create_event(_CANCEL_ALL, session).is_set()


def _is_cancel_batch(session: str) -> bool:
    return _get_or_create_event(_CANCEL_BATCH, session).is_set()


def _request_cancel_all(session: str) -> None:
    _get_or_create_event(_CANCEL_ALL, session).set()


def _request_cancel_batch(session: str) -> None:
    _get_or_create_event(_CANCEL_BATCH, session).set()


def _clear_cancel_all(session: str) -> None:
    with _CANCEL_LOCK:
        ev = _CANCEL_ALL.get(session)
        if ev is not None:
            ev.clear()


def _clear_cancel_batch(session: str) -> None:
    with _CANCEL_LOCK:
        ev = _CANCEL_BATCH.get(session)
        if ev is not None:
            ev.clear()


def _mark_task_running(session: str, task: str, running: bool) -> None:
    with _CANCEL_LOCK:
        tasks = _RUNNING_TASKS.get(session)
        if tasks is None:
            tasks = set()
            _RUNNING_TASKS[session] = tasks
        if running:
            tasks.add(task)
        else:
            tasks.discard(task)
        if not tasks:
            _RUNNING_TASKS.pop(session, None)


def _is_any_task_running(session: str) -> bool:
    with _CANCEL_LOCK:
        return bool(_RUNNING_TASKS.get(session))


def _is_task_running(session: str, task: str) -> bool:
    with _CANCEL_LOCK:
        tasks = _RUNNING_TASKS.get(session)
        return bool(tasks and task in tasks)


def _register_active_subproc(session: str, proc: subprocess.Popen, stage: str) -> None:
    with _CANCEL_LOCK:
        _ACTIVE_SUBPROCS[session] = proc
        _ACTIVE_SUBPROCS_STAGE[session] = stage


def _unregister_active_subproc(session: str, proc: subprocess.Popen) -> None:
    with _CANCEL_LOCK:
        cur = _ACTIVE_SUBPROCS.get(session)
        if cur is proc:
            _ACTIVE_SUBPROCS.pop(session, None)
            _ACTIVE_SUBPROCS_STAGE.pop(session, None)


def _get_active_subproc(session: str) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    with _CANCEL_LOCK:
        return _ACTIVE_SUBPROCS.get(session), _ACTIVE_SUBPROCS_STAGE.get(session)


def _terminate_process(proc: subprocess.Popen) -> None:
    """
    Best-effort termination of a subprocess stage worker.
    - On POSIX: we start each worker in its own session and kill its process group.
    - On Windows: we terminate/kill the process (child processes are uncommon here).
    """
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return

    # First try graceful termination.
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        pass

    # Escalate to hard kill.
    try:
        if proc.poll() is None:
            if os.name == "nt":
                proc.kill()
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
            proc.wait(timeout=3)
    except Exception:
        pass


def _cancel_now(session: str, *, scope: str) -> str:
    """
    Trigger cancellation for this session.
    scope:
      - "batch": only batch runs should stop
      - "all": stop everything (and kill any active subprocess stage)
    Returns a human-friendly description of what was cancelled.
    """
    if scope not in {"batch", "all"}:
        scope = "batch"

    if scope == "batch":
        if not _is_task_running(session, "batch"):
            return "Nothing to cancel (batch is not running)."
        _request_cancel_batch(session)
        return "Cancel requested: batch processing."

    # scope == "all"
    _request_cancel_all(session)
    _request_cancel_batch(session)

    proc, stage = _get_active_subproc(session)
    if proc is None and not _is_task_running(session, "batch"):
        # Avoid "sticky" cancels that would affect the next run.
        _clear_cancel_all(session)
        _clear_cancel_batch(session)
        return "Nothing to cancel (no active subprocess stage detected)."
    if proc is not None:
        _terminate_process(proc)
        return f"Cancel requested: all processing (killed active subprocess stage: {stage or 'unknown'})."
    return "Cancel requested: all processing."


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


_BATCH_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_WIN_INVALID_NAME_CHARS = '<>:"/\\|?*'


def _resolve_user_path(path_str: Optional[str], *, base_dir: str) -> Optional[Path]:
    """
    Resolve a user-supplied path in a cross-platform way.
    - Empty/None => None
    - Relative paths => relative to `base_dir` (app folder)
    - Quoted paths are supported
    """
    if path_str is None:
        return None
    s = str(path_str).strip()
    if not s:
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if not s:
        return None
    if os.path.isabs(s):
        return Path(os.path.abspath(s))
    return Path(os.path.abspath(os.path.join(base_dir, s)))


def _sanitize_folder_name(name: str) -> str:
    """
    Make a reasonably safe folder name for Windows + Linux.
    """
    name = str(name or "").strip()
    # Windows disallows trailing dots/spaces for folder names.
    name = name.rstrip(" .")
    for ch in _WIN_INVALID_NAME_CHARS:
        name = name.replace(ch, "_")
    if not name:
        name = "run"
    # Very small reserved-name guard (Windows)
    upper = name.upper()
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if upper in reserved:
        name = f"_{name}"
    return name


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        return "?"
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _list_images_in_folder(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise gr.Error(f"Input folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _BATCH_IMAGE_EXTS]
    files.sort(key=lambda p: p.name.lower())
    return files


def _move_run_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    try:
        src.rename(dst)
    except Exception:
        shutil.move(str(src), str(dst))


def batch_process_folder(
    enabled: bool,
    input_folder: str,
    output_folder: str,
    randomize_seed: bool,
    seed: int,
    resolution: str,
    custom_resolution: int,
    pipeline_strategy: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_guidance_interval_start: float,
    ss_guidance_interval_end: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    force_high_res_conditional: bool,
    low_vram: bool,
    model_variant: str,
    attention_backend: str,
    sampler_type: str,
    use_chunked_processing: bool,
    use_tiled_extraction: bool,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_guidance_interval_start: float,
    shape_slat_guidance_interval_end: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_guidance_interval_start: float,
    tex_slat_guidance_interval_end: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    no_texture_gen: bool,
    deferred_texture_after_cleanup: bool,
    projection_texture_refine: bool,
    projection_view_azimuths: str,
    projection_view_elevations: str,
    projection_blend_exponent: float,
    projection_ortho_scale: float,
    projection_fill_holes: bool,
    projection_max_hole_size: int,
    max_num_tokens: int,
    multiview_mode: str,
    decimation_target: int,
    texture_size: int,
    remesh_method: str,
    fill_holes_max_perimeter: float,
    repair_method: str,
    simplify_method: str,
    prune_invisible_faces: bool,
    merge_vertices_dist: float,
    shade_smooth: bool,
    shade_smooth_angle: float,
    force_double_sided: bool,
    no_pbr_export: bool,
    export_formats: List[str],
    ultrashape_enabled: bool,
    ultrashape_retexture_after_refine: bool,
    ultrashape_conservative_mode: bool,
    ultrashape_checkpoint: str,
    ultrashape_config_name: str,
    ultrashape_dtype: str,
    ultrashape_low_vram: bool,
    ultrashape_remove_bg: bool,
    ultrashape_steps: int,
    ultrashape_guidance_scale: float,
    ultrashape_octree_resolution: int,
    ultrashape_num_chunks: int,
    ultrashape_target_face_count: int,
    ultrashape_num_latents: int,
    ultrashape_box_v: float,
    ultrashape_mc_level: float,
    ultrashape_normalize_scale: float,
    ultrashape_num_sharp_points: int,
    ultrashape_num_uniform_points: int,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    """
    Batch-process all images in a folder using the *same* pipeline + settings as single-image runs.
    Outputs are saved into per-image folders named after the input image (stem).
    """
    if not enabled:
        yield "Batch Processing is disabled."
        return

    session = _session_key(req)
    _mark_task_running(session, "batch", True)
    try:
        def _should_cancel() -> bool:
            return _is_cancel_all(session) or _is_cancel_batch(session)

        in_dir = _resolve_user_path(input_folder, base_dir=APP_DIR)
        if in_dir is None:
            raise gr.Error("Batch Input Folder is required.")
        out_root = _resolve_user_path(output_folder, base_dir=APP_DIR) or Path(OUTPUTS_DIR)

        files = _list_images_in_folder(in_dir)
        if not files:
            raise gr.Error(f"No supported images found in: {in_dir}\nSupported: {', '.join(sorted(_BATCH_IMAGE_EXTS))}")

        out_root.mkdir(parents=True, exist_ok=True)

        log_lines: List[str] = []
        current_desc: str = ""
        last_yield = 0.0
        started = time.time()
        processed = 0
        skipped = 0
        failed = 0
        total = len(files)

        def _append(line: str) -> None:
            nonlocal log_lines
            ts = datetime.now().strftime("%H:%M:%S")
            log_lines.append(f"[{ts}] {_clean_status_text(line)}")
            # Keep UI responsive (avoid giant payloads)
            if len(log_lines) > 400:
                log_lines = log_lines[-350:]

        def _render_status() -> str:
            done = processed + skipped + failed
            elapsed = time.time() - started
            remaining = max(0, total - done)
            avg = (elapsed / done) if done > 0 else None
            eta = (avg * remaining) if avg is not None else None
            summary = (
                f"Batch: {done}/{total} done | processed={processed}, skipped={skipped}, failed={failed} | "
                f"elapsed={_format_eta(elapsed)} | ETA={_format_eta(eta)}"
            )
            lines = [summary] + log_lines
            if current_desc:
                lines.append(f"Current: {current_desc}")
            return "\n".join(lines[-420:])

        def _maybe_yield(force: bool = False):
            nonlocal last_yield
            now = time.time()
            if force or (now - last_yield) > 0.7:
                last_yield = now
                return _render_status()
            return None

        _append(f"Input folder: {in_dir}")
        _append(f"Output folder: {out_root}")
        _append(f"Found {total} image(s). Starting…")
        progress(0.0, desc="Batch starting…")
        yield _render_status()

        if _should_cancel():
            _append("CANCELLED by user. Stopping batch.")
            progress(1.0, desc="Batch cancelled.")
            yield _render_status()
            return

        for i, img_path in enumerate(files, start=1):
            if _should_cancel():
                _append("CANCELLED by user. Stopping batch.")
                progress(1.0, desc="Batch cancelled.")
                yield _render_status()
                return

            name = _sanitize_folder_name(img_path.stem)
            target_dir = out_root / name

            # Update desc shown in the Gradio progress UI
            current_desc = f"[{i}/{total}] {img_path.name}"
            progress((i - 1) / total, desc=current_desc)

            if target_dir.exists():
                skipped += 1
                _append(f"SKIP [{i}/{total}] {img_path.name} → {target_dir} (already exists)")
                maybe = _maybe_yield(force=True)
                if maybe is not None:
                    yield maybe
                continue

            run_seed = get_seed(randomize_seed, int(seed))
            _append(f"RUN  [{i}/{total}] {img_path.name} (seed={run_seed})")
            maybe = _maybe_yield(force=True)
            if maybe is not None:
                yield maybe

            # Scale inner per-image progress into overall batch progress
            base = (i - 1) / total
            span = 1.0 / total

            def _scaled_progress(p: float, desc: Optional[str] = None):
                nonlocal current_desc
                if desc:
                    current_desc = str(desc)
                try:
                    pp = float(p)
                except Exception:
                    pp = 0.0
                pp = max(0.0, min(1.0, pp))
                progress(base + pp * span, desc=current_desc)

            # --- Run generate + extract using the same pipeline functions (no duplicated logic) ---
            try:
                with Image.open(str(img_path)) as im:
                    pil_img = im.convert("RGBA").copy()

                state: Optional[dict] = None
                for s, _html, _st in image_to_3d(
                    pil_img,
                    int(run_seed),
                    resolution,
                    custom_resolution,
                    pipeline_strategy,
                    ss_guidance_strength,
                    ss_guidance_rescale,
                    ss_guidance_interval_start,
                    ss_guidance_interval_end,
                    ss_sampling_steps,
                    ss_rescale_t,
                    force_high_res_conditional,
                    low_vram,
                    model_variant,
                    attention_backend,
                    sampler_type,
                    use_chunked_processing,
                    use_tiled_extraction,
                    shape_slat_guidance_strength,
                    shape_slat_guidance_rescale,
                    shape_slat_guidance_interval_start,
                    shape_slat_guidance_interval_end,
                    shape_slat_sampling_steps,
                    shape_slat_rescale_t,
                    tex_slat_guidance_strength,
                    tex_slat_guidance_rescale,
                    tex_slat_guidance_interval_start,
                    tex_slat_guidance_interval_end,
                    tex_slat_sampling_steps,
                    tex_slat_rescale_t,
                    no_texture_gen,
                    projection_texture_refine,
                    projection_view_azimuths,
                    projection_view_elevations,
                    projection_blend_exponent,
                    projection_ortho_scale,
                    projection_fill_holes,
                    projection_max_hole_size,
                    max_num_tokens,
                    multiview_mode,
                    subprocess_mode,
                    req,
                    skip_preview=True,
                    progress=_scaled_progress,
                ):
                    if s is not None:
                        state = s
                    maybe = _maybe_yield()
                    if maybe is not None:
                        yield maybe
                    if _should_cancel():
                        _append("CANCELLED by user. Stopping batch.")
                        progress(1.0, desc="Batch cancelled.")
                        yield _render_status()
                        return

                if not state or not isinstance(state, dict) or not state.get("_run_dir"):
                    raise gr.Error("Generation returned no valid state. See console/logs.")

                glb_path: Optional[str] = None
                for gp, _dl, _st in extract_glb(
                    state,
                    int(decimation_target),
                    int(texture_size),
                    str(remesh_method),
                    float(fill_holes_max_perimeter),
                    str(repair_method),
                    str(simplify_method),
                    bool(no_texture_gen),
                    bool(deferred_texture_after_cleanup),
                    bool(projection_texture_refine),
                    str(projection_view_azimuths),
                    str(projection_view_elevations),
                    float(projection_blend_exponent),
                    float(projection_ortho_scale),
                    bool(projection_fill_holes),
                    int(projection_max_hole_size),
                    bool(prune_invisible_faces),
                    float(merge_vertices_dist),
                    bool(shade_smooth),
                    float(shade_smooth_angle),
                    bool(force_double_sided),
                    bool(no_pbr_export),
                    export_formats,
                    extract_use_chunked_processing,
                    extract_use_tiled_extraction,
                    ultrashape_enabled,
                    ultrashape_retexture_after_refine,
                    ultrashape_conservative_mode,
                    ultrashape_checkpoint,
                    ultrashape_config_name,
                    ultrashape_dtype,
                    ultrashape_low_vram,
                    ultrashape_remove_bg,
                    ultrashape_steps,
                    ultrashape_guidance_scale,
                    ultrashape_octree_resolution,
                    ultrashape_num_chunks,
                    ultrashape_target_face_count,
                    ultrashape_num_latents,
                    ultrashape_box_v,
                    ultrashape_mc_level,
                    ultrashape_normalize_scale,
                    ultrashape_num_sharp_points,
                    ultrashape_num_uniform_points,
                    subprocess_mode,
                    req=req,
                    progress=_scaled_progress,
                ):
                    if gp:
                        glb_path = gp
                    maybe = _maybe_yield()
                    if maybe is not None:
                        yield maybe
                    if _should_cancel():
                        _append("CANCELLED by user. Stopping batch.")
                        progress(1.0, desc="Batch cancelled.")
                        yield _render_status()
                        return

                run_dir = Path(str(state.get("_run_dir")))
                _move_run_dir(run_dir, target_dir)
                processed += 1
                _append(f"DONE [{i}/{total}] Saved → {target_dir}")
                if glb_path:
                    # Note: glb_path points to the old location; after moving, it's still valid *as a file*, but path string differs.
                    pass
                yield _render_status()
            except UserCancelled:
                _append("CANCELLED by user. Stopping batch.")
                progress(1.0, desc="Batch cancelled.")
                yield _render_status()
                return
            except Exception as e:
                failed += 1
                _append(f"FAIL [{i}/{total}] {img_path.name}: {type(e).__name__}: {e}")
                yield _render_status()
                continue

        current_desc = ""
        progress(1.0, desc="Batch complete.")
        _append("Batch complete.")
        yield _render_status()
    finally:
        _mark_task_running(session, "batch", False)
        # Clear cancellation flags after the batch run ends so future runs work normally.
        _clear_cancel_batch(session)
        _clear_cancel_all(session)


def preprocess_image(
    image: Image.Image,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Image.Image:
    if image is None:
        raise gr.Error("Please provide an image.")

    if not subprocess_mode:
        # Used by Upload and Examples. On first run it may load the full pipeline.
        progress(0.05, desc="Loading Image→3D pipeline (TRELLIS.2-4B)…")
        pipe = get_image_pipeline()
        progress(0.2, desc="Preprocessing image (background removal / crop)…")
        out = pipe.preprocess_image(image)
        progress(1.0, desc="Image ready.")
        return out

    # Subprocess mode: run preprocessing in a short-lived worker process so the UI process keeps 0 VRAM.
    progress(0.02, desc="Starting subprocess: preprocess…")
    user_dir = Path(TMP_DIR) / str(req.session_hash) / "preprocess"
    user_dir.mkdir(parents=True, exist_ok=True)
    work_dir = user_dir / "work"
    log_path = user_dir / "preprocess.log"

    ts = int(time.time() * 1000)
    in_path = user_dir / f"input_{ts}.png"
    out_path = user_dir / f"preprocessed_{ts}.png"
    image.save(str(in_path))

    payload = {
        "model_repo": "microsoft/TRELLIS.2-4B",
        "input_image_path": str(in_path),
        "output_image_path": str(out_path),
    }

    last = ""
    for ev in _iter_subprocess_stage("preprocess_image", payload, work_dir, log_path, session=_session_key(req)):
        if ev["type"] == "log":
            last = ev["text"]
            # Keep UI responsive without spamming.
            progress(0.5, desc=(last[:120] + "…") if len(last) > 120 else last)
        else:
            result = ev["result"]
            out_path = Path(result["output_image_path"])

    progress(0.95, desc="Loading preprocessed image…")
    out_img = Image.open(str(out_path))
    progress(1.0, desc="Image ready.")
    return out_img


def preprocess_image_capture_raw(
    image: Image.Image,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Image.Image, Image.Image]:
    """
    Returns (preprocessed_image, raw_original_image).
    Used so we can save both into outputs/<run_id>/ when generating.
    """
    if image is None:
        raise gr.Error("Please provide an image.")
    raw = image.copy()
    processed = preprocess_image(image, subprocess_mode, req, progress)
    return processed, raw


def _coerce_uploaded_file_to_path(file_obj: Any) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "name", None) or str(file_obj)


def _coerce_uploaded_files_to_paths(files_obj: Any) -> List[str]:
    if not files_obj:
        return []
    if isinstance(files_obj, (str, dict)):
        files = [files_obj]
    else:
        files = list(files_obj)
    out: List[str] = []
    for item in files:
        p = _coerce_uploaded_file_to_path(item)
        if p:
            out.append(str(p))
    return out


def _update_uploaded_input_preview(files_obj: Any):
    paths = _coerce_uploaded_files_to_paths(files_obj)
    default_note = (
        "Upload one image for single-image mode. Upload multiple angles of the same object "
        "to enable multi-image mode. The first image is used as the primary view."
    )
    if not paths:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            gr.update(value=default_note, visible=True),
        )
    if len(paths) == 1:
        return (
            gr.update(value=str(paths[0]), visible=True),
            gr.update(value=[], visible=False),
            gr.update(value="Single-image mode ready: 1 image loaded.", visible=True),
        )
    return (
        gr.update(value=None, visible=False),
        gr.update(value=[str(path) for path in paths], visible=True),
        gr.update(
            value=(
                f"**Multi-image enabled**: {len(paths)} images loaded. "
                "The first image will be treated as the primary view."
            ),
            visible=True,
        ),
    )


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

def _normalize_pipeline_strategy(strategy: Optional[str]) -> str:
    strategy_norm = str(strategy or "reference_auto").strip().lower()
    if strategy_norm in PIPELINE_STRATEGY_CHOICES:
        return strategy_norm
    return "reference_auto"


def _get_pipeline_type(
    resolution_str: str,
    pipeline_strategy: Optional[str] = None,
) -> tuple[str, int]:
    """
    Convert resolution string to pipeline type and target resolution.
    
    Returns:
        (pipeline_type, target_resolution)
        
    Examples:
        "512" -> ("512", 512)
        "768" -> ("768_cascade", 768)
        "1024" -> ("1024_cascade", 1024) by default
        "1280" -> ("1280_cascade", 1280)
        "1536" -> ("1536_cascade", 1536)
        "2048" -> ("2048_cascade", 2048)
    """
    try:
        res = int(resolution_str)
    except (ValueError, TypeError):
        raise ValueError(f"Resolution must be a number, got: {resolution_str}")
    
    if res < 512:
        raise ValueError(f"Resolution must be >= 512, got: {res}")
    
    if res % 128 != 0:
        raise ValueError(f"Resolution must be divisible by 128, got: {res}")
    
    if res == 512:
        return "512", 512
    elif res == 1024:
        strategy_norm = _normalize_pipeline_strategy(pipeline_strategy)
        if strategy_norm == "direct_1024":
            return "1024", 1024
        if strategy_norm == "hybrid_512g_1024t":
            return "512g_1024t", 1024
        return "1024_cascade", 1024
    else:
        # Any other resolution uses cascade
        return f"{res}_cascade", res


def image_to_3d(
    image_or_files: Any,
    seed: int,
    resolution: str,
    custom_resolution: int,
    pipeline_strategy: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_guidance_interval_start: float,
    ss_guidance_interval_end: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    force_high_res_conditional: bool,
    low_vram: bool,
    model_variant: str,
    attention_backend: str,
    sampler_type: str,
    use_chunked_processing: bool,
    use_tiled_extraction: bool,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_guidance_interval_start: float,
    shape_slat_guidance_interval_end: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_guidance_interval_start: float,
    tex_slat_guidance_interval_end: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    no_texture_gen: bool,
    projection_texture_refine: bool,
    projection_view_azimuths: str,
    projection_view_elevations: str,
    projection_blend_exponent: float,
    projection_ortho_scale: float,
    projection_fill_holes: bool,
    projection_max_hole_size: int,
    max_num_tokens: int,
    multiview_mode: str,
    subprocess_mode: bool,
    req: gr.Request,
    skip_preview: bool = False,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[dict], str, str]:
    # Stream step-by-step status so users aren't "in the dark" during long runs.
    status = ""
    session = _session_key(req)

    # Mutable container for log file path (set after run_dir is allocated)
    _log_file_path: List[Optional[Path]] = [None]

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        msg = _clean_status_text(msg)
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        status = (status + "\n" if status else "") + line
        print(line, flush=True)
        # Stream to log file in real-time if path is set
        if _log_file_path[0] is not None:
            try:
                with open(_log_file_path[0], "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass  # Don't fail generation if logging fails
        if p is not None:
            progress(p, desc=msg)
        return status


    extra_view_paths: List[str] = []
    if isinstance(image_or_files, Image.Image):
        image = image_or_files.convert("RGBA").copy()
    else:
        uploaded_paths = _coerce_uploaded_files_to_paths(image_or_files)
        if not uploaded_paths:
            raise gr.Error("Please upload at least one image (or pick an example).")
        primary_path = Path(uploaded_paths[0])
        try:
            with Image.open(str(primary_path)) as im:
                image = im.convert("RGBA").copy()
        except Exception as e:
            raise gr.Error(f"Could not read input image: {primary_path} ({type(e).__name__}: {e})")
        extra_view_paths = uploaded_paths[1:]

    mv_mode = str(multiview_mode or "stochastic").strip().lower()
    if mv_mode not in {"stochastic", "multidiffusion"}:
        mv_mode = "stochastic"
    requested_model_variant = normalize_model_variant(model_variant)
    model_variant_cfg = resolve_model_variant(requested_model_variant)
    requested_attention_backend = str(attention_backend or DEFAULT_ATTENTION_BACKEND).strip().lower()
    requested_sampler_type = normalize_sampler_type(sampler_type)
    runtime_cfg = apply_runtime_backends(requested_attention_backend)
    runtime_cfg["model_variant"] = requested_model_variant
    runtime_cfg["model_repo"] = model_variant_cfg["model_repo"]
    runtime_cfg["config_file"] = model_variant_cfg["config_file"]
    runtime_cfg["sampler_type"] = requested_sampler_type

    # Multi-image generation: the first uploaded image is always view #1.
    source_images: List[Image.Image] = [image.convert("RGBA")]
    for p in extra_view_paths:
        try:
            with Image.open(str(p)) as im:
                source_images.append(im.convert("RGBA").copy())
        except Exception as e:
            raise gr.Error(f"Could not read multi-view image: {p} ({type(e).__name__}: {e})")
    use_multiview = len(source_images) > 1
    pipeline_strategy_norm = _normalize_pipeline_strategy(pipeline_strategy)
    if use_multiview and pipeline_strategy_norm == "hybrid_512g_1024t":
        _log(
            "Hybrid 512 geometry + 1024 texture mode is only supported for single-image Generate. "
            "Falling back to reference_auto for multi-view Generate.",
            0.0,
        )
        pipeline_strategy_norm = "reference_auto"
    projection_settings = {
        "enabled": bool(projection_texture_refine),
        "azimuths": [],
        "elevations": [],
        "blend_exponent": float(projection_blend_exponent),
        "ortho_scale": float(projection_ortho_scale),
        "fill_holes": bool(projection_fill_holes),
        "max_hole_size": int(projection_max_hole_size),
    }
    if projection_settings["enabled"]:
        try:
            azimuths, elevations = resolve_projection_views(
                [Path(f"view_{idx:02d}.png") for idx in range(len(source_images))],
                projection_view_azimuths,
                projection_view_elevations,
            )
        except Exception as e:
            raise gr.Error(f"Invalid projection view angles: {e}")
        projection_settings["azimuths"] = [float(v) for v in azimuths]
        projection_settings["elevations"] = [float(v) for v in elevations]

    # Handle custom resolution override
    if custom_resolution and custom_resolution > 0:
        resolution = str(custom_resolution)
        _log(f"Using custom resolution: {resolution}", 0.0)
    
    # Validate and get pipeline type
    try:
        pipeline_type, target_res = _get_pipeline_type(resolution, pipeline_strategy_norm)
    except ValueError as e:
        raise gr.Error(str(e))

    # Allocate an outputs run folder (never overwrites).
    run = allocate_run_dir(OUTPUTS_DIR, digits=4)
    run_dir = run.run_dir
    run_id = run.run_id
    logs_dir = ensure_dir(run_dir / "logs")
    _log_file_path[0] = run_dir / "running_logs.txt"  # Enable log file streaming

    # Persist the raw/preprocessed inputs so every run is inspectable.
    input_path = run_dir / "00_input.png"
    preprocessed_path = run_dir / "01_preprocessed.png"
    try:
        source_images[0].save(str(input_path))
    except Exception:
        # Don't fail the run just because saving failed; continue.
        pass
    if use_multiview:
        for i, img in enumerate(source_images[1:], start=2):
            try:
                img.save(str(run_dir / f"00_input_view{i:02d}.png"))
            except Exception:
                pass

    _log(f"Starting Image → 3D generation (resolution: {resolution}, pipeline: {pipeline_type})…", 0.0)
    _log(
        "Runtime settings: "
        f"requested_backend={requested_attention_backend}, "
        f"dense_backend={runtime_cfg['dense_backend']}, "
        f"sparse_backend={runtime_cfg['sparse_backend']}, "
        f"sampler={requested_sampler_type}",
        0.0,
    )
    if use_multiview:
        _log(
            f"Multi-view input detected: {len(source_images)} views (mode={mv_mode}). "
            "View #1 is the first uploaded image.",
            0.0,
        )
    yield None, empty_html, gr.update(value=_trim_status(status), visible=True)

    if subprocess_mode:
        # Unload any global pipelines to free VRAM for subprocess
        unload_global_pipelines()

        # Subprocess stage pipeline (zero VRAM kept by the UI process).
        _log(f"Subprocess mode ON. Run: {run_id} → {safe_relpath(run_dir, APP_DIR)}", 0.01)
        yield None, empty_html, status

        work_dir = Path(TMP_DIR) / str(req.session_hash) / "subprocess" / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Artifact paths (saved under outputs/<run_id>/)
        cond_512_path = run_dir / "02_cond_512.pt"
        cond_1024_path = (run_dir / "03_cond_1024.pt") if pipeline_type != "512" else None
        coords_path = run_dir / "04_coords.pt"
        shape_slat_path = run_dir / "05_shape_slat.npz"
        shape_res_path = run_dir / "05_shape_res.json"
        tex_slat_path = None if no_texture_gen else (run_dir / "06_tex_slat.npz")
        preview_mesh_blob_path = run_dir / "07_preview_mesh_voxel.pt"
        preview_dir = run_dir / "07_preview"
        preview_manifest_path = run_dir / "07_preview_manifest.json"
        preview_html_path = run_dir / "07_preview.html"

        # Record parameters for reproducibility.
        _write_json(
            str(run_dir / "run.json"),
            {
                "run_id": run_id,
                "type": "image_to_3d",
                "subprocess_mode": True,
                "seed": int(seed),
                "resolution": resolution,
                "pipeline_strategy": pipeline_strategy_norm,
                "pipeline_type": pipeline_type,
                "multiview": bool(use_multiview),
                "multiview_mode": mv_mode if use_multiview else None,
                "view_count": int(len(source_images)),
                "no_texture_gen": bool(no_texture_gen),
                "max_num_tokens": int(max_num_tokens),
                "force_high_res_conditional": bool(force_high_res_conditional),
                "low_vram": bool(low_vram),
                "model_variant": requested_model_variant,
                "model_repo": model_variant_cfg["model_repo"],
                "attention_backend": requested_attention_backend,
                "dense_attention_backend": runtime_cfg["dense_backend"],
                "sparse_attention_backend": runtime_cfg["sparse_backend"],
                "sampler_type": requested_sampler_type,
                "use_chunked_processing": bool(use_chunked_processing),
                "use_tiled_extraction": bool(use_tiled_extraction),
                "ss_params": {
                    "steps": int(ss_sampling_steps),
                    "guidance_strength": float(ss_guidance_strength),
                    "guidance_rescale": float(ss_guidance_rescale),
                    "guidance_interval": [float(ss_guidance_interval_start), float(ss_guidance_interval_end)],
                    "rescale_t": float(ss_rescale_t),
                },
                "shape_params": {
                    "steps": int(shape_slat_sampling_steps),
                    "guidance_strength": float(shape_slat_guidance_strength),
                    "guidance_rescale": float(shape_slat_guidance_rescale),
                    "guidance_interval": [float(shape_slat_guidance_interval_start), float(shape_slat_guidance_interval_end)],
                    "rescale_t": float(shape_slat_rescale_t),
                },
                "tex_params": {
                    "steps": int(tex_slat_sampling_steps),
                    "guidance_strength": float(tex_slat_guidance_strength),
                    "guidance_rescale": float(tex_slat_guidance_rescale),
                    "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                    "rescale_t": float(tex_slat_rescale_t),
                },
                "projection_texture_refine": bool(projection_settings["enabled"]),
                "projection_azimuths": list(projection_settings["azimuths"]),
                "projection_elevations": list(projection_settings["elevations"]),
                "projection_blend_exponent": float(projection_settings["blend_exponent"]),
                "projection_ortho_scale": float(projection_settings["ortho_scale"]),
                "projection_fill_holes": bool(projection_settings["fill_holes"]),
                "projection_max_hole_size": int(projection_settings["max_hole_size"]),
            },
        )

        # Helper: run a subprocess stage with light streaming updates.
        last_ui_update = 0.0

        def _stage(stage_name: str, payload: dict, p: float) -> dict:
            nonlocal status, last_ui_update
            _log(f"Starting stage: {stage_name}", p)
            yield None, empty_html, status

            log_path = Path(logs_dir) / f"{stage_name}.log"
            result = None
            for ev in _iter_subprocess_stage(stage_name, payload, work_dir, log_path, session=_session_key(req)):
                if ev["type"] == "log":
                    # Append a small subset of logs to the UI box (keeps it readable).
                    line = ev["text"]
                    if line:
                        status = status + "\n" + line
                        status = _trim_status(status)
                        # Also write to main generation log for consolidated view
                        if _log_file_path[0] is not None:
                            try:
                                with open(_log_file_path[0], "a", encoding="utf-8") as f:
                                    f.write(line + "\n")
                            except Exception:
                                pass
                    now = time.time()
                    if now - last_ui_update > 0.6:
                        last_ui_update = now
                        yield None, empty_html, status
                else:
                    result = ev["result"]
            if result is None:
                raise gr.Error(f"Stage {stage_name} produced no result.")
            return result

        def _cancelled_exit(msg: str = "CANCELLED by user."):
            nonlocal status
            # Make sure the UI gets a final line instead of a stack trace.
            _log(msg, 0.0)
            yield None, empty_html, status
            _clear_cancel_all(session)
            _clear_cancel_batch(session)

        # Stage: preprocess image(s) (writes 01_preprocessed*.png)
        preprocess_inputs: List[Path] = []
        for src_idx, src_image in enumerate(source_images, start=1):
            in_path = input_path if src_idx == 1 else (run_dir / f"00_input_view{src_idx:02d}.png")
            if not in_path.exists():
                try:
                    src_image.save(str(in_path))
                except Exception as e:
                    raise gr.Error(
                        f"Could not save source view {src_idx} for subprocess preprocessing: "
                        f"{type(e).__name__}: {e}"
                    )
            preprocess_inputs.append(in_path)
        preprocessed_view_paths: List[Path] = []
        num_preprocess_views = len(preprocess_inputs)
        for view_idx, in_path in enumerate(preprocess_inputs, start=1):
            out_path = preprocessed_path if view_idx == 1 else (run_dir / f"01_preprocessed_view{view_idx:02d}.png")
            preprocess_progress = 0.05 + 0.02 * ((view_idx - 1) / max(1, num_preprocess_views - 1))
            if use_multiview:
                _log(f"Preprocessing view {view_idx}/{num_preprocess_views}…", preprocess_progress)
            preprocess_payload = {
                "model_repo": model_variant_cfg["model_repo"],
                "config_file": model_variant_cfg["config_file"],
                "input_image_path": str(in_path),
                "output_image_path": str(out_path),
            }
            try:
                _ = yield from _stage("preprocess_image", preprocess_payload, preprocess_progress)
            except UserCancelled:
                yield from _cancelled_exit()
                return
            preprocessed_view_paths.append(out_path)

        if use_multiview:
            # Multi-view subprocess stage: sample sparse/shape/tex latents from all views.
            multiview_payload = {
                "model_repo": model_variant_cfg["model_repo"],
                "config_file": model_variant_cfg["config_file"],
                "seed": int(seed),
                "resolution": resolution,
                "pipeline_strategy": pipeline_strategy_norm,
                "image_paths": [str(p) for p in preprocessed_view_paths],
                "shape_slat_path": str(shape_slat_path),
                "out_res_path": str(shape_res_path),
                "tex_slat_path": str(tex_slat_path) if tex_slat_path is not None else None,
                "multiview_mode": mv_mode,
                "max_num_tokens": int(max_num_tokens),
                "no_texture_gen": bool(no_texture_gen),
                "low_vram": bool(low_vram),
                "attention_backend": requested_attention_backend,
                "sampler_type": requested_sampler_type,
                "ss_params": {
                    "steps": int(ss_sampling_steps),
                    "guidance_strength": float(ss_guidance_strength),
                    "guidance_rescale": float(ss_guidance_rescale),
                    "guidance_interval": [float(ss_guidance_interval_start), float(ss_guidance_interval_end)],
                    "rescale_t": float(ss_rescale_t),
                },
                "shape_params": {
                    "steps": int(shape_slat_sampling_steps),
                    "guidance_strength": float(shape_slat_guidance_strength),
                    "guidance_rescale": float(shape_slat_guidance_rescale),
                    "guidance_interval": [float(shape_slat_guidance_interval_start), float(shape_slat_guidance_interval_end)],
                    "rescale_t": float(shape_slat_rescale_t),
                },
                "tex_params": {
                    "steps": int(tex_slat_sampling_steps),
                    "guidance_strength": float(tex_slat_guidance_strength),
                    "guidance_rescale": float(tex_slat_guidance_rescale),
                    "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                    "rescale_t": float(tex_slat_rescale_t),
                },
            }
            try:
                mv_result = yield from _stage("sample_multiview_latents", multiview_payload, 0.58)
            except UserCancelled:
                yield from _cancelled_exit()
                return
            res = int(mv_result["res"])
        else:
            # Stage: encode conditioning
            cond_payload = {
                "model_repo": model_variant_cfg["model_repo"],
                "config_file": model_variant_cfg["config_file"],
                "image_path": str(preprocessed_path),
                "resolution": resolution,
                "pipeline_strategy": pipeline_strategy_norm,
                "cond_512_path": str(cond_512_path),
                "cond_1024_path": str(cond_1024_path) if cond_1024_path is not None else None,
                "force_high_res_conditional": bool(force_high_res_conditional),
                "attention_backend": requested_attention_backend,
                "sampler_type": requested_sampler_type,
            }
            try:
                _ = yield from _stage("encode_cond", cond_payload, 0.08)
            except UserCancelled:
                yield from _cancelled_exit()
                return

            # Track RNG across subprocess stages so results match the single-process reference pipeline
            # for a given seed (instead of re-seeding each stage and changing the noise sequence).
            rng_after_sparse_path = run_dir / "04_rng_after_sparse.pt"
            rng_after_shape_path = run_dir / "05_rng_after_shape.pt"

            # Stage: sparse structure
            sparse_payload = {
                "model_repo": model_variant_cfg["model_repo"],
                "config_file": model_variant_cfg["config_file"],
                "seed": int(seed),
                "resolution": resolution,
                "pipeline_strategy": pipeline_strategy_norm,
                "cond_512_path": str(cond_512_path),
                "coords_path": str(coords_path),
                "force_high_res_conditional": bool(force_high_res_conditional),
                "rng_state_out_path": str(rng_after_sparse_path),
                "attention_backend": requested_attention_backend,
                "sampler_type": requested_sampler_type,
                "ss_params": {
                    "steps": int(ss_sampling_steps),
                    "guidance_strength": float(ss_guidance_strength),
                    "guidance_rescale": float(ss_guidance_rescale),
                    "guidance_interval": [float(ss_guidance_interval_start), float(ss_guidance_interval_end)],
                    "rescale_t": float(ss_rescale_t),
                },
            }
            try:
                _ = yield from _stage("sample_sparse_structure", sparse_payload, 0.18)
            except UserCancelled:
                yield from _cancelled_exit()
                return

            # Stage: shape latent
            shape_payload = {
                "model_repo": model_variant_cfg["model_repo"],
                "config_file": model_variant_cfg["config_file"],
                "seed": int(seed),
                "resolution": resolution,
                "pipeline_strategy": pipeline_strategy_norm,
                "cond_512_path": str(cond_512_path),
                "cond_1024_path": str(cond_1024_path) if cond_1024_path is not None else None,
                "coords_path": str(coords_path),
                "shape_slat_path": str(shape_slat_path),
                "out_res_path": str(shape_res_path),
                "rng_state_in_path": str(rng_after_sparse_path),
                "rng_state_out_path": str(rng_after_shape_path),
                "attention_backend": requested_attention_backend,
                "sampler_type": requested_sampler_type,
                "shape_params": {
                    "steps": int(shape_slat_sampling_steps),
                    "guidance_strength": float(shape_slat_guidance_strength),
                    "guidance_rescale": float(shape_slat_guidance_rescale),
                    "guidance_interval": [float(shape_slat_guidance_interval_start), float(shape_slat_guidance_interval_end)],
                    "rescale_t": float(shape_slat_rescale_t),
                },
                "max_num_tokens": int(max_num_tokens),
            }
            try:
                shape_result = yield from _stage("sample_shape_slat", shape_payload, 0.40)
            except UserCancelled:
                yield from _cancelled_exit()
                return
            res = int(shape_result["res"])

            # Stage: texture latent (optional)
            if not no_texture_gen:
                tex_cond_path = str(cond_512_path if pipeline_type == "512" else cond_1024_path)
                tex_payload = {
                    "model_repo": model_variant_cfg["model_repo"],
                    "config_file": model_variant_cfg["config_file"],
                    "seed": int(seed),
                    "resolution": resolution,
                    "pipeline_strategy": pipeline_strategy_norm,
                    "cond_path": tex_cond_path,
                    "shape_slat_path": str(shape_slat_path),
                    "tex_slat_path": str(tex_slat_path),
                    "rng_state_in_path": str(rng_after_shape_path),
                    "attention_backend": requested_attention_backend,
                    "sampler_type": requested_sampler_type,
                    "tex_params": {
                        "steps": int(tex_slat_sampling_steps),
                        "guidance_strength": float(tex_slat_guidance_strength),
                        "guidance_rescale": float(tex_slat_guidance_rescale),
                        "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                        "rescale_t": float(tex_slat_rescale_t),
                    },
                }
                try:
                    _ = yield from _stage("sample_tex_slat", tex_payload, 0.58)
                except UserCancelled:
                    yield from _cancelled_exit()
                    return

        # Stage: preview render (writes JPEGs + manifest)
        # NOTE: render_preview is optional for GLB extraction. If it fails (e.g., OOM),
        # we can still proceed to GLB extraction as long as the latents are saved.
        # In batch mode, we skip preview generation entirely to save time and memory.
        preview_failed = skip_preview
        preview_error_msg = "Preview skipped (batch mode)" if skip_preview else ""
        
        if not skip_preview:
            try:
                _ = yield from _stage(
                    "preview_decode_mesh",
                    {
                        "model_repo": model_variant_cfg["model_repo"],
                        "config_file": model_variant_cfg["config_file"],
                        "shape_slat_path": str(shape_slat_path),
                        "tex_slat_path": str(tex_slat_path) if tex_slat_path is not None else None,
                        "res": int(res),
                        "no_texture_gen": bool(no_texture_gen),
                        "use_chunked_processing": bool(use_chunked_processing),
                        "use_tiled_extraction": bool(use_tiled_extraction),
                        "mesh_blob_path": str(preview_mesh_blob_path),
                    },
                    0.78,
                )
                _ = yield from _stage(
                    "preview_render_mesh",
                    {
                        "mesh_blob_path": str(preview_mesh_blob_path),
                        "preview_dir": str(preview_dir),
                        "preview_manifest_path": str(preview_manifest_path),
                    },
                    0.82,
                )
            except UserCancelled:
                yield from _cancelled_exit()
                return
            except Exception as e:
                # Check if we have enough latents to proceed without preview
                has_shape = shape_slat_path.exists()
                has_tex = (tex_slat_path is not None and tex_slat_path.exists()) or no_texture_gen
                if has_shape and has_tex:
                    preview_failed = True
                    preview_error_msg = f"{type(e).__name__}: {e}"
                    _log(f"WARNING: Preview rendering failed: {preview_error_msg}", 0.85)
                    _log("Latents saved successfully. You can still extract GLB!", 0.86)
                else:
                    # No latents, re-raise the error
                    raise
        else:
            _log("Skipping preview generation (batch mode)", 0.82)

        # Handle preview failure - return minimal state that allows GLB extraction
        if preview_failed:
            state = {
                "_mode": "subprocess",
                "_run_id": run_id,
                "_run_dir": str(run_dir),
                "_input_image_path": str(input_path),
                "_preprocessed_image_path": str(preprocessed_path),
                "_preprocessed_view_paths": [str(p) for p in preprocessed_view_paths],
                "_pipeline_type": pipeline_type,
                "seed": int(seed),
                "res": int(res),
                "shape_slat_path": str(shape_slat_path),
                "tex_slat_path": str(tex_slat_path) if tex_slat_path is not None else None,
                "_model_variant": requested_model_variant,
                "_gen_tex_params": {
                    "steps": int(tex_slat_sampling_steps),
                    "guidance_strength": float(tex_slat_guidance_strength),
                    "guidance_rescale": float(tex_slat_guidance_rescale),
                    "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                    "rescale_t": float(tex_slat_rescale_t),
                    "attention_backend": requested_attention_backend,
                    "sampler_type": requested_sampler_type,
                    "dense_attention_backend": runtime_cfg["dense_backend"],
                    "sparse_attention_backend": runtime_cfg["sparse_backend"],
                },
                "_runtime_settings": {
                    "model_variant": requested_model_variant,
                    "model_repo": model_variant_cfg["model_repo"],
                    "config_file": model_variant_cfg["config_file"],
                    "attention_backend": requested_attention_backend,
                    "dense_attention_backend": runtime_cfg["dense_backend"],
                    "sparse_attention_backend": runtime_cfg["sparse_backend"],
                    "sampler_type": requested_sampler_type,
                },
                "_projection_settings": dict(projection_settings),
                "preview_manifest_path": None,  # No preview available
                "_preview_failed": True,
                "_preview_error": preview_error_msg,
            }
            if use_multiview:
                state["_multiview"] = True
                state["_multiview_mode"] = mv_mode
                state["_multiview_count"] = int(len(source_images))
            # Create an error HTML that shows the logs and allows extraction
            error_html = f"""
            <div class="previewer-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; padding: 20px;">
                <div style="background: rgba(255, 100, 100, 0.15); border: 1px solid rgba(255, 100, 100, 0.4); border-radius: 8px; padding: 20px; max-width: 600px; text-align: center;">
                    <h3 style="color: #ff6b6b; margin: 0 0 10px 0;">WARNING: Preview Rendering Failed</h3>
                    <p style="color: #ccc; margin: 0 0 15px 0; font-size: 14px;">
                        {preview_error_msg[:200]}{'...' if len(preview_error_msg) > 200 else ''}
                    </p>
                    <p style="color: #8f8; margin: 0; font-size: 14px;">
                        <strong>Good news:</strong> Latents were saved successfully!<br>
                        Click <strong>"Extract GLB"</strong> to generate your 3D model.
                    </p>
                </div>
            </div>
            """
            _log('Done (preview skipped). Click "Extract GLB" to generate model.', 1.0)
            # Keep status visible so user can see what happened
            yield state, error_html, gr.update(value=_trim_status(status), visible=True)
            return

        # Build the HTML preview from the saved JPEGs (CPU-only).
        _log("Building preview UI…", 0.96)
        _ensure_mode_icons()
        manifest = _read_json(str(preview_manifest_path))
        files = manifest.get("files", {})

        images_html = ""
        for m_idx, mode in enumerate(MODES):
            render_key = mode["render_key"]
            for s_idx in range(STEPS):
                unique_id = f"view-m{m_idx}-s{s_idx}"
                is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
                vis_class = "visible" if is_visible else ""
                img_path = files.get(render_key, [None] * STEPS)[s_idx]
                if img_path:
                    img_base64 = _jpeg_file_to_data_uri(img_path)
                else:
                    img_base64 = _image_to_base64(Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8)))
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
                    <p>- <b>Render Mode</b> - Click a circular button to switch render modes.</p>
                    <p>- <b>View Angle</b> - Drag the slider to change view angle.</p>
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
        preview_html_path.write_text(full_html, encoding="utf-8")

        state = {
            "_mode": "subprocess",
            "_run_id": run_id,
            "_run_dir": str(run_dir),
            "_input_image_path": str(input_path),
            "_preprocessed_image_path": str(preprocessed_path),
            "_preprocessed_view_paths": [str(p) for p in preprocessed_view_paths],
            "_pipeline_type": pipeline_type,
            "seed": int(seed),
            "res": int(res),
            "shape_slat_path": str(shape_slat_path),
            "tex_slat_path": str(tex_slat_path) if tex_slat_path is not None else None,
            "_model_variant": requested_model_variant,
            "_gen_tex_params": {
                "steps": int(tex_slat_sampling_steps),
                "guidance_strength": float(tex_slat_guidance_strength),
                "guidance_rescale": float(tex_slat_guidance_rescale),
                "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                "rescale_t": float(tex_slat_rescale_t),
                "attention_backend": requested_attention_backend,
                "sampler_type": requested_sampler_type,
                "dense_attention_backend": runtime_cfg["dense_backend"],
                "sparse_attention_backend": runtime_cfg["sparse_backend"],
            },
            "_runtime_settings": {
                "model_variant": requested_model_variant,
                "model_repo": model_variant_cfg["model_repo"],
                "config_file": model_variant_cfg["config_file"],
                "attention_backend": requested_attention_backend,
                "dense_attention_backend": runtime_cfg["dense_backend"],
                "sparse_attention_backend": runtime_cfg["sparse_backend"],
                "sampler_type": requested_sampler_type,
            },
            "_projection_settings": dict(projection_settings),
            "preview_manifest_path": str(preview_manifest_path),
        }
        _log('Done. You can now click "Extract GLB".', 1.0)
        if use_multiview:
            state["_multiview"] = True
            state["_multiview_mode"] = mv_mode
            state["_multiview_count"] = int(len(source_images))
        # Hide the overlay once preview is ready so users can see the render.
        yield state, full_html, gr.update(value=_trim_status(status), visible=False)
        return

    if use_multiview:
        _log("Loading TRELLIS.2 multi-view pipeline (first run can take a while)…", 0.01)
        pipe = get_multiview_pipeline(requested_model_variant)
    else:
        _log("Loading TRELLIS.2 pipeline (first run can take a while)…", 0.01)
        pipe = get_image_pipeline(requested_model_variant)
    runtime_cfg = _configure_image_pipeline_runtime(
        pipe,
        attention_backend=requested_attention_backend,
        sampler_type=requested_sampler_type,
    )
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

    # Persist per-stage artifacts for this run (even in in-process mode).
    cond_512_path = run_dir / "02_cond_512.pt"
    cond_1024_path = (run_dir / "03_cond_1024.pt") if pipeline_type != "512" else None
    coords_path = run_dir / "04_coords.pt"
    shape_slat_path = run_dir / "05_shape_slat.npz"
    shape_res_path = run_dir / "05_shape_res.json"
    tex_slat_path = None if no_texture_gen else (run_dir / "06_tex_slat.npz")
    preview_dir = run_dir / "07_preview"
    preview_manifest_path = run_dir / "07_preview_manifest.json"
    preview_html_path = run_dir / "07_preview.html"

    ss_params = {
        "steps": ss_sampling_steps,
        "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale,
        "guidance_interval": [float(ss_guidance_interval_start), float(ss_guidance_interval_end)],
        "rescale_t": ss_rescale_t,
    }
    shape_params = {
        "steps": shape_slat_sampling_steps,
        "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale,
        "guidance_interval": [float(shape_slat_guidance_interval_start), float(shape_slat_guidance_interval_end)],
        "rescale_t": shape_slat_rescale_t,
    }
    tex_params = {
        "steps": tex_slat_sampling_steps,
        "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale,
        "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
        "rescale_t": tex_slat_rescale_t,
    }

    # Preprocess (rembg + crop) at generate-time so examples/uploads behave consistently.
    preprocessed_views: List[Image.Image] = []
    n_views = len(source_images)
    if use_multiview:
        _log(f"Preprocessing {n_views} views (background removal / crop)…", 0.06)
    else:
        _log("Preprocessing image (background removal / crop)…", 0.06)
    for i, src in enumerate(source_images, start=1):
        if use_multiview:
            p = 0.06 + 0.02 * ((i - 1) / max(1, n_views - 1))
            _log(f"Preprocessing view {i}/{n_views}…", p)
        out = pipe.preprocess_image(src)
        preprocessed_views.append(out)
        try:
            if i == 1:
                out.save(str(preprocessed_path))
            else:
                out.save(str(run_dir / f"01_preprocessed_view{i:02d}.png"))
        except Exception:
            pass
    image = preprocessed_views[0]
    yield None, empty_html, status

    mesh = None
    if use_multiview:
        _log(f"Running multi-view generation ({n_views} views, mode={mv_mode})…", 0.18)
        mv_out, latent = pipe.run_multi_image(
            preprocessed_views,
            num_samples=1,
            seed=int(seed),
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=tex_params,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=pipeline_type,
            max_num_tokens=int(max_num_tokens),
            mode=mv_mode,
            no_texture_gen=bool(no_texture_gen),
        )
        mesh = mv_out[0] if isinstance(mv_out, list) else mv_out
        shape_slat, tex_slat, res = latent
        _log("Multi-view latent sampling complete.", 0.62)
        yield None, empty_html, status

        # Save latents for inspection / later extraction.
        try:
            np.savez_compressed(
                str(shape_slat_path),
                feats=shape_slat.feats.detach().cpu().numpy(),
                coords=shape_slat.coords.detach().cpu().numpy(),
            )
            _write_json(
                str(shape_res_path),
                {
                    "res": int(res),
                    "pipeline_type": pipeline_type,
                    "multiview": True,
                    "view_count": int(n_views),
                    "multiview_mode": mv_mode,
                },
            )
            if tex_slat is not None and tex_slat_path is not None:
                np.savez_compressed(
                    str(tex_slat_path),
                    feats=tex_slat.feats.detach().cpu().numpy(),
                    coords=tex_slat.coords.detach().cpu().numpy(),
                )
        except Exception:
            pass

        # run_multi_image already decodes; only re-decode when user explicitly requested decode options.
        if use_tiled_extraction or use_chunked_processing:
            _log("Applying decode settings to multiview latents…", 0.75)
            mesh = pipe.decode_latent(shape_slat, tex_slat, res, use_tiled_extraction, use_chunked_processing)[0]
            yield None, empty_html, status
        else:
            _log("Using multiview decoded mesh.", 0.75)
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
            images = {m["render_key"]: [np.zeros((1024, 1024, 3), dtype=np.uint8) for _ in range(STEPS)] for m in MODES}
        yield None, empty_html, status

        # Save preview frames to disk + manifest.
        try:
            preview_dir.mkdir(parents=True, exist_ok=True)
            manifest_files: Dict[str, List[str]] = {}
            for m_idx, mode in enumerate(MODES):
                key = mode["render_key"]
                manifest_files[key] = []
                for s_idx in range(STEPS):
                    path = preview_dir / f"view-m{m_idx}-s{s_idx}.jpg"
                    Image.fromarray(images[key][s_idx]).save(str(path), format="JPEG", quality=85)
                    manifest_files[key].append(str(path))
            _write_json(
                str(preview_manifest_path),
                {
                    "modes": [{"name": m["name"], "render_key": m["render_key"]} for m in MODES],
                    "steps": STEPS,
                    "files": manifest_files,
                },
            )
        except Exception:
            pass

        _log("Packing generation state (for GLB extraction)…", 0.93)
        state = pack_state((shape_slat, tex_slat, res))
        torch.cuda.empty_cache()
        yield None, empty_html, status

        _log("Building preview UI…", 0.97)
        images_html = ""
        for m_idx, mode in enumerate(MODES):
            for s_idx in range(STEPS):
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
                    <p>- <b>Render Mode</b> - Click a circular button to switch render modes.</p>
                    <p>- <b>View Angle</b> - Drag the slider to change view angle.</p>
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

        try:
            preview_html_path.write_text(full_html, encoding="utf-8")
        except Exception:
            pass
        try:
            state["_mode"] = "inproc"
            state["_run_id"] = run_id
            state["_run_dir"] = str(run_dir)
            state["_input_image_path"] = str(input_path)
            state["_preprocessed_image_path"] = str(preprocessed_path)
            state["_pipeline_type"] = pipeline_type
            state["_pipeline_strategy"] = pipeline_strategy_norm
            state["seed"] = int(seed)
            state["shape_slat_path"] = str(shape_slat_path)
            state["tex_slat_path"] = str(tex_slat_path) if tex_slat_path is not None else None
            state["_gen_tex_params"] = {
                "steps": int(tex_slat_sampling_steps),
                "guidance_strength": float(tex_slat_guidance_strength),
                "guidance_rescale": float(tex_slat_guidance_rescale),
                "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                "rescale_t": float(tex_slat_rescale_t),
                "attention_backend": requested_attention_backend,
                "sampler_type": requested_sampler_type,
                "dense_attention_backend": runtime_cfg["dense_backend"],
                "sparse_attention_backend": runtime_cfg["sparse_backend"],
            }
            state["_runtime_settings"] = {
                "attention_backend": requested_attention_backend,
                "dense_attention_backend": runtime_cfg["dense_backend"],
                "sparse_attention_backend": runtime_cfg["sparse_backend"],
                "sampler_type": requested_sampler_type,
            }
            state["preview_manifest_path"] = str(preview_manifest_path)
            state["_multiview"] = True
            state["_multiview_mode"] = mv_mode
            state["_multiview_count"] = int(n_views)
        except Exception:
            pass

        _log('Done. You can now click "Extract GLB".', 1.0)
        yield state, full_html, gr.update(value=_trim_status(status), visible=False)
        return

    # Single-view path keeps explicit stage-by-stage sampling for granular progress.
    images = [image]
    torch.manual_seed(seed)

    _log("Computing image embeddings (512px)…", 0.08)
    cond_512 = pipe.get_cond(images, 512)
    try:
        torch.save({k: v.detach().cpu() for k, v in cond_512.items()}, str(cond_512_path))
    except Exception:
        pass
    yield None, empty_html, status

    cond_1024 = None
    if pipeline_type != "512":
        _log("Computing image embeddings (1024px)…", 0.12)
        cond_1024 = pipe.get_cond(images, 1024)
        if cond_1024_path is not None:
            try:
                torch.save({k: v.detach().cpu() for k, v in cond_1024.items()}, str(cond_1024_path))
            except Exception:
                pass
        yield None, empty_html, status

    # Sparse structure resolution: direct 1024 uses 64; cascade and hybrid stay on 32.
    ss_res = 64 if pipeline_type == "1024" else 32
    _log("Stage 1/3: Sampling sparse structure…", 0.18)
    coords = pipe.sample_sparse_structure(cond_512, ss_res, 1, ss_params)
    try:
        torch.save(coords.detach().cpu(), str(coords_path))
    except Exception:
        pass
    yield None, empty_html, status

    if low_vram and coords.shape[0] > 24000 and pipeline_type in {"1024", "1024_cascade", "1536_cascade", "2048_cascade"}:
        if "shape_slat_flow_model_512" in pipe.models:
            _log(
                f"Low VRAM safeguard: token count {coords.shape[0]} is high, "
                "switching Generate to hybrid 512 geometry + 1024 texture mode.",
                0.24,
            )
            coords = (coords // 2).unique(dim=0)
            pipeline_type = "512g_1024t"
            yield None, empty_html, status
        else:
            _log(
                f"Low VRAM safeguard: token count {coords.shape[0]} is high, "
                "but the 512 shape model is not loaded, so Generate will continue on the original pipeline.",
                0.24,
            )
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
    elif pipeline_type == "512g_1024t":
        _log("Stage 2/3: Sampling shape latent (hybrid 512g + 1024t)…", 0.35)
        shape_slat = pipe.sample_shape_slat(cond_512, pipe.models["shape_slat_flow_model_512"], coords, shape_params)
        yield None, empty_html, status

        if not no_texture_gen:
            _log("Stage 3/3: Sampling texture latent (1024 hybrid)…", 0.55)
            tex_slat = pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, tex_params)
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
    elif "_cascade" in pipeline_type:
        # Any cascade resolution (768, 1024, 1280, 1536, 2048, custom)
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

    # Save latents for inspection / later subprocess extraction.
    try:
        np.savez_compressed(
            str(shape_slat_path),
            feats=shape_slat.feats.detach().cpu().numpy(),
            coords=shape_slat.coords.detach().cpu().numpy(),
        )
        _write_json(str(shape_res_path), {"res": int(res), "pipeline_type": pipeline_type})
        if tex_slat is not None and tex_slat_path is not None:
            np.savez_compressed(
                str(tex_slat_path),
                feats=tex_slat.feats.detach().cpu().numpy(),
                coords=tex_slat.coords.detach().cpu().numpy(),
            )
    except Exception:
        pass

    _log("Decoding latent to mesh…", 0.75)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res, use_tiled_extraction, use_chunked_processing)[0]
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

    # Save preview frames to disk (JPEG) + a manifest (used by subprocess mode too).
    try:
        preview_dir.mkdir(parents=True, exist_ok=True)
        manifest_files: Dict[str, List[str]] = {}
        for m_idx, mode in enumerate(MODES):
            key = mode["render_key"]
            manifest_files[key] = []
            for s_idx in range(STEPS):
                path = preview_dir / f"view-m{m_idx}-s{s_idx}.jpg"
                Image.fromarray(images[key][s_idx]).save(str(path), format="JPEG", quality=85)
                manifest_files[key].append(str(path))
        _write_json(
            str(preview_manifest_path),
            {
                "modes": [{"name": m["name"], "render_key": m["render_key"]} for m in MODES],
                "steps": STEPS,
                "files": manifest_files,
            },
        )
    except Exception:
        pass

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
                <p>- <b>Render Mode</b> - Click a circular button to switch render modes.</p>
                <p>- <b>View Angle</b> - Drag the slider to change view angle.</p>
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

    # Persist preview HTML and attach run metadata to the returned state so extraction can
    # save into the same outputs/<run_id>/ folder without overwriting.
    try:
        preview_html_path.write_text(full_html, encoding="utf-8")
    except Exception:
        pass
    try:
        state["_mode"] = "inproc"
        state["_run_id"] = run_id
        state["_run_dir"] = str(run_dir)
        state["_input_image_path"] = str(input_path)
        state["_preprocessed_image_path"] = str(preprocessed_path)
        state["_preprocessed_view_paths"] = _collect_preprocessed_view_paths(run_dir)
        state["_pipeline_type"] = pipeline_type
        state["_pipeline_strategy"] = pipeline_strategy_norm
        state["seed"] = int(seed)
        state["shape_slat_path"] = str(shape_slat_path)
        state["tex_slat_path"] = str(tex_slat_path) if tex_slat_path is not None else None
        state["_model_variant"] = requested_model_variant
        state["_gen_tex_params"] = {
            "steps": int(tex_slat_sampling_steps),
            "guidance_strength": float(tex_slat_guidance_strength),
            "guidance_rescale": float(tex_slat_guidance_rescale),
            "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
            "rescale_t": float(tex_slat_rescale_t),
            "attention_backend": requested_attention_backend,
            "sampler_type": requested_sampler_type,
            "dense_attention_backend": runtime_cfg["dense_backend"],
            "sparse_attention_backend": runtime_cfg["sparse_backend"],
        }
        state["_runtime_settings"] = {
            "model_variant": requested_model_variant,
            "model_repo": model_variant_cfg["model_repo"],
            "config_file": model_variant_cfg["config_file"],
            "attention_backend": requested_attention_backend,
            "dense_attention_backend": runtime_cfg["dense_backend"],
            "sparse_attention_backend": runtime_cfg["sparse_backend"],
            "sampler_type": requested_sampler_type,
        }
        state["_projection_settings"] = dict(projection_settings)
        state["preview_manifest_path"] = str(preview_manifest_path)
    except Exception:
        pass

    _log('Done. You can now click "Extract GLB".', 1.0)
    # Hide the overlay once preview is ready so users can see the render.
    yield state, full_html, gr.update(value=_trim_status(status), visible=False)


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    remesh_method: str,
    fill_holes_max_perimeter: float,
    repair_method: str,
    simplify_method: str,
    no_texture_gen: bool,
    deferred_texture_after_cleanup: bool,
    projection_texture_refine: bool,
    projection_view_azimuths: str,
    projection_view_elevations: str,
    projection_blend_exponent: float,
    projection_ortho_scale: float,
    projection_fill_holes: bool,
    projection_max_hole_size: int,
    prune_invisible_faces: bool,
    merge_vertices_dist: float,
    shade_smooth: bool,
    shade_smooth_angle: float,
    force_double_sided: bool,
    no_pbr_export: bool,
    export_formats: List[str],
    extract_use_chunked_processing: bool,
    extract_use_tiled_extraction: bool,
    ultrashape_enabled: bool,
    ultrashape_retexture_after_refine: bool,
    ultrashape_conservative_mode: bool,
    ultrashape_checkpoint: str,
    ultrashape_config_name: str,
    ultrashape_dtype: str,
    ultrashape_low_vram: bool,
    ultrashape_remove_bg: bool,
    ultrashape_steps: int,
    ultrashape_guidance_scale: float,
    ultrashape_octree_resolution: int,
    ultrashape_num_chunks: int,
    ultrashape_target_face_count: int,
    ultrashape_num_latents: int,
    ultrashape_box_v: float,
    ultrashape_mc_level: float,
    ultrashape_normalize_scale: float,
    ultrashape_num_sharp_points: int,
    ultrashape_num_uniform_points: int,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    if state is None:
        # This happens when users click "Extract GLB" before clicking "Generate"
        # (or right after changing the image / clicking an example).
        raise gr.Error("Nothing to extract yet. Click **Generate** first.")

    session = _session_key(req)

    # Always use staged subprocess extraction so each heavy stage runs in a
    # short-lived worker process with on-disk handoff artifacts.
    force_staged_subprocess_extract = True
    if (
        force_staged_subprocess_extract
        or subprocess_mode
        or (isinstance(state, dict) and state.get("_mode") == "subprocess")
        or bool(deferred_texture_after_cleanup)
        or (bool(ultrashape_enabled) and bool(ultrashape_retexture_after_refine))
    ):
        status = ""

        def _log(msg: str, p: Optional[float] = None) -> str:
            nonlocal status
            msg = _clean_status_text(msg)
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            status = (status + "\n" if status else "") + line
            print(line, flush=True)
            if p is not None:
                progress(p, desc=msg)
            return status

        run_dir = Path(state.get("_run_dir", Path(TMP_DIR) / str(req.session_hash)))
        logs_dir = ensure_dir(run_dir / "logs")
        work_dir = Path(TMP_DIR) / str(req.session_hash) / "subprocess" / str(state.get("_run_id", "extract"))
        work_dir.mkdir(parents=True, exist_ok=True)

        shape_slat_path = state.get("shape_slat_path")
        if not shape_slat_path:
            raise gr.Error("Missing shape latent on disk. Please Generate again with subprocess mode enabled.")
        tex_slat_path = state.get("tex_slat_path")
        res = int(state.get("res"))

        out_dir = run_dir / EXTRACT_FINAL_DIRNAME

        _log("Starting GLB extraction (subprocess)…", 0.0)
        # Show the overlay while extracting.
        yield None, None, gr.update(value=_trim_status(status), visible=True)

        export_formats = export_formats or ["glb"]
        if "glb" not in export_formats:
            export_formats = ["glb"] + list(export_formats)

        remesh_method, simplify_method, repair_method = _normalize_extract_methods(
            remesh_method=remesh_method,
            simplify_method=simplify_method,
            repair_method=repair_method,
            log_fn=_log,
        )
        requested_remesh_method = str(remesh_method)
        remesh_fallback_reason: Optional[str] = None
        if requested_remesh_method == "faithful_contouring" and not _is_faithful_contouring_available():
            remesh_method = "dual_contouring"
            remesh_fallback_reason = "missing_faithc_dependencies_precheck"
            _log(
                "WARNING: remesh_method='faithful_contouring' requires optional FaithC dependencies "
                "(`faithcontour` + `atom3d`) which are not installed. Falling back to 'dual_contouring'."
            )
            yield None, None, status
        _log(
            f"Remesh audit: requested={requested_remesh_method!r}, stage_input={str(remesh_method)!r}"
        )
        yield None, None, status
        stage3_requested_remesh_method = requested_remesh_method
        stage3_effective_remesh_method = str(remesh_method)
        stage3_remesh_fallback_reason = remesh_fallback_reason

        last_ui_update = 0.0
        log_path = Path(logs_dir) / "extract_glb.log"
        runtime_settings = (state.get("_runtime_settings") if isinstance(state, dict) else None) or {}
        selected_model_variant = normalize_model_variant(
            runtime_settings.get("model_variant") or (state.get("_model_variant") if isinstance(state, dict) else None)
        )
        model_variant_cfg = resolve_model_variant(selected_model_variant)
        model_repo = str(runtime_settings.get("model_repo") or model_variant_cfg["model_repo"])
        config_file = str(runtime_settings.get("config_file") or model_variant_cfg["config_file"])
        reference_image_path = state.get("_preprocessed_image_path") or state.get("_input_image_path")
        saved_view_paths = state.get("_preprocessed_view_paths") if isinstance(state, dict) else None
        projection_view_paths = [
            str(Path(p))
            for p in (saved_view_paths or _collect_preprocessed_view_paths(run_dir))
            if p and Path(p).is_file()
        ]
        if not projection_view_paths and reference_image_path and Path(str(reference_image_path)).is_file():
            projection_view_paths = [str(Path(str(reference_image_path)))]
        saved_projection = (state.get("_projection_settings") if isinstance(state, dict) else None) or {}
        projection_azimuth_text = str(projection_view_azimuths or "").strip()
        projection_elevation_text = str(projection_view_elevations or "").strip()
        if not projection_azimuth_text and saved_projection.get("azimuths"):
            projection_azimuth_text = ",".join(str(v) for v in saved_projection.get("azimuths", []))
        if not projection_elevation_text and saved_projection.get("elevations"):
            projection_elevation_text = ",".join(str(v) for v in saved_projection.get("elevations", []))
        projection_settings = {
            "enabled": bool(projection_texture_refine),
            "image_paths": list(projection_view_paths),
            "azimuths": [],
            "elevations": [],
            "blend_exponent": float(
                projection_blend_exponent
                if projection_blend_exponent is not None
                else saved_projection.get("blend_exponent", 2.0)
            ),
            "ortho_scale": float(
                projection_ortho_scale
                if projection_ortho_scale is not None
                else saved_projection.get("ortho_scale", 1.1)
            ),
            "fill_holes": bool(
                projection_fill_holes
                if projection_fill_holes is not None
                else saved_projection.get("fill_holes", True)
            ),
            "max_hole_size": int(
                projection_max_hole_size
                if projection_max_hole_size is not None
                else saved_projection.get("max_hole_size", 20)
            ),
        }
        if projection_settings["enabled"]:
            if not projection_settings["image_paths"]:
                projection_settings["enabled"] = False
                _log("Projection refinement requested but no saved preprocessed view images were found. Skipping projection.", 0.06)
                yield None, None, status
            else:
                try:
                    azimuths, elevations = resolve_projection_views(
                        [Path(p) for p in projection_settings["image_paths"]],
                        projection_azimuth_text,
                        projection_elevation_text,
                    )
                except Exception as e:
                    raise gr.Error(f"Invalid projection view angles: {e}")
                projection_settings["azimuths"] = [float(v) for v in azimuths]
                projection_settings["elevations"] = [float(v) for v in elevations]
        ultrashape_payload = {
            "enabled": bool(ultrashape_enabled),
            "retexture_after_refine": bool(ultrashape_retexture_after_refine),
            "conservative_mode": bool(ultrashape_conservative_mode),
            "retexture_params": (state.get("_gen_tex_params") if isinstance(state, dict) else None),
            "image_path": reference_image_path,
            "checkpoint": str(ultrashape_checkpoint or "").strip(),
            "config_name": str(ultrashape_config_name or "infer_dit_refine.yaml").strip(),
            "dtype": str(ultrashape_dtype or "bfloat16").strip(),
            "low_vram": bool(ultrashape_low_vram),
            "remove_bg": bool(ultrashape_remove_bg),
            "steps": int(ultrashape_steps),
            "guidance_scale": float(ultrashape_guidance_scale),
            "octree_resolution": int(ultrashape_octree_resolution),
            "num_chunks": int(ultrashape_num_chunks),
            "target_face_count": int(ultrashape_target_face_count),
            "num_latents": int(ultrashape_num_latents),
            "box_v": float(ultrashape_box_v),
            "mc_level": float(ultrashape_mc_level),
            "normalize_scale": float(ultrashape_normalize_scale),
            "num_sharp_points": int(ultrashape_num_sharp_points),
            "num_uniform_points": int(ultrashape_num_uniform_points),
            "seed": int(state.get("seed", 42) if isinstance(state, dict) else 42),
        }

        if projection_settings["enabled"] and bool(deferred_texture_after_cleanup):
            _log("Projection texture refinement is enabled, so Deferred Texture Rebuild will be skipped.", 0.06)
            yield None, None, status
        if projection_settings["enabled"] and bool(ultrashape_enabled) and bool(ultrashape_retexture_after_refine):
            _log("Projection texture refinement is enabled, so UltraShape re-texture will be skipped.", 0.06)
            yield None, None, status
        do_ultrashape_retexture = bool(
            ultrashape_enabled and ultrashape_retexture_after_refine and (not bool(no_texture_gen)) and (not projection_settings["enabled"])
        )
        do_deferred_retexture = bool(
            deferred_texture_after_cleanup and (not bool(no_texture_gen)) and (not projection_settings["enabled"])
        )
        do_retexture = bool(do_ultrashape_retexture or do_deferred_retexture)
        ref_path_obj = Path(str(reference_image_path)) if reference_image_path else None
        if do_retexture and (ref_path_obj is None or not ref_path_obj.is_file()):
            do_retexture = False
            _log("Deferred re-texture requested but no valid reference image found. Skipping re-texture.", 0.06)
            yield None, None, status

        if do_retexture:
            stage3_glb_prefix = "glb_stage3_shape_only"
        elif bool(projection_settings["enabled"]) and bool(no_texture_gen):
            stage3_glb_prefix = "glb_stage3_shape_only_projection"
        elif bool(projection_settings["enabled"]):
            stage3_glb_prefix = "glb_stage3_textured_projection_base"
        elif bool(no_texture_gen):
            stage3_glb_prefix = "glb_final_shape_only"
        else:
            stage3_glb_prefix = "glb_final_textured_direct"
        final_retexture_glb_prefix = "glb_final_textured_retexture"
        final_projection_glb_prefix = "glb_final_textured_projection"
        stage3_glb_path: Optional[str] = None
        stage3_has_textures = bool((not no_texture_gen) and (not do_retexture))
        final_has_textures = bool(stage3_has_textures)
        if bool(no_texture_gen):
            final_texture_source = "none_shape_only"
        elif bool(projection_settings["enabled"]):
            final_texture_source = "none_waiting_projection"
        elif bool(do_retexture):
            final_texture_source = "none_waiting_retexture"
        else:
            final_texture_source = "direct_attr_bake"

        mesh_blob_00 = out_dir / "00_decoded_mesh_voxel.pt"
        mesh_blob_01 = out_dir / "01_ultrashape_mesh_voxel.pt"
        current_mesh_blob = mesh_blob_00
        final_glb_path: Optional[str] = None

        def _run_stage(stage_name: str, stage_payload: dict, base_progress: float, ui_msg: str):
            nonlocal status, last_ui_update
            _log(ui_msg, base_progress)
            result_local = None
            for ev in _iter_subprocess_stage(stage_name, stage_payload, work_dir, log_path, session=session):
                if ev["type"] == "log":
                    line = ev["text"]
                    if line:
                        status = _trim_status(status + "\n" + line)
                    now = time.time()
                    if now - last_ui_update > 0.6:
                        last_ui_update = now
                        yield None, None, status
                else:
                    result_local = ev["result"]
            if not result_local:
                raise gr.Error(f"Stage '{stage_name}' returned no result.")
            return result_local

        try:
            # Stage 1: decode latent -> mesh blob
            decode_payload = {
                "model_repo": model_repo,
                "config_file": config_file,
                "shape_slat_path": str(shape_slat_path),
                "tex_slat_path": str(tex_slat_path) if tex_slat_path else None,
                "res": int(res),
                "no_texture_gen": bool(no_texture_gen),
                "extract_use_chunked_processing": bool(extract_use_chunked_processing),
                "extract_use_tiled_extraction": bool(extract_use_tiled_extraction),
                "mesh_blob_path": str(mesh_blob_00),
            }
            decode_result = yield from _run_stage(
                "extract_decode_mesh",
                decode_payload,
                0.08,
                "Stage 1/4: Decoding latent mesh (subprocess)…",
            )
            current_mesh_blob = Path(str(decode_result["mesh_blob_path"]))

            # Stage 2: optional UltraShape refine -> mesh blob
            if ultrashape_enabled:
                if ref_path_obj is None or (not ref_path_obj.is_file()):
                    _log("UltraShape enabled but no valid reference image found. Skipping refinement.", 0.20)
                    yield None, None, status
                else:
                    ultra_result = yield from _run_stage(
                        "extract_ultrashape_refine",
                        {
                            "mesh_blob_in": str(current_mesh_blob),
                            "mesh_blob_out": str(mesh_blob_01),
                            "image_path": str(ref_path_obj),
                            "ultrashape": ultrashape_payload,
                        },
                        0.22,
                        "Stage 2/4: Running UltraShape refinement (subprocess)…",
                    )
                    current_mesh_blob = Path(str(ultra_result["mesh_blob_path"]))

            # Stage 3: mesh blob -> GLB
            to_glb_export_formats = ["glb"] if (do_retexture or projection_settings["enabled"]) else list(export_formats)
            to_glb_result = yield from _run_stage(
                "extract_to_glb",
                {
                    "mesh_blob_path": str(current_mesh_blob),
                    "res": int(res),
                    "decimation_target": int(decimation_target),
                    "texture_size": int(texture_size),
                    "remesh_method": remesh_method,
                    "fill_holes_max_perimeter": float(fill_holes_max_perimeter),
                    "repair_method": repair_method,
                    "simplify_method": simplify_method,
                    "prune_invisible_faces": bool(prune_invisible_faces),
                    "merge_vertices_dist": float(merge_vertices_dist),
                    "shade_smooth": bool(shade_smooth),
                    "shade_smooth_angle": float(shade_smooth_angle),
                    "force_double_sided": bool(force_double_sided),
                    "no_pbr_export": bool(no_pbr_export),
                    "texture_extraction": bool((not no_texture_gen) and (not do_retexture)),
                    "out_dir": str(out_dir),
                    "prefix": stage3_glb_prefix,
                    "export_formats": to_glb_export_formats,
                },
                0.42,
                "Stage 3/4: Converting mesh to GLB (subprocess)…",
            )
            stage3_glb_path = str(to_glb_result["glb_path"])
            final_glb_path = stage3_glb_path
            stage3_requested_remesh_method = str(
                to_glb_result.get("requested_remesh_method", stage3_requested_remesh_method)
            )
            stage3_effective_remesh_method = str(
                to_glb_result.get("effective_remesh_method", stage3_effective_remesh_method)
            )
            stage3_remesh_fallback_reason = (
                to_glb_result.get("remesh_fallback_reason")
                or stage3_remesh_fallback_reason
            )
            if stage3_requested_remesh_method != stage3_effective_remesh_method:
                _log(
                    "Remesh audit result: "
                    f"requested={stage3_requested_remesh_method!r}, "
                    f"effective={stage3_effective_remesh_method!r}, "
                    f"fallback_reason={(stage3_remesh_fallback_reason or 'unknown')!r}"
                )
            else:
                _log(
                    "Remesh audit result: "
                    f"requested={stage3_requested_remesh_method!r}, "
                    f"effective={stage3_effective_remesh_method!r}"
                )
            yield None, None, status

            # Stage 4: optional retexture or projection refinement
            if do_retexture and ref_path_obj is not None and ref_path_obj.is_file():
                try:
                    if do_ultrashape_retexture:
                        _log(
                            "UltraShape re-texture enabled: extraction will run shape-only first, then regenerate textures.",
                            0.56,
                        )
                    elif do_deferred_retexture:
                        _log(
                            "Deferred re-texture enabled: extraction will run shape-only first, then regenerate textures.",
                            0.56,
                        )
                    yield None, None, status

                    tex_seed = int(state.get("seed", 42) if isinstance(state, dict) else 42)
                    tex_params = (state.get("_gen_tex_params") if isinstance(state, dict) else None) or {}
                    runtime_settings = (state.get("_runtime_settings") if isinstance(state, dict) else None) or {}
                    tex_attention_backend = str(
                        tex_params.get("attention_backend")
                        or runtime_settings.get("attention_backend")
                        or DEFAULT_ATTENTION_BACKEND
                    ).strip().lower()
                    tex_sampler_type = normalize_sampler_type(
                        tex_params.get("sampler_type")
                        or runtime_settings.get("sampler_type")
                        or DEFAULT_SAMPLER_TYPE
                    )
                    tex_res = int(res)
                    if tex_res != 512:
                        tex_res = min(tex_res, 1536)
                    retex_dir = run_dir / RETEXTURE_WORK_DIRNAME
                    cond_path = retex_dir / "03_cond.pt"
                    shape_slat_tex_path = retex_dir / "04_shape_slat.pt"
                    tex_slat_tex_path = retex_dir / "05_tex_slat.pt"
                    preprocessed_ref_path = retex_dir / "02_reference_preprocessed.png"

                    common_tex_payload = {
                        "model_repo": "microsoft/TRELLIS.2-4B",
                        "config_file": "texturing_pipeline.json",
                        "seed": tex_seed,
                        "resolution": int(tex_res),
                        "attention_backend": tex_attention_backend,
                        "sampler_type": tex_sampler_type,
                    }

                    _ = yield from _run_stage(
                        "tex_encode_cond",
                        {
                            **common_tex_payload,
                            "image_path": str(ref_path_obj),
                            "preprocessed_image_path": str(preprocessed_ref_path),
                            "cond_path": str(cond_path),
                        },
                        0.58,
                        "Stage 4/4: Re-texture pass (cond encoding)…",
                    )

                    _ = yield from _run_stage(
                        "tex_encode_shape",
                        {
                            **common_tex_payload,
                            "mesh_path": str(final_glb_path),
                            "shape_slat_path": str(shape_slat_tex_path),
                        },
                        0.66,
                        "Stage 4/4: Re-texture pass (shape encoding)…",
                    )

                    _ = yield from _run_stage(
                        "tex_sample_tex_slat",
                        {
                            **common_tex_payload,
                            "cond_path": str(cond_path),
                            "shape_slat_path": str(shape_slat_tex_path),
                            "tex_slat_path": str(tex_slat_tex_path),
                            "tex_params": {
                                "steps": int(tex_params.get("steps", 12)),
                                "guidance_strength": float(tex_params.get("guidance_strength", 1.0)),
                                "guidance_rescale": float(tex_params.get("guidance_rescale", 0.0)),
                                "guidance_interval": list(tex_params.get("guidance_interval", [0.6, 0.9])),
                                "rescale_t": float(tex_params.get("rescale_t", 3.0)),
                            },
                        },
                        0.74,
                        "Stage 4/4: Re-texture pass (texture sampling)…",
                    )

                    retex_result = yield from _run_stage(
                        "tex_decode_and_bake",
                        {
                            **common_tex_payload,
                            "mesh_path": str(final_glb_path),
                            "tex_slat_path": str(tex_slat_tex_path),
                            "texture_size": int(texture_size),
                            "out_dir": str(out_dir),
                            "prefix": final_retexture_glb_prefix,
                        },
                        0.82,
                        "Stage 4/4: Re-texture pass (decode + bake)…",
                    )
                    final_glb_path = str(retex_result["glb_path"])
                    final_has_textures = True
                    final_texture_source = "retexture_pass"
                except Exception as e:
                    _log(
                        f"Re-texture subprocess stages failed ({type(e).__name__}: {e}). "
                        "Keeping shape-only extracted GLB.",
                        0.86,
                    )
                    final_has_textures = False
                    final_texture_source = "none_retexture_failed"
                    try:
                        if final_glb_path:
                            stage3_path_obj = Path(final_glb_path)
                            _, fallback_final_p = next_indexed_path(
                                out_dir, prefix="glb_final_shape_only", ext="glb", digits=4, start=1
                            )
                            if stage3_path_obj.resolve() != fallback_final_p.resolve():
                                shutil.copyfile(str(stage3_path_obj), str(fallback_final_p))
                                final_glb_path = str(fallback_final_p)
                    except Exception:
                        pass
                    yield None, None, status
            elif projection_settings["enabled"]:
                try:
                    _log(
                        f"Projection texture refinement enabled: projecting {len(projection_settings['image_paths'])} preprocessed views onto the cleaned mesh.",
                        0.56,
                    )
                    yield None, None, status
                    projection_result = yield from _run_stage(
                        "project_texture_multiview",
                        {
                            "mesh_path": str(final_glb_path),
                            "image_paths": list(projection_settings["image_paths"]),
                            "azimuths": list(projection_settings["azimuths"]),
                            "elevations": list(projection_settings["elevations"]),
                            "texture_size": int(texture_size),
                            "blend_exponent": float(projection_settings["blend_exponent"]),
                            "ortho_scale": float(projection_settings["ortho_scale"]),
                            "fill_holes": bool(projection_settings["fill_holes"]),
                            "max_hole_size": int(projection_settings["max_hole_size"]),
                            "out_dir": str(out_dir),
                            "prefix": final_projection_glb_prefix,
                        },
                        0.76,
                        "Stage 4/4: Projection texture refinement…",
                    )
                    final_glb_path = str(projection_result["glb_path"])
                    final_has_textures = True
                    final_texture_source = "projection_multiview"
                except Exception as e:
                    _log(
                        f"Projection texture refinement failed ({type(e).__name__}: {e}). "
                        "Keeping the extracted GLB from Stage 3.",
                        0.86,
                    )
                    final_has_textures = bool(stage3_has_textures)
                    final_texture_source = (
                        "stage3_fallback_after_projection_failure"
                        if final_has_textures
                        else "none_projection_failed"
                    )
                    yield None, None, status

            needs_final_export_formats = bool(do_retexture or projection_settings["enabled"])
            extra_formats = [f for f in export_formats if str(f).lower().strip() != "glb"]
            if needs_final_export_formats and extra_formats:
                _ = yield from _run_stage(
                    "mesh_export_formats",
                    {
                        "mesh_path": str(final_glb_path),
                        "out_dir": str(out_dir),
                        "export_formats": extra_formats,
                    },
                    0.90,
                    "Exporting additional formats…",
                )
        except UserCancelled:
            _log("CANCELLED by user.", 0.0)
            yield None, None, status
            _clear_cancel_all(session)
            _clear_cancel_batch(session)
            return

        if not final_glb_path:
            raise gr.Error("Extraction failed (no GLB path returned). See logs in the run folder.")

        glb_path = str(final_glb_path)
        _write_extract_artifacts_manifest(
            out_dir=out_dir,
            stage3_glb_path=stage3_glb_path,
            stage3_has_textures=stage3_has_textures,
            final_glb_path=glb_path,
            final_has_textures=final_has_textures,
            final_texture_source=final_texture_source,
            export_formats=export_formats,
            retexture_requested=bool(do_retexture),
            requested_remesh_method=stage3_requested_remesh_method,
            effective_remesh_method=stage3_effective_remesh_method,
            remesh_fallback_reason=stage3_remesh_fallback_reason,
        )
        _log(f"Saved: {safe_relpath(glb_path, APP_DIR)}", 0.98)
        _log("Done.", 1.0)
        yield glb_path, glb_path, status
        return

    texture_extraction = not no_texture_gen

    run_dir = Path(state.get("_run_dir", os.path.join(TMP_DIR, str(req.session_hash))))
    out_dir = run_dir / EXTRACT_FINAL_DIRNAME
    try:
        shape_slat, tex_slat, res = unpack_state(state)
    except Exception:
        raise gr.Error("Invalid/empty generation state. Please click **Generate** again.")

    status = ""

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        msg = _clean_status_text(msg)
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        status = (status + "\n" if status else "") + line
        print(line, flush=True)
        if p is not None:
            progress(p, desc=msg)
        return status

    _log("Starting GLB extraction…", 0.0)
    # Show the overlay while extracting.
    yield None, None, gr.update(value=_trim_status(status), visible=True)

    _log("Loading TRELLIS.2 pipeline…", 0.05)
    pipe = get_image_pipeline(state.get("_model_variant") if isinstance(state, dict) else DEFAULT_MODEL_VARIANT)
    yield None, None, status
    
    _log("Decoding latent to mesh…", 0.15)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res, extract_use_tiled_extraction, extract_use_chunked_processing)[0]
    yield None, None, status

    if ultrashape_enabled:
        image_path = state.get("_preprocessed_image_path") or state.get("_input_image_path")
        image_file = Path(str(image_path)) if image_path else None
        if image_file is None or not image_file.is_file():
            _log("UltraShape enabled, but no valid reference image path was found. Skipping UltraShape.", 0.22)
            yield None, None, status
        else:
            _log("Running UltraShape mesh refinement...", 0.24)
            try:
                from ultrashape_integration import refine_mesh_with_ultrashape

                mesh = refine_mesh_with_ultrashape(
                    mesh,
                    image_path=str(image_file),
                    app_dir=str(APP_DIR),
                    models_dir=str(MODELS_DIR),
                    checkpoint=str(ultrashape_checkpoint or "").strip(),
                    config_name=str(ultrashape_config_name or "infer_dit_refine.yaml").strip(),
                    dtype=str(ultrashape_dtype or "bfloat16").strip(),
                    low_vram=bool(ultrashape_low_vram),
                    steps=int(ultrashape_steps),
                    guidance_scale=float(ultrashape_guidance_scale),
                    octree_resolution=int(ultrashape_octree_resolution),
                    num_chunks=int(ultrashape_num_chunks),
                    mc_level=float(ultrashape_mc_level),
                    box_v=float(ultrashape_box_v),
                    seed=int(state.get("seed", 42)),
                    remove_bg=bool(ultrashape_remove_bg),
                    normalize_scale=float(ultrashape_normalize_scale),
                    num_sharp_points=int(ultrashape_num_sharp_points),
                    num_uniform_points=int(ultrashape_num_uniform_points),
                    num_latents=int(ultrashape_num_latents),
                    target_face_count=int(ultrashape_target_face_count),
                    conservative_mode=bool(ultrashape_conservative_mode),
                    enable_pbar=True,
                )
                _log(
                    f"UltraShape refinement complete (vertices={len(mesh.vertices)}, faces={len(mesh.faces)}).",
                    0.28,
                )
            except Exception as e:
                _log(
                    f"UltraShape refinement failed ({type(e).__name__}: {e}). Continuing with base TRELLIS mesh.",
                    0.28,
                )
            yield None, None, status

    _log("Post-processing + baking GLB (this can take a while)…", 0.3)
    yield None, None, status

    ultrashape_retexture_requested = bool(
        ultrashape_enabled and ultrashape_retexture_after_refine and texture_extraction
    )
    retexture_image_file: Optional[Path] = None
    if ultrashape_retexture_requested:
        image_path = state.get("_preprocessed_image_path") or state.get("_input_image_path")
        retexture_image_file = Path(str(image_path)) if image_path else None
        if retexture_image_file is None or (not retexture_image_file.is_file()):
            ultrashape_retexture_requested = False
            _log("UltraShape re-texture requested, but no valid reference image was found. Using existing texture extraction.", 0.32)
            yield None, None, status
        else:
            _log("UltraShape re-texture enabled: extraction will run shape-only first, then regenerate textures.", 0.32)
            yield None, None, status

    if ultrashape_retexture_requested:
        stage3_glb_prefix = "glb_stage3_shape_only"
    elif bool(no_texture_gen):
        stage3_glb_prefix = "glb_final_shape_only"
    else:
        stage3_glb_prefix = "glb_final_textured_direct"
    final_retexture_glb_prefix = "glb_final_textured_retexture"
    stage3_glb_path: Optional[str] = None
    stage3_has_textures = bool((not no_texture_gen) and (not ultrashape_retexture_requested))
    final_has_textures = bool(stage3_has_textures)
    if bool(no_texture_gen):
        final_texture_source = "none_shape_only"
    elif bool(ultrashape_retexture_requested):
        final_texture_source = "none_waiting_retexture"
    else:
        final_texture_source = "direct_attr_bake"

    remesh_method, simplify_method, repair_method = _normalize_extract_methods(
        remesh_method=remesh_method,
        simplify_method=simplify_method,
        repair_method=repair_method,
        log_fn=_log,
    )
    requested_remesh_method = str(remesh_method)
    remesh_fallback_reason: Optional[str] = None
    if requested_remesh_method == "faithful_contouring" and not _is_faithful_contouring_available():
        remesh_method = "dual_contouring"
        remesh_fallback_reason = "missing_faithc_dependencies_precheck"
        _log(
            "WARNING: remesh_method='faithful_contouring' requires optional FaithC dependencies "
            "(`faithcontour` + `atom3d`) which are not installed. Falling back to 'dual_contouring'."
        )
        yield None, None, status

    to_glb_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipe.pbr_attr_layout,
        "grid_size": res,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": decimation_target,
        "fill_holes_max_perimeter": float(fill_holes_max_perimeter),
        "repair_method": repair_method,
        "simplify_method": simplify_method,
        "texture_extraction": (False if ultrashape_retexture_requested else texture_extraction),
        "texture_size": texture_size,
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0,
        "remesh_method": remesh_method,
        "prune_invisible": prune_invisible_faces,
        "merge_vertices_dist": float(merge_vertices_dist),
        "shade_smooth": bool(shade_smooth),
        "shade_smooth_angle": float(shade_smooth_angle),
        "force_double_sided": bool(force_double_sided),
        "no_pbr": bool(no_pbr_export),
        "use_tqdm": True,
    }
    _log(
        f"Remesh audit: requested={requested_remesh_method!r}, stage_input={str(to_glb_kwargs['remesh_method'])!r}"
    )
    yield None, None, status
    try:
        glb = o_voxel.postprocess.to_glb(**to_glb_kwargs)
    except Exception as e:
        # Failsafe: if FaithC is missing or faithful contouring OOMs, retry with
        # a safer remesher instead of crashing extraction.
        can_fallback = requested_remesh_method == "faithful_contouring"
        is_missing_faithc = isinstance(e, ImportError) and ("Faithful Contouring is not installed" in str(e))
        is_oom = isinstance(e, torch.OutOfMemoryError) or (
            "out of memory" in str(e).lower() and "cuda" in str(e).lower()
        )
        if can_fallback and (is_missing_faithc or is_oom):
            fallback_method = "dual_contouring"
            if is_missing_faithc:
                _log(f"WARNING: {e} Falling back to remesh_method={fallback_method!r}.")
                remesh_fallback_reason = "faithc_missing_during_to_glb"
            else:
                _log(
                    "WARNING: faithful_contouring ran out of GPU memory. "
                    f"Retrying with remesh_method={fallback_method!r}."
                )
                remesh_fallback_reason = "oom_in_faithful_contouring"
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            to_glb_kwargs["remesh_method"] = fallback_method
            glb = o_voxel.postprocess.to_glb(**to_glb_kwargs)
        else:
            raise
    effective_remesh_method = str(to_glb_kwargs["remesh_method"])
    if requested_remesh_method != effective_remesh_method:
        _log(
            "Remesh audit result: "
            f"requested={requested_remesh_method!r}, "
            f"effective={effective_remesh_method!r}, "
            f"fallback_reason={(remesh_fallback_reason or 'unknown')!r}"
        )
    else:
        _log(
            "Remesh audit result: "
            f"requested={requested_remesh_method!r}, "
            f"effective={effective_remesh_method!r}"
        )
    yield None, None, status

    if ultrashape_retexture_requested:
        _log("Saving stage-3 shape-only GLB…", 0.68)
        _, stage3_glb_path_p = next_indexed_path(out_dir, prefix=stage3_glb_prefix, ext="glb", digits=4, start=1)
        glb.export(str(stage3_glb_path_p), extension_webp=False)
        stage3_glb_path = str(stage3_glb_path_p)
        yield None, None, status

    if ultrashape_retexture_requested and retexture_image_file is not None:
        try:
            _log("Re-texturing UltraShape mesh with TRELLIS texturing pipeline…", 0.72)
            yield None, None, status

            try:
                pipe.cpu()
            except Exception:
                pass
            torch.cuda.empty_cache()

            mesh_for_tex = glb
            if isinstance(mesh_for_tex, trimesh.Scene):
                mesh_for_tex = mesh_for_tex.to_mesh()

            tex_pipe = get_texturing_pipeline()
            runtime_settings = (state.get("_runtime_settings") if isinstance(state, dict) else None) or {}
            tex_attention_backend = str(
                (state.get("_gen_tex_params") or {}).get("attention_backend") if isinstance(state, dict) else ""
            ).strip().lower() or str(runtime_settings.get("attention_backend") or DEFAULT_ATTENTION_BACKEND).strip().lower()
            tex_sampler_type = normalize_sampler_type(
                ((state.get("_gen_tex_params") or {}).get("sampler_type") if isinstance(state, dict) else None)
                or runtime_settings.get("sampler_type")
                or DEFAULT_SAMPLER_TYPE
            )
            _configure_texturing_pipeline_runtime(
                tex_pipe,
                attention_backend=tex_attention_backend,
                sampler_type=tex_sampler_type,
            )
            tex_pipe.low_vram = bool(ultrashape_low_vram)

            tex_res = int(res)
            if tex_res != 512:
                tex_res = min(tex_res, 1536)
            tex_params_src = state.get("_gen_tex_params") if isinstance(state, dict) else None
            tex_params = {
                "steps": int((tex_params_src or {}).get("steps", 12)),
                "guidance_strength": float((tex_params_src or {}).get("guidance_strength", 1.0)),
                "guidance_rescale": float((tex_params_src or {}).get("guidance_rescale", 0.0)),
                "guidance_interval": list((tex_params_src or {}).get("guidance_interval", [0.6, 0.9])),
                "rescale_t": float((tex_params_src or {}).get("rescale_t", 3.0)),
            }
            if len(tex_params["guidance_interval"]) != 2:
                tex_params["guidance_interval"] = [0.6, 0.9]

            with Image.open(str(retexture_image_file)) as _im:
                tex_ref = _im.convert("RGBA")
            tex_ref = tex_pipe.preprocess_image(tex_ref)
            mesh_for_tex = tex_pipe.preprocess_mesh(mesh_for_tex)

            torch.manual_seed(int(state.get("seed", 42) if isinstance(state, dict) else 42))
            with torch.inference_mode():
                cond = tex_pipe.get_cond([tex_ref], 512 if tex_res == 512 else 1024)
                shape_slat_tex = tex_pipe.encode_shape_slat(mesh_for_tex, tex_res)
                tex_model = tex_pipe.models["tex_slat_flow_model_512"] if tex_res == 512 else tex_pipe.models["tex_slat_flow_model_1024"]
                tex_slat = tex_pipe.sample_tex_slat(cond, tex_model, shape_slat_tex, tex_params)
                pbr_voxel = tex_pipe.decode_tex_slat(tex_slat)
                glb = tex_pipe.postprocess_mesh(mesh_for_tex, pbr_voxel, tex_res, int(texture_size))
            final_has_textures = True
            final_texture_source = "retexture_pass"

            _log("UltraShape re-texture complete.", 0.84)
            yield None, None, status
        except Exception as e:
            _log(
                f"UltraShape re-texture failed ({type(e).__name__}: {e}). Falling back to standard texture extraction.",
                0.84,
            )
            try:
                fallback_kwargs = dict(to_glb_kwargs)
                fallback_kwargs["texture_extraction"] = True
                glb = o_voxel.postprocess.to_glb(**fallback_kwargs)
                final_has_textures = True
                final_texture_source = "fallback_attr_bake_after_retexture_error"
                _log("Fallback texture extraction complete.", 0.86)
            except Exception as e2:
                final_has_textures = False
                final_texture_source = "none_retexture_failed"
                _log(
                    f"Fallback texture extraction also failed ({type(e2).__name__}: {e2}). Keeping shape-only GLB.",
                    0.86,
                )
            yield None, None, status

    _log("Saving GLB…", 0.9)
    export_formats = export_formats or ["glb"]
    if "glb" not in export_formats:
        export_formats = ["glb"] + list(export_formats)

    if ultrashape_retexture_requested:
        final_glb_prefix = final_retexture_glb_prefix if final_has_textures else "glb_final_shape_only"
    else:
        final_glb_prefix = stage3_glb_prefix

    use_shape_only_fallback = bool(ultrashape_retexture_requested and (not final_has_textures) and stage3_glb_path)
    if use_shape_only_fallback:
        _, fallback_final_p = next_indexed_path(out_dir, prefix="glb_final_shape_only", ext="glb", digits=4, start=1)
        try:
            shutil.copyfile(str(stage3_glb_path), str(fallback_final_p))
        except Exception:
            glb.export(str(fallback_final_p), extension_webp=False)
        glb_path = str(fallback_final_p)
        final_glb_prefix = "glb_final_shape_only"
        _, parsed_idx = _split_indexed_stem(Path(glb_path).stem)
        idx = parsed_idx if parsed_idx is not None else 1
    else:
        idx, glb_path_p = next_indexed_path(out_dir, prefix=final_glb_prefix, ext="glb", digits=4, start=1)
        glb.export(str(glb_path_p), extension_webp=False)
        glb_path = str(glb_path_p)
        if stage3_glb_path is None:
            stage3_glb_path = glb_path

    # Optional extra exports (best effort; never fail the main GLB export).
    extras = [f for f in export_formats if f != "glb"]
    for fmt in extras:
        try:
            fmt = str(fmt).lower().strip()
            out_path = _export_path_for_format(out_dir, fmt, final_glb_prefix, idx)
            glb.export(str(out_path))
        except Exception as e:
            _log(f"Extra export '{fmt}' failed: {type(e).__name__}: {e}", 0.95)

    _write_extract_artifacts_manifest(
        out_dir=out_dir,
        stage3_glb_path=stage3_glb_path,
        stage3_has_textures=stage3_has_textures,
        final_glb_path=glb_path,
        final_has_textures=final_has_textures,
        final_texture_source=final_texture_source,
        export_formats=export_formats,
        retexture_requested=bool(ultrashape_retexture_requested),
        requested_remesh_method=requested_remesh_method,
        effective_remesh_method=effective_remesh_method,
        remesh_fallback_reason=remesh_fallback_reason,
    )
    torch.cuda.empty_cache()
    _log(f"Saved: {safe_relpath(glb_path, APP_DIR)}", 0.98)
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
    tex_slat_guidance_interval_start: float,
    tex_slat_guidance_interval_end: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    low_vram: bool,
    attention_backend: str,
    sampler_type: str,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    status = ""
    session = _session_key(req)

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        msg = _clean_status_text(msg)
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
    requested_attention_backend = str(attention_backend or DEFAULT_ATTENTION_BACKEND).strip().lower()
    requested_sampler_type = normalize_sampler_type(sampler_type)
    runtime_cfg = apply_runtime_backends(requested_attention_backend)
    runtime_cfg["sampler_type"] = requested_sampler_type

    if subprocess_mode:
        # Unload any global pipelines to free VRAM for subprocess
        unload_global_pipelines()

        run = allocate_run_dir(OUTPUTS_DIR, digits=4)
        run_dir = run.run_dir
        run_id = run.run_id
        work_dir = Path(TMP_DIR) / str(req.session_hash) / "subprocess" / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Persist inputs
        src_mesh = Path(mesh_file)
        mesh_copy = run_dir / f"00_mesh{src_mesh.suffix.lower() or '.ply'}"
        try:
            shutil.copyfile(str(src_mesh), str(mesh_copy))
        except Exception:
            mesh_copy = src_mesh
        img_path = run_dir / "01_reference.png"
        try:
            image.save(str(img_path))
        except Exception:
            pass

        _write_json(
            str(run_dir / "run.json"),
            {
                "run_id": run_id,
                "type": "texturing",
                "subprocess_mode": True,
                "seed": int(seed),
                "resolution": int(resolution),
                "texture_size": int(texture_size),
                "attention_backend": requested_attention_backend,
                "dense_attention_backend": runtime_cfg["dense_backend"],
                "sparse_attention_backend": runtime_cfg["sparse_backend"],
                "sampler_type": requested_sampler_type,
                "tex_params": {
                    "steps": int(tex_slat_sampling_steps),
                    "guidance_strength": float(tex_slat_guidance_strength),
                    "guidance_rescale": float(tex_slat_guidance_rescale),
                    "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                    "rescale_t": float(tex_slat_rescale_t),
                    "attention_backend": requested_attention_backend,
                    "sampler_type": requested_sampler_type,
                },
            },
        )

        _log(f"Subprocess mode ON. Run: {run_id} → {safe_relpath(run_dir, APP_DIR)}", 0.02)
        _log(
            "Runtime settings: "
            f"requested_backend={requested_attention_backend}, "
            f"dense_backend={runtime_cfg['dense_backend']}, "
            f"sparse_backend={runtime_cfg['sparse_backend']}, "
            f"sampler={requested_sampler_type}",
            0.02,
        )
        yield None, None, status

        # Define intermediate paths
        preprocessed_image_path = run_dir / "02_reference_preprocessed.png"
        cond_path = run_dir / "03_cond.npz"
        shape_slat_path = run_dir / "04_shape_slat.pt"  # Use .pt to preserve spatial cache
        tex_slat_path = run_dir / "05_tex_slat.pt"  # Use .pt to preserve spatial cache
        out_dir = run_dir / "08_texturing"

        tex_params = {
            "steps": int(tex_slat_sampling_steps),
            "guidance_strength": float(tex_slat_guidance_strength),
            "guidance_rescale": float(tex_slat_guidance_rescale),
            "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
            "rescale_t": float(tex_slat_rescale_t),
        }

        common_payload = {
            "model_repo": "microsoft/TRELLIS.2-4B",
            "config_file": "texturing_pipeline.json",
            "seed": int(seed),
            "resolution": int(resolution),
            "attention_backend": requested_attention_backend,
            "sampler_type": requested_sampler_type,
        }

        last_ui_update = 0.0
        
        # Single unified log file for all texturing stages
        unified_log_path = run_dir / "texture_tab_run_logs.txt"

        try:
            # Stage 1: Encode conditioning (image embeddings)
            _log("Computing image embeddings...", 0.1)
            yield None, None, status
            payload1 = {
                **common_payload,
                "image_path": str(img_path),
                "preprocessed_image_path": str(preprocessed_image_path),
                "cond_path": str(cond_path),
            }
            for ev in _iter_subprocess_stage("tex_encode_cond", payload1, work_dir, unified_log_path, session=session):
                if ev["type"] == "log":
                    line = ev["text"]
                    if line:
                        status = status + "\n" + line
                        status = _trim_status(status)
                    now = time.time()
                    if now - last_ui_update > 0.6:
                        last_ui_update = now
                        yield None, None, status
                else:
                    result1 = ev["result"]

            # Stage 2: Encode shape
            _log("Encoding mesh to shape latent...", 0.3)
            yield None, None, status
            payload2 = {
                **common_payload,
                "mesh_path": str(mesh_copy),
                "shape_slat_path": str(shape_slat_path),
            }
            for ev in _iter_subprocess_stage("tex_encode_shape", payload2, work_dir, unified_log_path, session=session):
                if ev["type"] == "log":
                    line = ev["text"]
                    if line:
                        status = status + "\n" + line
                        status = _trim_status(status)
                    now = time.time()
                    if now - last_ui_update > 0.6:
                        last_ui_update = now
                        yield None, None, status
                else:
                    result2 = ev["result"]

            # Stage 3: Sample texture latent
            _log("Sampling texture latent...", 0.5)
            yield None, None, status
            payload3 = {
                **common_payload,
                "cond_path": str(cond_path),
                "shape_slat_path": str(shape_slat_path),
                "tex_slat_path": str(tex_slat_path),
                "tex_params": tex_params,
            }
            for ev in _iter_subprocess_stage("tex_sample_tex_slat", payload3, work_dir, unified_log_path, session=session):
                if ev["type"] == "log":
                    line = ev["text"]
                    if line:
                        status = status + "\n" + line
                        status = _trim_status(status)
                    now = time.time()
                    if now - last_ui_update > 0.6:
                        last_ui_update = now
                        yield None, None, status
                else:
                    result3 = ev["result"]

            # Stage 4: Decode and bake
            _log("Decoding and baking textures...", 0.7)
            yield None, None, status
            payload4 = {
                **common_payload,
                "mesh_path": str(mesh_copy),
                "tex_slat_path": str(tex_slat_path),
                "texture_size": int(texture_size),
                "out_dir": str(out_dir),
                "prefix": "textured",
            }
            result = None
            for ev in _iter_subprocess_stage("tex_decode_and_bake", payload4, work_dir, unified_log_path, session=session):
                if ev["type"] == "log":
                    line = ev["text"]
                    if line:
                        status = status + "\n" + line
                        status = _trim_status(status)
                    now = time.time()
                    if now - last_ui_update > 0.6:
                        last_ui_update = now
                        yield None, None, status
                else:
                    result = ev["result"]

        except UserCancelled:
            _log("CANCELLED by user.", 0.0)
            yield None, None, status
            _clear_cancel_all(session)
            _clear_cancel_batch(session)
            return

        if not result or "glb_path" not in result:
            raise gr.Error("Texturing failed (no GLB path returned). See logs in the run folder.")

        glb_path = result["glb_path"]
        _log("Done.", 1.0)
        yield glb_path, glb_path, status
        return

    # In-process mode still writes all artifacts into a new outputs/<run_id>/ folder.
    run = allocate_run_dir(OUTPUTS_DIR, digits=4)
    run_dir = run.run_dir
    run_id = run.run_id

    # Persist inputs
    src_mesh = Path(mesh_file)
    mesh_copy = run_dir / f"00_mesh{src_mesh.suffix.lower() or '.ply'}"
    try:
        shutil.copyfile(str(src_mesh), str(mesh_copy))
    except Exception:
        mesh_copy = src_mesh
    raw_img_path = run_dir / "01_reference_raw.png"
    try:
        image.save(str(raw_img_path))
    except Exception:
        pass

    _write_json(
        str(run_dir / "run.json"),
        {
            "run_id": run_id,
            "type": "texturing",
            "subprocess_mode": False,
            "seed": int(seed),
            "resolution": int(resolution),
            "texture_size": int(texture_size),
            "attention_backend": requested_attention_backend,
            "dense_attention_backend": runtime_cfg["dense_backend"],
            "sparse_attention_backend": runtime_cfg["sparse_backend"],
            "sampler_type": requested_sampler_type,
            "tex_params": {
                "steps": int(tex_slat_sampling_steps),
                "guidance_strength": float(tex_slat_guidance_strength),
                "guidance_rescale": float(tex_slat_guidance_rescale),
                "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                "rescale_t": float(tex_slat_rescale_t),
                "attention_backend": requested_attention_backend,
                "sampler_type": requested_sampler_type,
            },
        },
    )

    _log(f"Run: {run_id} → {safe_relpath(run_dir, APP_DIR)}", 0.0)
    _log(
        "Runtime settings: "
        f"requested_backend={requested_attention_backend}, "
        f"dense_backend={runtime_cfg['dense_backend']}, "
        f"sparse_backend={runtime_cfg['sparse_backend']}, "
        f"sampler={requested_sampler_type}",
        0.0,
    )
    yield None, None, status

    _log("Loading texturing pipeline (first run can take a while)…", 0.05)
    pipe = get_texturing_pipeline()
    runtime_cfg = _configure_texturing_pipeline_runtime(
        pipe,
        attention_backend=requested_attention_backend,
        sampler_type=requested_sampler_type,
    )
    pipe.low_vram = low_vram  # Respect user's low VRAM setting
    yield None, None, status

    _log("Loading mesh…", 0.1)

    mesh = trimesh.load(str(mesh_copy))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    yield None, None, status

    res_int = int(resolution)
    tex_params = {
        "steps": tex_slat_sampling_steps,
        "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale,
        "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
        "rescale_t": tex_slat_rescale_t,
    }

    _log("Preprocessing reference image…", 0.18)
    image = pipe.preprocess_image(image)
    try:
        image.save(str(run_dir / "02_reference_preprocessed.png"))
    except Exception:
        pass
    yield None, None, status

    _log("Preprocessing mesh…", 0.22)
    mesh = pipe.preprocess_mesh(mesh)
    yield None, None, status

    _log(f"Computing image embeddings ({512 if res_int == 512 else 1024}px)…", 0.3)
    torch.manual_seed(seed)
    with torch.inference_mode():
        cond = pipe.get_cond([image], 512) if res_int == 512 else pipe.get_cond([image], 1024)
    yield None, None, status

    _log("Encoding mesh to shape latent…", 0.4)
    with torch.inference_mode():
        shape_slat = pipe.encode_shape_slat(mesh, res_int)
    yield None, None, status

    tex_model = pipe.models["tex_slat_flow_model_512"] if res_int == 512 else pipe.models["tex_slat_flow_model_1024"]
    _log("Sampling texture latent…", 0.55)
    with torch.inference_mode():
        tex_slat = pipe.sample_tex_slat(cond, tex_model, shape_slat, tex_params)
    yield None, None, status

    _log("Decoding texture latent…", 0.72)
    with torch.inference_mode():
        pbr_voxel = pipe.decode_tex_slat(tex_slat)
    yield None, None, status

    _log("Baking textures onto mesh…", 0.84)
    with torch.inference_mode():
        output = pipe.postprocess_mesh(mesh, pbr_voxel, res_int, texture_size)
    yield None, None, status

    _log("Saving textured GLB…", 0.9)
    out_dir = run_dir / "08_texturing"
    _, glb_path_p = next_indexed_path(out_dir, prefix="textured", ext="glb", digits=4, start=1)
    output.export(str(glb_path_p), extension_webp=False)
    glb_path = str(glb_path_p)
    torch.cuda.empty_cache()
    _log("Done.", 1.0)
    yield glb_path, glb_path, status


def _coerce_file_to_path(f: Any) -> Optional[str]:
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("name") or f.get("path")
    return getattr(f, "name", None) or str(f)


def ultrashape_refine_mesh(
    mesh_file: Any,
    image: Image.Image,
    seed: int,
    output_format: str,
    export_formats: List[str],
    checkpoint: str,
    config_name: str,
    dtype: str,
    low_vram: bool,
    remove_bg: bool,
    steps: int,
    guidance_scale: float,
    octree_resolution: int,
    num_chunks: int,
    target_face_count: int,
    num_latents: int,
    box_v: float,
    mc_level: float,
    normalize_scale: float,
    num_sharp_points: int,
    num_uniform_points: int,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    status = ""
    session = _session_key(req)

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        msg = _clean_status_text(msg)
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        status = (status + "\n" if status else "") + line
        print(line, flush=True)
        if p is not None:
            progress(p, desc=msg)
        return status

    mesh_path = _coerce_file_to_path(mesh_file)
    if not mesh_path:
        raise gr.Error("Please upload a mesh file.")
    if image is None:
        raise gr.Error("Please provide a reference image.")

    src_mesh = Path(mesh_path)
    if not src_mesh.is_file():
        raise gr.Error(f"Mesh file not found: {mesh_path}")

    out_fmt = str(output_format or "glb").lower().strip()
    if out_fmt not in {"glb", "obj", "ply", "stl"}:
        out_fmt = "glb"
    ex_fmts = export_formats or [out_fmt]
    ex_fmts = [str(x).lower().strip() for x in ex_fmts if str(x).lower().strip() in {"glb", "obj", "ply", "stl"}]
    if out_fmt not in ex_fmts:
        ex_fmts = [out_fmt] + ex_fmts

    run = allocate_run_dir(OUTPUTS_DIR, digits=4)
    run_dir = run.run_dir
    run_id = run.run_id
    out_dir = run_dir / "08_ultrashape_refine"

    mesh_copy = run_dir / f"00_mesh{src_mesh.suffix.lower() or '.ply'}"
    try:
        shutil.copyfile(str(src_mesh), str(mesh_copy))
    except Exception:
        mesh_copy = src_mesh

    img_path = run_dir / "01_reference.png"
    try:
        image.save(str(img_path))
    except Exception:
        pass

    _write_json(
        str(run_dir / "run.json"),
        {
            "run_id": run_id,
            "type": "ultrashape_refine",
            "subprocess_mode": bool(subprocess_mode),
            "seed": int(seed),
            "output_format": out_fmt,
            "export_formats": list(ex_fmts),
            "checkpoint": str(checkpoint or "").strip(),
            "config_name": str(config_name or "infer_dit_refine.yaml").strip(),
            "dtype": str(dtype or "bfloat16").strip(),
            "low_vram": bool(low_vram),
            "remove_bg": bool(remove_bg),
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "octree_resolution": int(octree_resolution),
            "num_chunks": int(num_chunks),
            "target_face_count": int(target_face_count),
            "num_latents": int(num_latents),
            "box_v": float(box_v),
            "mc_level": float(mc_level),
            "normalize_scale": float(normalize_scale),
            "num_sharp_points": int(num_sharp_points),
            "num_uniform_points": int(num_uniform_points),
        },
    )

    _log(f"Run: {run_id} → {safe_relpath(run_dir, APP_DIR)}", 0.0)
    yield None, None, status

    if subprocess_mode:
        unload_global_pipelines()
        work_dir = Path(TMP_DIR) / str(req.session_hash) / "subprocess" / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "mesh_path": str(mesh_copy),
            "image_path": str(img_path),
            "out_dir": str(out_dir),
            "prefix": "ultrashape_refined",
            "output_format": out_fmt,
            "export_formats": list(ex_fmts),
            "checkpoint": str(checkpoint or "").strip(),
            "config_name": str(config_name or "infer_dit_refine.yaml").strip(),
            "dtype": str(dtype or "bfloat16").strip(),
            "low_vram": bool(low_vram),
            "remove_bg": bool(remove_bg),
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "octree_resolution": int(octree_resolution),
            "num_chunks": int(num_chunks),
            "target_face_count": int(target_face_count),
            "num_latents": int(num_latents),
            "box_v": float(box_v),
            "mc_level": float(mc_level),
            "normalize_scale": float(normalize_scale),
            "num_sharp_points": int(num_sharp_points),
            "num_uniform_points": int(num_uniform_points),
            "seed": int(seed),
        }
        log_path = run_dir / "ultrashape_refine_logs.txt"
        last_ui_update = 0.0
        result = None

        _log("Running UltraShape refinement (subprocess)…", 0.05)
        yield None, None, status

        try:
            for ev in _iter_subprocess_stage("ultrashape_refine_mesh", payload, work_dir, log_path, session=session):
                if ev["type"] == "log":
                    line = ev["text"]
                    if line:
                        status = _trim_status(status + "\n" + line)
                    now = time.time()
                    if now - last_ui_update > 0.6:
                        last_ui_update = now
                        yield None, None, status
                else:
                    result = ev["result"]
        except UserCancelled:
            _log("CANCELLED by user.", 0.0)
            yield None, None, status
            _clear_cancel_all(session)
            _clear_cancel_batch(session)
            return

        if not result or "mesh_path" not in result:
            raise gr.Error("UltraShape refinement failed (no output mesh path).")
        mesh_out = str(result["mesh_path"])
        preview_out = str(result.get("preview_path") or mesh_out)
        _log(f"Saved: {safe_relpath(mesh_out, APP_DIR)}", 0.98)
        _log("Done.", 1.0)
        yield preview_out, mesh_out, status
        return

    _log("Loading mesh…", 0.08)
    mesh = trimesh.load(str(mesh_copy))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    yield None, None, status

    _log("Running UltraShape refinement…", 0.2)
    from ultrashape_integration import refine_mesh_with_ultrashape

    refined = refine_mesh_with_ultrashape(
        mesh,
        image_path=str(img_path),
        app_dir=str(APP_DIR),
        models_dir=str(MODELS_DIR),
        checkpoint=str(checkpoint or "").strip(),
        config_name=str(config_name or "infer_dit_refine.yaml").strip(),
        dtype=str(dtype or "bfloat16").strip(),
        low_vram=bool(low_vram),
        remove_bg=bool(remove_bg),
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        octree_resolution=int(octree_resolution),
        num_chunks=int(num_chunks),
        target_face_count=int(target_face_count),
        num_latents=int(num_latents),
        box_v=float(box_v),
        mc_level=float(mc_level),
        normalize_scale=float(normalize_scale),
        num_sharp_points=int(num_sharp_points),
        num_uniform_points=int(num_uniform_points),
        seed=int(seed),
        enable_pbar=True,
    )
    yield None, None, status

    _log("Saving refined mesh…", 0.9)
    _, out_path = next_indexed_path(out_dir, prefix="ultrashape_refined", ext=out_fmt, digits=4, start=1)
    refined.export(str(out_path))
    preview_path = out_path
    if out_fmt != "glb":
        try:
            preview_path = out_dir / f"{out_path.stem}_preview.glb"
            refined.export(str(preview_path))
        except Exception:
            preview_path = out_path
    for fmt in ex_fmts:
        if fmt == out_fmt:
            continue
        try:
            p = out_dir / f"{fmt}_{out_path.stem.split('_')[-1]}.{fmt}"
            refined.export(str(p))
        except Exception:
            pass
    torch.cuda.empty_cache()
    _log(f"Saved: {safe_relpath(str(out_path), APP_DIR)}", 0.98)
    _log("Done.", 1.0)
    yield str(preview_path), str(out_path), status


# --------------------------------- App UI -----------------------------------

with gr.Blocks(
    title="TRELLIS.2 Premium",
    delete_cache=(600, 600),
) as demo:
    gr.Markdown(
        """
<div style="display:flex; flex-wrap:wrap; align-items:baseline; gap:10px 16px;">
  <div style="font-size:1.35rem; font-weight:700;">
    Trellis 2 Ultimate SECourses Premium App V6.1 :
    <a href="https://www.patreon.com/posts/147686623" target="_blank">https://www.patreon.com/posts/147686623</a>
  </div>
  <div style="color:var(--body-text-color-subdued);">
    Generate a 3D asset from an image, export as GLB, and optionally texture an existing mesh.
  </div>
</div>
"""
    )

    with gr.Row():
        subprocess_mode = gr.Checkbox(
            label="Subprocess stage processing (zero leftover VRAM between stages)",
            value=True,
        )

    demo.load(start_session)
    demo.unload(end_session)

    with gr.Tabs() as main_tabs:
        # ---------------------------- Tab 1: Image -> 3D ----------------------------
        with gr.Tab("Image → 3D"):
            with gr.Row():
                with gr.Column(scale=1, min_width=380, elem_id="image_input_panel"):
                    input_images = gr.Files(
                        label="Upload Input Image (Or Multi-Angle)",
                        file_types=["image"],
                        file_count="multiple",
                        type="filepath",
                        allow_reordering=True,
                    )
                    input_preview_single = gr.Image(
                        label="Input Preview",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        height=400,
                    )
                    input_preview_gallery = gr.Gallery(
                        label="Input Preview",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        columns=2,
                        height=400,
                        object_fit="contain",
                    )
                    input_upload_status = gr.Markdown(
                        "Upload one image for single-image mode. Upload multiple angles of the same object "
                        "to enable multi-image mode. The first image is used as the primary view."
                    )
                    multiview_mode = gr.Dropdown(
                        choices=["stochastic", "multidiffusion"],
                        value="multidiffusion",
                        label="Multi-View Fusion Mode (Generate)",
                        info=(
                            "stochastic: rotates conditioning views across steps (faster). "
                            "multidiffusion: averages all views each step (more consistent, slower)."
                        ),
                    )

                    with gr.Row():
                        resolution = gr.Radio(["512", "768", "1024", "1280", "1536", "2048"], label="Resolution (Generate)", value="1024", info="Output mesh resolution. Higher = finer detail but more VRAM. 512 uses direct sampling; 768+ use cascade for quality. ⬆Quality ⬆VRAM", scale=3)
                        custom_resolution = gr.Number(label="Custom Resolution", value=0, precision=0, minimum=0, maximum=4096, step=128, info="Set to 0 to use radio selection. Must be ≥512 and divisible by 128. Overrides radio if >0.", scale=1)
                    pipeline_strategy = gr.Dropdown(
                        PIPELINE_STRATEGY_CHOICES,
                        label="Pipeline Strategy (Generate)",
                        value="reference_auto",
                        info=(
                            "reference_auto keeps the current high-quality 1024 cascade path. "
                            "direct_1024 uses the upstream direct 1024 model. "
                            "hybrid_512g_1024t uses 512 geometry with 1024 texture and is useful when Generate needs a lower-VRAM fallback."
                        ),
                    )
                    with gr.Row():
                        seed = gr.Slider(0, MAX_SEED, label="Seed (Generate)", value=99, step=1, scale=4, info="Random seed for reproducibility. Same seed + settings = same output.")
                        randomize_seed = gr.Checkbox(label="Randomize Seed (Generate)", value=False, scale=1, info="Generate random seed each run for variety.")
                    decimation_target = gr.Slider(100000, 9000000, label="Decimation Target (Extract GLB)", value=1000000, step=10000, info="Target polygon count during mesh simplification. Higher = more geometric detail preserved but larger files. ⬆Quality, minimal VRAM impact.")
                    remesh_method = gr.Dropdown(
                        REMESH_METHOD_CHOICES,
                        label="Remesh Method (Extract GLB)",
                        value=("dual_contouring_vb" if "dual_contouring_vb" in REMESH_METHOD_CHOICES else "dual_contouring"),
                        info=(
                            "Surface reconstruction algorithm used during Extract GLB. "
                            "dual_contouring_vb is the preferred default when the installed CuMesh build supports it, "
                            "dual_contouring is the fallback path, "
                            "and faithful_contouring preserves thin/open geometry best but depends on FaithC."
                        ),
                    )
                    if "faithful_contouring" not in REMESH_METHOD_CHOICES:
                        gr.Markdown(
                            "**Note:** `faithful_contouring` remeshing requires optional FaithC dependencies "
                            "(`faithcontour` + `atom3d`). Not detected in this environment, so the option is hidden."
                        )
                    if "dual_contouring_vb" not in REMESH_METHOD_CHOICES:
                        gr.Markdown(
                            "**Note:** `dual_contouring_vb` is hidden because the current CuMesh build does not expose "
                            "`reconstruct_mesh_dc`. If you build/install a newer wheel, this upstream remesher can be enabled."
                        )
                    if "pymeshfix" not in REPAIR_METHOD_CHOICES:
                        gr.Markdown(
                            "**Note:** `pymeshfix` repair is hidden because its full runtime stack is unavailable in this venv. "
                            "That path needs both `pymeshfix` and `pyvista` importable."
                        )
                    simplify_method = gr.Dropdown(
                        SIMPLIFY_METHOD_CHOICES,
                        label="Simplify Method (Extract GLB)",
                        value="cumesh",
                        info="Polygon reduction backend for Extract GLB. cumesh is the tested default, meshlib is the upstream CPU alternative for manual use, and none skips simplification."
                    )
                    prune_invisible_faces = gr.Checkbox(label="Prune Invisible Faces (Extract GLB)", value=False, info="Remove triangles not visible from outside. Reduces polygon count, may affect internal geometry. Slight ⬇VRAM.")
                    with gr.Row():
                        shade_smooth = gr.Checkbox(
                            label="Enable Shade Smooth (Extract GLB)",
                            value=False,
                            info="Apply smooth-shaded normals to the Extract GLB export. Works together with Shade Smooth Angle."
                        )
                        force_double_sided = gr.Checkbox(
                            label="Force Double-Sided Materials (Extract GLB)",
                            value=True,
                            info="Keep Extract GLB materials double-sided after remesh. Helpful for thin sheets, leaves, cloth, and open surfaces."
                        )
                        no_pbr_export = gr.Checkbox(
                            label="Basecolor-Only Material (Extract GLB)",
                            value=False,
                            info="Export Extract GLB with base color + alpha only and skip metallic/roughness textures. Useful for DCC cleanup or lighter downstream materials."
                        )
                    no_texture_gen = gr.Checkbox(label="Skip Texture Generation (Generate + Extract GLB)", value=False, info="Output shape-only mesh without PBR textures. Faster processing, significantly ⬇VRAM usage.")
                    deferred_texture_after_cleanup = gr.Checkbox(
                        label="Deferred Texture Rebuild (Extract GLB)",
                        value=True,
                        info="Run a separate final TRELLIS texturing pass on the cleaned extracted mesh. Improves texture/mesh alignment after remesh/simplify; slower runtime. Runs as staged subprocesses."
                    )
                    projection_texture_refine = gr.Checkbox(
                        label="Projection Texture Refinement (Extract GLB)",
                        value=False,
                        info="Project the input view images onto the cleaned extracted mesh using camera angles. Best for 2-6 known views. Overrides Deferred Texture Rebuild when enabled.",
                    )
                    texture_size = gr.Slider(1024, 4096, label="Texture Size (Extract GLB)", value=4096, step=1024, info="Resolution of baked texture maps (albedo, normal, etc). Higher = sharper textures. ⬆Quality ⬆VRAM during baking.")
                    export_formats = gr.CheckboxGroup(
                        choices=["glb", "gltf", "obj", "ply", "stl"],
                        value=["glb", "gltf", "obj", "ply", "stl"],
                        label="Export Formats (Extract GLB)",
                    )
                    example_image_dir = os.path.join(APP_DIR, "assets", "example_image")
                    example_image_paths = [
                        os.path.join(example_image_dir, image)
                        for image in sorted(os.listdir(example_image_dir), key=_example_image_sort_key)
                        if os.path.isfile(os.path.join(example_image_dir, image))
                    ]
                    gr.Markdown(
                        "### Input Examples\nClick a thumbnail to load it as the current input and jump back to the top input area."
                    )
                    image_examples = gr.Gallery(
                        value=example_image_paths,
                        label=None,
                        show_label=False,
                        container=False,
                        columns=5,
                        object_fit="contain",
                        allow_preview=False,
                        interactive=False,
                        type="filepath",
                    )

                    def _load_image_example(evt: gr.SelectData):
                        if evt is None:
                            return None
                        index = evt.index
                        if isinstance(index, (list, tuple)):
                            index = index[0] if index else None
                        if not isinstance(index, int):
                            return None
                        if index < 0 or index >= len(example_image_paths):
                            return None
                        path = example_image_paths[index]
                        if not path:
                            return None
                        return [str(path)]

                with gr.Column(scale=3, min_width=680):
                    with gr.Walkthrough(selected=0) as walkthrough:
                        with gr.Step("Preview", id=0):
                            with gr.Column(elem_id="preview_stack"):
                                preview_output = gr.HTML(
                                    empty_html, label="3D Asset Preview", show_label=True, container=True
                                )
                                # Progress shown directly on top of the preview (no separate side panel).
                                status_box = gr.Textbox(
                                    value="Upload one or more images (or pick an example), then click Generate (Step 1).",
                                    lines=20,
                                    max_lines=20,
                                    interactive=False,
                                    show_label=False,
                                    container=False,
                                    elem_id="preview_status_overlay",
                                )
                            with gr.Row():
                                generate_btn = gr.Button(
                                    "Generate (Step 1)",
                                    variant="primary",
                                    elem_id="generate_step_btn",
                                    elem_classes=["stage-action-btn", "generate-stage-btn"],
                                )
                                extract_btn = gr.Button(
                                    "Extract GLB (Step 2)",
                                    interactive=False,
                                    elem_id="extract_step_btn",
                                    elem_classes=["stage-action-btn", "extract-stage-btn"],
                                )
                                view_extract_btn = gr.Button("View Extracted", interactive=False)
                            cancel_confirm_state = gr.State({"armed": False, "armed_at": 0.0, "scope": ""})
                            with gr.Row():
                                open_outputs_top_btn = gr.Button("📂 Open outputs folder", variant="secondary")
                                view_logs_btn = gr.Button("📄 View Logs", variant="secondary")
                                cancel_processing_btn = gr.Button("🛑 Cancel processing", variant="stop")
                            with gr.Accordion("Config Presets (Save / Load)", open=True):
                                gr.Markdown(
                                    "Saves/loads **all settings** from Image->3D, Texturing, UltraShape Refine, and Rigging tabs (uploaded images/files are not included)."
                                )
                                with gr.Row():
                                    ui_preset_dropdown = gr.Dropdown(
                                        label="Select Preset",
                                        choices=_list_ui_presets(),
                                        value=(_get_last_used_ui_preset() or "best"),
                                        allow_custom_value=False,
                                        scale=2,
                                    )
                                    ui_preset_name = gr.Textbox(
                                        label="New Preset Name",
                                        placeholder="my_settings",
                                        scale=2,
                                    )
                                    ui_preset_save_btn = gr.Button("💾 Save", variant="primary", scale=1)
                                with gr.Row():
                                    ui_preset_load_btn = gr.Button("📂 Load Selected", scale=1)
                                    ui_preset_reset_btn = gr.Button("🔄 Reset Defaults", variant="secondary", scale=1)
                                    ui_preset_delete_btn = gr.Button("Delete", variant="stop", scale=1)
                                ui_preset_status = gr.Markdown("")
                            with gr.Accordion(label="📦 Batch Processing", open=False):
                                batch_enabled = gr.Checkbox(label="Enable batch processing", value=False)
                                batch_input_folder = gr.Textbox(
                                    label="Input folder (required)",
                                    placeholder="e.g. ./my_images (or an absolute path)",
                                )
                                batch_output_folder = gr.Textbox(
                                    label="Output folder (optional)",
                                    placeholder="Leave blank to use ./outputs",
                                )
                                with gr.Row():
                                    batch_run_btn = gr.Button("Run Batch", variant="primary", interactive=False)
                                batch_status_box = gr.Textbox(
                                    label="Batch Progress",
                                    value="",
                                    lines=12,
                                    interactive=False,
                                )
                            with gr.Accordion("Projection Texture Settings (Extract GLB)", open=True):
                                with gr.Row():
                                    projection_view_azimuths = gr.Textbox(
                                        label="Projection View Azimuths (Extract GLB)",
                                        value="",
                                        placeholder="e.g. 0,180,90,270",
                                        info="Comma-separated azimuth angles in degrees matching uploaded image order. Leave blank to use default view orders for 1, 2, 4, or 6 images.",
                                    )
                                    projection_view_elevations = gr.Textbox(
                                        label="Projection View Elevations (Extract GLB)",
                                        value="",
                                        placeholder="e.g. 0,0,0,0",
                                        info="Comma-separated elevation angles in degrees matching uploaded image order. Leave blank to use the default preset with front/back/side/top/bottom assumptions.",
                                    )
                                with gr.Row():
                                    projection_blend_exponent = gr.Slider(
                                        0.5,
                                        8.0,
                                        label="Projection Blend Exponent (Extract GLB)",
                                        value=2.0,
                                        step=0.5,
                                        info="Higher values favor the most front-facing camera for each texel; lower values blend views more evenly.",
                                    )
                                    projection_ortho_scale = gr.Slider(
                                        0.5,
                                        2.5,
                                        label="Projection Ortho Scale (Extract GLB)",
                                        value=1.1,
                                        step=0.05,
                                        info="Approximate orthographic camera framing used when mapping views back to the mesh. Increase if projections look cropped.",
                                    )
                                    projection_fill_holes = gr.Checkbox(
                                        label="Fill Projection Holes (Extract GLB)",
                                        value=True,
                                        info="Inpaint uncovered texels and add seam padding after projection.",
                                    )
                                    projection_max_hole_size = gr.Slider(
                                        0,
                                        256,
                                        label="Projection Max Hole Size (Extract GLB)",
                                        value=20,
                                        step=1,
                                        info="Limit internal-hole filling to patches at or below this size. Set 0 to fill all internal holes.",
                                    )
                            with gr.Row():
                                fill_holes_max_perimeter = gr.Slider(
                                    0.0,
                                    0.2,
                                    label="Fill Holes Max Perimeter (Extract GLB)",
                                    value=0.03,
                                    step=0.005,
                                    info="Hole-filling threshold for Extract GLB cleanup. Higher values close larger gaps but can over-seal intended openings."
                                )
                                repair_method = gr.Dropdown(
                                    REPAIR_METHOD_CHOICES,
                                    label="Repair Method (Extract GLB)",
                                    value=("meshlib" if "meshlib" in REPAIR_METHOD_CHOICES else "disabled"),
                                    info="Final repair backend for Extract GLB. meshlib is the tested default when available because it closes the extracted shell more reliably on the current pipeline; cumesh is the lighter fallback; pymeshfix is a slow manual last-resort option on large meshes; disabled skips the final repair pass."
                                )
                            with gr.Row():
                                merge_vertices_dist = gr.Slider(
                                    0.0,
                                    0.05,
                                    label="Merge Vertices Distance (Extract GLB)",
                                    value=0.0,
                                    step=0.001,
                                    info="Optional post-remesh vertex welding distance for Extract GLB. Useful for tiny cracks or duplicate seams; keep near zero unless you need cleanup."
                                )
                                shade_smooth_angle = gr.Slider(
                                    0,
                                    80,
                                    label="Shade Smooth Angle (Extract GLB)",
                                    value=35,
                                    step=1,
                                    info="Normal-smoothing split angle for Extract GLB exports. 0 keeps the existing normals; 30-45 often improves visual smoothness without removing sharp edges."
                                )
                            with gr.Accordion(label="Advanced Settings Trellis2 (Generate)", open=True):
                                gr.Markdown("**Stage 1: Sparse Structure Generation (Generate)**")
                                with gr.Row():
                                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1, info="CFG scale - how strongly model follows image. Higher = more faithful but can oversaturate. 7.5 default. Slight ⬆VRAM (2 forward passes).")
                                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01, info="Reduces over-exposure from high CFG by normalizing variance. 0.7 recommended. No VRAM impact.")
                                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1, info="Denoising iterations. More = cleaner but slower. 12 is efficient. ⬆Quality, no per-step VRAM increase.")
                                with gr.Row():
                                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1, info="Time schedule warping. Higher = more steps on coarse structure. 5.0 default improves structure. No VRAM impact.")
                                    ss_guidance_interval_start = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01, info="WARNING: ADVANCED: Model default is 0.6. Only apply CFG in final refinement phase. Changing may reduce quality!")
                                    ss_guidance_interval_end = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval End", value=1.0, step=0.01, info="WARNING: ADVANCED: Model default is 1.0. Keep at 1.0 unless you know what you're doing.")
                                with gr.Row():
                                    force_high_res_conditional = gr.Checkbox(
                                        label="Force High-Res Conditioning (Generate)",
                                        value=False,
                                        info="Use 1024 resolution for sparse structure conditioning instead of 512. May improve stability but increases VRAM usage."
                                    )
                                    low_vram = gr.Checkbox(
                                        label="Low VRAM Mode",
                                        value=False,
                                        info="Move models between CPU/GPU during generation. Reduces VRAM usage but slower and may reduce quality. Disable for best results."
                                    )
                                with gr.Row():
                                    model_variant = gr.Dropdown(
                                        MODEL_VARIANT_CHOICES,
                                        label="Model Weights",
                                        value=DEFAULT_MODEL_VARIANT,
                                        info="standard uses the Microsoft BF16/FP16 checkpoints. fp8 uses the external FP8 checkpoint pack for lower VRAM and faster generation, but outputs are not numerically identical.",
                                    )
                                    attention_backend = gr.Dropdown(
                                        ATTENTION_BACKEND_CHOICES,
                                        label="Attention Backend",
                                        value=DEFAULT_ATTENTION_BACKEND,
                                        info="auto picks the best dense attention backend and a compatible sparse fallback. sdpa is supported for dense attention; sparse attention will safely fall back when needed.",
                                    )
                                    sampler_type = gr.Dropdown(
                                        SAMPLER_TYPE_CHOICES,
                                        label="Sampler",
                                        value=DEFAULT_SAMPLER_TYPE,
                                        info="Heun is the default generate sampler. Euler, RK4, and RK5 are available for speed or higher-order integration tradeoffs.",
                                    )
                                with gr.Row():
                                    use_chunked_processing = gr.Checkbox(
                                        label="Chunked Triangle Processing (Generate)",
                                        value=False,
                                        info="Process mesh triangles in chunks during Generate preview. Reduces VRAM spikes, no quality impact. Enable if you get OOM."
                                    )
                                    use_tiled_extraction = gr.Checkbox(
                                        label="Tiled Mesh Extraction (Generate)",
                                        value=False,
                                        info="Extract mesh in spatial tiles during Generate preview. Only enable if you get OOM during Generate. May degrade quality."
                                    )
                                    extract_use_chunked_processing = gr.Checkbox(
                                        label="Chunked Triangle Processing (Extract GLB)",
                                        value=False,
                                        info="Process mesh triangles in chunks during Extract GLB. Reduces VRAM spikes, no quality impact. Enable if you get OOM."
                                    )
                                    extract_use_tiled_extraction = gr.Checkbox(
                                        label="Tiled Mesh Extraction (Extract GLB)",
                                        value=False,
                                        info="Extract mesh in spatial tiles during Extract GLB. Only enable if you get OOM during extraction. May degrade quality."
                                    )

                                gr.Markdown("**Stage 2: Shape Generation (Generate)**")
                                with gr.Row():
                                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1, info="CFG for shape latent. Higher = stronger image adherence. 7.5 default. Slight ⬆VRAM (2 passes).")
                                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01, info="Variance normalization to prevent CFG artifacts. 0.5 recommended. No VRAM impact.")
                                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1, info="Denoising steps for shape. More = cleaner geometry. ⬆Quality, no per-step VRAM increase.")
                                with gr.Row():
                                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1, info="Time warping for shape sampling. 3.0 default balances coarse/fine detail. No VRAM impact.")
                                    shape_slat_guidance_interval_start = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01, info="WARNING: ADVANCED: Model default is 0.6. Only apply CFG in final refinement phase. Changing may reduce quality!")
                                    shape_slat_guidance_interval_end = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval End", value=1.0, step=0.01, info="WARNING: ADVANCED: Model default is 1.0. Keep at 1.0 unless you know what you're doing.")
                                    max_num_tokens = gr.Slider(
                                        10000, 999999,
                                        label="Max Tokens (Generate - VRAM vs Quality)",
                                        value=400000,
                                        step=1000,
                                        info="KEY VRAM CONTROL. Max voxel tokens in cascade. Lower = less VRAM and possible auto-resolution reduction. Higher = more potential detail with higher VRAM usage.")

                                gr.Markdown("**Stage 3: Material Generation (Generate)**")
                                with gr.Row():
                                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1, info="CFG for texture. Low (1.0) works well since shape provides strong conditioning. Slight ⬆VRAM if >1.")
                                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01, info="Variance normalization. 0.0 = disabled (not needed at low guidance). No VRAM impact.")
                                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1, info="Steps for texture generation. 12 is efficient. ⬆Quality, no per-step VRAM increase.")
                                with gr.Row():
                                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1, info="Time warping for texture. 3.0 default. No VRAM impact.")
                                    tex_slat_guidance_interval_start = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01, info="WARNING: ADVANCED: Model default is 0.6. Apply CFG in middle refinement phase. Changing may reduce quality!")
                                    tex_slat_guidance_interval_end = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval End", value=0.9, step=0.01, info="WARNING: ADVANCED: Model default is 0.9. Texture uses 0.6-0.9 range. Changing may reduce quality!")
                                gr.Markdown("**UltraShape Refinement (Extract GLB)**")
                                with gr.Row():
                                    ultrashape_enabled = gr.Checkbox(
                                        label="Enable UltraShape Refinement",
                                        value=False,
                                        info="Run UltraShape image-guided mesh refinement before final GLB baking. Improves geometry detail; adds time/VRAM."
                                    )
                                    ultrashape_retexture_after_refine = gr.Checkbox(
                                        label="Re-generate Texture After UltraShape",
                                        value=False,
                                        info="Rebuild texture maps after UltraShape mesh changes to keep textures aligned. Adds runtime."
                                    )
                                    ultrashape_conservative_mode = gr.Checkbox(
                                        label="Conservative UltraShape Mode",
                                        value=False,
                                        info="Reduces geometry drift to preserve TRELLIS texture/shape alignment. Slightly less aggressive refinement."
                                    )
                                with gr.Row():
                                    ultrashape_checkpoint = gr.Textbox(
                                        label="UltraShape Checkpoint (optional)",
                                        value="",
                                        placeholder="Leave empty to auto-pick first file in models/UltraShape",
                                    )
                                    ultrashape_config_name = gr.Textbox(
                                        label="UltraShape Config",
                                        value="infer_dit_refine.yaml",
                                    )
                                with gr.Row():
                                    ultrashape_dtype = gr.Dropdown(
                                        ["float16", "bfloat16", "float32"],
                                        label="UltraShape DType",
                                        value="bfloat16",
                                        info="Precision mode for UltraShape inference. Lower precision usually reduces VRAM and can improve speed; float32 is typically heavier and slower.",
                                    )
                                    ultrashape_low_vram = gr.Checkbox(
                                        label="UltraShape Low VRAM",
                                        value=True,
                                        info="Enable CPU offload where supported. Lower VRAM, slower runtime."
                                    )
                                    ultrashape_remove_bg = gr.Checkbox(
                                        label="UltraShape Remove BG",
                                        value=False,
                                        info="Apply TRELLIS rembg to the reference image before refinement. Adds an extra preprocessing pass (more runtime, small extra memory). Affects image guidance only, not the input mesh."
                                    )
                                with gr.Row():
                                    ultrashape_steps = gr.Slider(
                                        10, 200,
                                        label="UltraShape Steps",
                                        value=50,
                                        step=5,
                                        info="More diffusion steps increase processing time. VRAM impact is usually modest compared with resolution/token controls."
                                    )
                                    ultrashape_guidance_scale = gr.Slider(
                                        1.0, 15.0,
                                        label="UltraShape Guidance",
                                        value=5.0,
                                        step=0.5,
                                        info="Conditioning strength. Usually has minor VRAM/runtime impact versus steps and resolution."
                                    )
                                    ultrashape_octree_resolution = gr.Slider(
                                        256, 1024,
                                        label="UltraShape Octree Res",
                                        value=384,
                                        step=64,
                                        info="Higher octree resolution increases VRAM usage and runtime during geometry decoding."
                                    )
                                with gr.Row():
                                    ultrashape_num_chunks = gr.Slider(
                                        1000, 50000,
                                        label="UltraShape Chunk Size",
                                        value=8000,
                                        step=1000,
                                        info="Larger chunks are usually faster but need more VRAM; smaller chunks reduce VRAM pressure but run slower."
                                    )
                                    ultrashape_target_face_count = gr.Slider(
                                        100000, 2000000,
                                        label="UltraShape Target Faces",
                                        value=500000,
                                        step=10000,
                                        info="Higher face targets increase mesh processing time and can raise memory use in later mesh operations."
                                    )
                                    ultrashape_num_latents = gr.Slider(
                                        0, 131072,
                                        label="UltraShape Num Latents (0=cfg)",
                                        value=0,
                                        step=1024,
                                        info="More latent tokens increase VRAM usage and runtime during voxel conditioning/sampling."
                                    )
                                with gr.Row():
                                    ultrashape_box_v = gr.Slider(
                                        0.5, 2.0,
                                        label="UltraShape Box V",
                                        value=1.0,
                                        step=0.1,
                                        info="Geometric control parameter. Usually negligible direct VRAM/runtime impact."
                                    )
                                    ultrashape_mc_level = gr.Slider(
                                        -0.1, 0.1,
                                        label="UltraShape MC Level",
                                        value=0.0,
                                        step=0.01,
                                        info="Surface threshold parameter. Usually negligible direct VRAM/runtime impact."
                                    )
                                    ultrashape_normalize_scale = gr.Slider(
                                        0.5, 1.0,
                                        label="UltraShape Normalize Scale",
                                        value=0.99,
                                        step=0.01,
                                        info="Input normalization control. Usually negligible direct VRAM/runtime impact."
                                    )
                                with gr.Row():
                                    ultrashape_num_sharp_points = gr.Slider(
                                        10000, 500000,
                                        label="UltraShape Sharp Points",
                                        value=204800,
                                        step=10000,
                                        info="More sharp-edge samples improve detail capture but increase sampling/voxelization runtime and memory."
                                    )
                                    ultrashape_num_uniform_points = gr.Slider(
                                        10000, 500000,
                                        label="UltraShape Uniform Points",
                                        value=204800,
                                        step=10000,
                                        info="More uniform samples improve coverage but increase sampling/voxelization runtime and memory."
                                    )
                        with gr.Step("Extract", id=1):
                            with gr.Row():
                                back_to_preview_btn = gr.Button("Back to Preview", variant="secondary")
                                fullscreen_glb_btn = gr.Button("Fullscreen", variant="secondary")
                            glb_output = gr.Model3D(
                                label="Extracted GLB",
                                height=724,
                                show_label=True,
                                display_mode="solid",
                                clear_color=(0.25, 0.25, 0.25, 1.0),
                                elem_id="extracted_glb_viewer",
                            )
                            with gr.Row():
                                download_btn = gr.DownloadButton(label="Download GLB", variant="primary")
                                open_outputs_btn = gr.Button("Open outputs folder", variant="secondary")

            output_buf = gr.State()
            # State to track logs visibility (starts as visible during generation)
            logs_visible_state = gr.State(True)

            def _reset_image_to_3d_ui():
                return (
                    None,  # output_buf
                    empty_html,  # preview_output
                    gr.update(interactive=False),  # extract_btn
                    gr.update(interactive=False),  # view_extract_btn
                    gr.Walkthrough(selected=0),  # walkthrough
                    None,  # glb_output
                    None,  # download_btn
                    gr.update(
                        visible=True,
                        value="Upload one or more images (or pick an example), then click Generate (Step 1).",
                    ),  # status_box
                    True,  # logs_visible_state
                    gr.update(value="📄 Hide Logs"),  # view_logs_btn
                )

            # Note: We intentionally do not auto-preprocess on upload/example click.
            # Both in-process and subprocess pipelines do preprocessing as part of "Generate",
            # and each run saves both the raw input + the preprocessed image under outputs/<run_id>/.

            # Any time the input image changes (upload, example click, clear), invalidate previous results.
            input_images.change(
                _update_uploaded_input_preview,
                inputs=[input_images],
                outputs=[input_preview_single, input_preview_gallery, input_upload_status],
            )
            image_examples.select(
                None,
                inputs=[],
                outputs=[],
                js="""() => {
                    const target = document.getElementById("image_input_panel");
                    if (target) {
                        window.setTimeout(() => {
                            target.scrollIntoView({ behavior: "smooth", block: "start" });
                        }, 0);
                    }
                }""",
                queue=False,
                api_visibility="private",
            )
            image_examples.select(
                _load_image_example,
                inputs=[],
                outputs=[input_images],
                show_progress="hidden",
                queue=False,
            )
            input_images.change(
                _reset_image_to_3d_ui,
                inputs=[],
                outputs=[output_buf, preview_output, extract_btn, view_extract_btn, walkthrough, glb_output, download_btn, status_box, logs_visible_state, view_logs_btn],
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
                outputs=[output_buf, preview_output, extract_btn, view_extract_btn, walkthrough, glb_output, download_btn, status_box, logs_visible_state, view_logs_btn],
            ).then(
                image_to_3d,
                inputs=[
                    input_images,
                    seed,
                    resolution,
                    custom_resolution,
                    pipeline_strategy,
                    ss_guidance_strength,
                    ss_guidance_rescale,
                    ss_guidance_interval_start,
                    ss_guidance_interval_end,
                    ss_sampling_steps,
                    ss_rescale_t,
                    force_high_res_conditional,
                    low_vram,
                    model_variant,
                    attention_backend,
                    sampler_type,
                    use_chunked_processing,
                    use_tiled_extraction,
                    shape_slat_guidance_strength,
                    shape_slat_guidance_rescale,
                    shape_slat_guidance_interval_start,
                    shape_slat_guidance_interval_end,
                    shape_slat_sampling_steps,
                    shape_slat_rescale_t,
                    tex_slat_guidance_strength,
                    tex_slat_guidance_rescale,
                    tex_slat_guidance_interval_start,
                    tex_slat_guidance_interval_end,
                    tex_slat_sampling_steps,
                    tex_slat_rescale_t,
                    no_texture_gen,
                    projection_texture_refine,
                    projection_view_azimuths,
                    projection_view_elevations,
                    projection_blend_exponent,
                    projection_ortho_scale,
                    projection_fill_holes,
                    projection_max_hole_size,
                    max_num_tokens,
                    multiview_mode,
                    subprocess_mode,
                ],
                outputs=[output_buf, preview_output, status_box],
            ).then(
                # Enable extract button only if we have valid latent paths (even if preview failed)
                lambda state: gr.update(interactive=True) if (
                    isinstance(state, dict) and state.get("shape_slat_path")
                ) else gr.update(interactive=False),
                inputs=[output_buf],
                outputs=extract_btn
            )

            # Keep users on the Preview step while extracting so progress stays visible on the preview.
            extract_btn.click(
                extract_glb,
                inputs=[
                    output_buf,
                    decimation_target,
                    texture_size,
                    remesh_method,
                    fill_holes_max_perimeter,
                    repair_method,
                    simplify_method,
                    no_texture_gen,
                    deferred_texture_after_cleanup,
                    projection_texture_refine,
                    projection_view_azimuths,
                    projection_view_elevations,
                    projection_blend_exponent,
                    projection_ortho_scale,
                    projection_fill_holes,
                    projection_max_hole_size,
                    prune_invisible_faces,
                    merge_vertices_dist,
                    shade_smooth,
                    shade_smooth_angle,
                    force_double_sided,
                    no_pbr_export,
                    export_formats,
                    extract_use_chunked_processing,
                    extract_use_tiled_extraction,
                    ultrashape_enabled,
                    ultrashape_retexture_after_refine,
                    ultrashape_conservative_mode,
                    ultrashape_checkpoint,
                    ultrashape_config_name,
                    ultrashape_dtype,
                    ultrashape_low_vram,
                    ultrashape_remove_bg,
                    ultrashape_steps,
                    ultrashape_guidance_scale,
                    ultrashape_octree_resolution,
                    ultrashape_num_chunks,
                    ultrashape_target_face_count,
                    ultrashape_num_latents,
                    ultrashape_box_v,
                    ultrashape_mc_level,
                    ultrashape_normalize_scale,
                    ultrashape_num_sharp_points,
                    ultrashape_num_uniform_points,
                    subprocess_mode,
                ],
                outputs=[glb_output, download_btn, status_box],
            ).then(
                # Enable "View Extracted" and automatically switch to the Extract step once ready.
                lambda: (gr.update(interactive=True), gr.Walkthrough(selected=1)),
                outputs=[view_extract_btn, walkthrough],
            )

            # Navigation-only controls (do NOT re-run extraction)
            view_extract_btn.click(lambda: gr.Walkthrough(selected=1), outputs=walkthrough)
            back_to_preview_btn.click(
                lambda: (gr.Walkthrough(selected=0), gr.update(visible=False), False, gr.update(value="📄 View Logs")),
                outputs=[walkthrough, status_box, logs_visible_state, view_logs_btn]
            )

            # Fullscreen toggle for the extracted 3D viewer (client-side only)
            fullscreen_glb_btn.click(
                fn=None,
                inputs=[],
                outputs=None,
                js="""
() => {
  const root = document.querySelector("#extracted_glb_viewer");
  const el = root?.querySelector('[data-testid="model3d"]') || root;
  if (!el) return;
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
  } else {
    el.requestFullscreen?.();
  }
}
""",
            )

            def _open_outputs_from_image_tab(current_status: str) -> str:
                os.makedirs(OUTPUTS_DIR, exist_ok=True)
                ts = datetime.now().strftime("%H:%M:%S")
                try:
                    _open_folder(OUTPUTS_DIR)
                    return _append_status(
                        current_status,
                        f"[{ts}] Opened outputs folder: {safe_relpath(OUTPUTS_DIR, APP_DIR)}",
                    )
                except Exception as e:
                    return _append_status(current_status, f"[{ts}] Could not open outputs folder: {e}")

            open_outputs_btn.click(
                fn=_open_outputs_from_image_tab,
                inputs=[status_box],
                outputs=[status_box],
                queue=False,
                show_progress="hidden",
            )

            def _open_outputs_from_main_controls(current_status: str, current_batch_status: str) -> Tuple[str, str]:
                os.makedirs(OUTPUTS_DIR, exist_ok=True)
                ts = datetime.now().strftime("%H:%M:%S")
                try:
                    _open_folder(OUTPUTS_DIR)
                    msg = f"[{ts}] Opened outputs folder: {safe_relpath(OUTPUTS_DIR, APP_DIR)}"
                except Exception as e:
                    msg = f"[{ts}] Could not open outputs folder: {e}"
                return (
                    _append_status(current_status, msg),
                    _append_status(current_batch_status, msg),
                )

            open_outputs_top_btn.click(
                fn=_open_outputs_from_main_controls,
                inputs=[status_box, batch_status_box],
                outputs=[status_box, batch_status_box],
                queue=False,
                show_progress="hidden",
            )
            
            def _toggle_logs(current_visible: bool) -> tuple:
                """Toggle visibility of status logs."""
                new_visible = not current_visible
                btn_text = "📄 Hide Logs" if new_visible else "📄 View Logs"
                return gr.update(visible=new_visible), new_visible, gr.update(value=btn_text)
            
            view_logs_btn.click(
                fn=_toggle_logs,
                inputs=[logs_visible_state],
                outputs=[status_box, logs_visible_state, view_logs_btn],
                queue=False,
                show_progress="hidden",
            )

            def _cancel_processing_click(
                confirm_state: dict,
                subprocess_mode: bool,
                current_status: str,
                current_batch_status: str,
                req: gr.Request,
            ) -> Tuple[dict, Any, str, str]:
                """
                Two-step cancel:
                  - 1st click arms cancellation (no-op)
                  - 2nd click within a short window triggers cancellation

                Behavior:
                  - If subprocess_mode is ON: cancels all processing and kills the active subprocess stage.
                  - If subprocess_mode is OFF: cancels batch processing only.
                """
                confirm_state = confirm_state if isinstance(confirm_state, dict) else {}
                now = time.time()
                session = _session_key(req)
                scope = "all" if subprocess_mode else "batch"
                # If a subprocess stage is currently running, always cancel-all (even if the checkbox is off),
                # because we *can* terminate it safely.
                proc, _stage = _get_active_subproc(session)
                if proc is not None:
                    scope = "all"

                armed = bool(confirm_state.get("armed", False))
                armed_at = float(confirm_state.get("armed_at", 0.0) or 0.0)
                armed_scope = str(confirm_state.get("scope", ""))

                ts = datetime.now().strftime("%H:%M:%S")
                confirm_window_s = 7.0

                if armed and armed_scope == scope and (now - armed_at) <= confirm_window_s:
                    msg = _cancel_now(session, scope=scope)
                    new_state = {"armed": False, "armed_at": 0.0, "scope": ""}
                    btn_update = gr.update(value="🛑 Cancel processing")
                    line = f"[{ts}] {msg}"
                    return (
                        new_state,
                        btn_update,
                        _append_status(current_status, line),
                        _append_status(current_batch_status, line),
                    )

                # Arm (no cancellation yet)
                label = "WARNING: CONFIRM cancel (click again)"
                if scope == "batch":
                    hint = (
                        f"[{ts}] Cancel armed. Click again to confirm (subprocess mode is OFF → batch only)."
                    )
                else:
                    hint = f"[{ts}] Cancel armed. Click again to confirm (this will stop ALL processing)."

                new_state = {"armed": True, "armed_at": now, "scope": scope}
                return (
                    new_state,
                    gr.update(value=label),
                    _append_status(current_status, hint),
                    _append_status(current_batch_status, hint),
                )

            cancel_processing_btn.click(
                fn=_cancel_processing_click,
                inputs=[cancel_confirm_state, subprocess_mode, status_box, batch_status_box],
                outputs=[cancel_confirm_state, cancel_processing_btn, status_box, batch_status_box],
                queue=False,
                show_progress="hidden",
            )

            # Batch Processing wiring (reuses the same image_to_3d -> extract_glb pipeline)
            batch_enabled.change(
                fn=lambda v: gr.update(interactive=bool(v)),
                inputs=[batch_enabled],
                outputs=[batch_run_btn],
                queue=False,
                show_progress="hidden",
            )
            batch_run_btn.click(
                fn=batch_process_folder,
                inputs=[
                    batch_enabled,
                    batch_input_folder,
                    batch_output_folder,
                    randomize_seed,
                    seed,
                    resolution,
                    custom_resolution,
                    pipeline_strategy,
                    ss_guidance_strength,
                    ss_guidance_rescale,
                    ss_guidance_interval_start,
                    ss_guidance_interval_end,
                    ss_sampling_steps,
                    ss_rescale_t,
                    force_high_res_conditional,
                    low_vram,
                    model_variant,
                    attention_backend,
                    sampler_type,
                    use_chunked_processing,
                    use_tiled_extraction,
                    shape_slat_guidance_strength,
                    shape_slat_guidance_rescale,
                    shape_slat_guidance_interval_start,
                    shape_slat_guidance_interval_end,
                    shape_slat_sampling_steps,
                    shape_slat_rescale_t,
                    tex_slat_guidance_strength,
                    tex_slat_guidance_rescale,
                    tex_slat_guidance_interval_start,
                    tex_slat_guidance_interval_end,
                    tex_slat_sampling_steps,
                    tex_slat_rescale_t,
                    no_texture_gen,
                    deferred_texture_after_cleanup,
                    projection_texture_refine,
                    projection_view_azimuths,
                    projection_view_elevations,
                    projection_blend_exponent,
                    projection_ortho_scale,
                    projection_fill_holes,
                    projection_max_hole_size,
                    max_num_tokens,
                    multiview_mode,
                    decimation_target,
                    texture_size,
                    remesh_method,
                    fill_holes_max_perimeter,
                    repair_method,
                    simplify_method,
                    prune_invisible_faces,
                    merge_vertices_dist,
                    shade_smooth,
                    shade_smooth_angle,
                    force_double_sided,
                    no_pbr_export,
                    export_formats,
                    ultrashape_enabled,
                    ultrashape_retexture_after_refine,
                    ultrashape_conservative_mode,
                    ultrashape_checkpoint,
                    ultrashape_config_name,
                    ultrashape_dtype,
                    ultrashape_low_vram,
                    ultrashape_remove_bg,
                    ultrashape_steps,
                    ultrashape_guidance_scale,
                    ultrashape_octree_resolution,
                    ultrashape_num_chunks,
                    ultrashape_target_face_count,
                    ultrashape_num_latents,
                    ultrashape_box_v,
                    ultrashape_mc_level,
                    ultrashape_normalize_scale,
                    ultrashape_num_sharp_points,
                    ultrashape_num_uniform_points,
                    subprocess_mode,
                ],
                outputs=[batch_status_box],
            )

        # ---------------------------- Tab 2: Texturing -------------------------------
        with gr.Tab("Texturing"):
            with gr.Row():
                with gr.Column(scale=1, min_width=380):
                    mesh_file = gr.File(label="Upload Mesh", file_types=[".glb", ".gltf", ".obj", ".ply", ".stl"], file_count="single")
                    tex_image = gr.Image(label="Reference Image", format="png", image_mode="RGBA", type="pil", height=400)

                    tex_resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
                    with gr.Row():
                        tex_seed = gr.Slider(0, MAX_SEED, label="Seed", value=99, step=1, scale=4)
                        tex_randomize_seed = gr.Checkbox(label="Randomize Seed", value=False, scale=1)
                    tex_texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

                    with gr.Accordion(label="Advanced Settings", open=True):
                        with gr.Row():
                            tex_low_vram = gr.Checkbox(
                                label="Low VRAM Mode",
                                value=True,
                                info="Move models between CPU/GPU during generation. Reduces VRAM usage. Recommended enabled for texturing to avoid OOM errors."
                            )
                        with gr.Row():
                            tex_attention_backend = gr.Dropdown(
                                ATTENTION_BACKEND_CHOICES,
                                label="Attention Backend",
                                value=DEFAULT_ATTENTION_BACKEND,
                                info="auto picks the best dense backend and a compatible sparse fallback for texturing stages.",
                            )
                            tex_sampler_type = gr.Dropdown(
                                SAMPLER_TYPE_CHOICES,
                                label="Sampler",
                                value=DEFAULT_SAMPLER_TYPE,
                                info="Heun is the default texturing sampler. Euler, RK4, and RK5 remain available.",
                            )
                        with gr.Row():
                            t_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                            t_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                        with gr.Row():
                            t_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                            t_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                        with gr.Row():
                            t_guidance_interval_start = gr.Slider(
                                0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01
                            )
                            t_guidance_interval_end = gr.Slider(
                                0.0, 1.0, label="Guidance Interval End", value=0.9, step=0.01
                            )

                    gr.Markdown("### Examples")
                    tex_examples = gr.Examples(
                        examples=[
                            [
                                os.path.join(APP_DIR, "assets", "example_texturing", "knight_helmet.glb"),
                                os.path.join(APP_DIR, "assets", "example_texturing", "knight_helmet.webp"),
                            ]
                        ],
                        inputs=[mesh_file, tex_image],
                        examples_per_page=6,
                    )

                with gr.Column(scale=2, min_width=520):
                    textured_glb_output = gr.Model3D(
                        label="Textured GLB",
                        height=724,
                        show_label=True,
                        display_mode="solid",
                        clear_color=(0.25, 0.25, 0.25, 1.0),
                        elem_id="textured_glb_viewer",
                    )
                    tex_status_box = gr.Textbox(
                        value="Upload a mesh + reference image (or use the example), then click Generate.",
                        lines=20,
                        max_lines=20,
                        interactive=False,
                        show_label=False,
                        container=False,
                        visible=False,
                    )
                    with gr.Row():
                        tex_generate_btn = gr.Button("Generate Textured GLB", variant="primary")
                        textured_download_btn = gr.DownloadButton(label="Download Textured GLB", variant="primary")
                    tex_cancel_confirm_state = gr.State({"armed": False, "armed_at": 0.0, "scope": ""})
                    with gr.Row():
                        tex_cancel_btn = gr.Button("Cancel processing", variant="stop")
                        tex_open_outputs_btn = gr.Button("Open outputs folder", variant="secondary")
                        tex_view_logs_btn = gr.Button("View Logs", variant="secondary")

            # State to track logs visibility for texturing tab
            tex_logs_visible_state = gr.State(False)

            def _tex_coerce_file_to_path(f):
                """Convert file object to path string for Model3D display."""
                if f is None:
                    return None
                if isinstance(f, str):
                    return f
                if isinstance(f, dict):
                    return f.get("name") or f.get("path")
                return getattr(f, "name", None) or str(f)

            # Display uploaded mesh in viewer when uploaded
            mesh_file.change(
                fn=_tex_coerce_file_to_path,
                inputs=[mesh_file],
                outputs=[textured_glb_output],
                queue=False,
                show_progress="hidden",
            )

            def _tex_toggle_logs(current_visible: bool) -> tuple:
                """Toggle visibility of status logs for texturing tab."""
                new_visible = not current_visible
                btn_text = "Hide Logs" if new_visible else "View Logs"
                return gr.update(visible=new_visible), new_visible, gr.update(value=btn_text)

            tex_view_logs_btn.click(
                fn=_tex_toggle_logs,
                inputs=[tex_logs_visible_state],
                outputs=[tex_status_box, tex_logs_visible_state, tex_view_logs_btn],
                queue=False,
                show_progress="hidden",
            )

            def _tex_cancel_processing_click(
                confirm_state: dict,
                subprocess_mode: bool,
                current_status: str,
                req: gr.Request,
            ) -> Tuple[dict, Any, str]:
                """Two-step cancel for texturing tab."""
                confirm_state = confirm_state if isinstance(confirm_state, dict) else {}
                now = time.time()
                session = _session_key(req)
                scope = "all" if subprocess_mode else "batch"
                proc, _stage = _get_active_subproc(session)
                if proc is not None:
                    scope = "all"

                armed = bool(confirm_state.get("armed", False))
                armed_at = float(confirm_state.get("armed_at", 0.0) or 0.0)
                armed_scope = str(confirm_state.get("scope", ""))

                ts = datetime.now().strftime("%H:%M:%S")
                confirm_window_s = 7.0

                if armed and armed_scope == scope and (now - armed_at) <= confirm_window_s:
                    msg = _cancel_now(session, scope=scope)
                    new_state = {"armed": False, "armed_at": 0.0, "scope": ""}
                    btn_update = gr.update(value="Cancel processing")
                    line = f"[{ts}] {msg}"
                    return (
                        new_state,
                        btn_update,
                        _append_status(current_status, line),
                    )

                # Arm (no cancellation yet)
                label = "CONFIRM cancel (click again)"
                if scope == "batch":
                    hint = f"[{ts}] Cancel armed. Click again to confirm (subprocess mode is OFF)."
                else:
                    hint = f"[{ts}] Cancel armed. Click again to confirm (this will stop processing)."

                new_state = {"armed": True, "armed_at": now, "scope": scope}
                return (
                    new_state,
                    gr.update(value=label),
                    _append_status(current_status, hint),
                )

            tex_cancel_btn.click(
                fn=_tex_cancel_processing_click,
                inputs=[tex_cancel_confirm_state, subprocess_mode, tex_status_box],
                outputs=[tex_cancel_confirm_state, tex_cancel_btn, tex_status_box],
                queue=False,
                show_progress="hidden",
            )

            tex_generate_btn.click(
                get_seed,
                inputs=[tex_randomize_seed, tex_seed],
                outputs=[tex_seed],
            ).then(
                # Show logs during generation
                lambda: (gr.update(visible=True), True, gr.update(value="Hide Logs")),
                outputs=[tex_status_box, tex_logs_visible_state, tex_view_logs_btn],
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
                    t_guidance_interval_start,
                    t_guidance_interval_end,
                    t_sampling_steps,
                    t_rescale_t,
                    tex_low_vram,
                    tex_attention_backend,
                    tex_sampler_type,
                    subprocess_mode,
                ],
                outputs=[textured_glb_output, textured_download_btn, tex_status_box],
            ).then(
                # Hide logs after generation completes
                lambda: (gr.update(visible=False), False, gr.update(value="View Logs")),
                outputs=[tex_status_box, tex_logs_visible_state, tex_view_logs_btn],
            )

            def _open_outputs_from_texturing_tab(current_status: str) -> str:
                os.makedirs(OUTPUTS_DIR, exist_ok=True)
                ts = datetime.now().strftime("%H:%M:%S")
                try:
                    _open_folder(OUTPUTS_DIR)
                    return _append_status(
                        current_status,
                        f"[{ts}] Opened outputs folder: {safe_relpath(OUTPUTS_DIR, APP_DIR)}",
                    )
                except Exception as e:
                    return _append_status(current_status, f"[{ts}] Could not open outputs folder: {e}")

            tex_open_outputs_btn.click(
                fn=_open_outputs_from_texturing_tab,
                inputs=[tex_status_box],
                outputs=[tex_status_box],
                queue=False,
                show_progress="hidden",
            )

        # ---------------------------- Tab 3: UltraShape Refine ----------------------------
        with gr.Tab("UltraShape Refine"):
            with gr.Row():
                with gr.Column(scale=1, min_width=380):
                    us_mesh_file = gr.File(
                        label="Upload Coarse Mesh",
                        file_types=[".glb", ".gltf", ".obj", ".ply", ".stl"],
                        file_count="single",
                    )
                    us_image = gr.Image(
                        label="Reference Image",
                        format="png",
                        image_mode="RGBA",
                        type="pil",
                        height=400,
                    )
                    with gr.Row():
                        us_seed = gr.Slider(0, MAX_SEED, label="Seed", value=99, step=1, scale=4)
                        us_randomize_seed = gr.Checkbox(label="Randomize Seed", value=False, scale=1)
                    with gr.Row():
                        us_output_format = gr.Dropdown(
                            ["glb", "obj", "ply", "stl"],
                            label="Primary Output Format",
                            value="glb",
                        )
                        us_export_formats = gr.CheckboxGroup(
                            choices=["glb", "obj", "ply", "stl"],
                            value=["glb"],
                            label="Extra Export Formats",
                        )
                    with gr.Accordion("Advanced Settings", open=True):
                        with gr.Row():
                            us_checkpoint = gr.Textbox(
                                label="Checkpoint (optional)",
                                value="",
                                placeholder="Leave empty to auto-pick models/UltraShape/*.pt",
                            )
                            us_config_name = gr.Textbox(
                                label="Config",
                                value="infer_dit_refine.yaml",
                            )
                        with gr.Row():
                            us_dtype = gr.Dropdown(
                                ["float16", "bfloat16", "float32"],
                                label="DType",
                                value="bfloat16",
                                info="Precision mode for UltraShape inference. Lower precision usually reduces VRAM and can improve speed; float32 is typically heavier and slower.",
                            )
                            us_low_vram = gr.Checkbox(
                                label="Low VRAM",
                                value=True,
                                info="Enable CPU offload where supported. Lower VRAM, slower runtime.",
                            )
                            us_remove_bg = gr.Checkbox(
                                label="Remove BG (Reference Image)",
                                value=False,
                                info="Applies TRELLIS rembg to the guidance image. Adds an extra preprocessing pass (more runtime, small extra memory). Not required for the mesh input.",
                            )
                        with gr.Row():
                            us_steps = gr.Slider(
                                10, 200,
                                label="Steps",
                                value=50,
                                step=5,
                                info="More diffusion steps increase processing time. VRAM impact is usually modest compared with resolution/token controls.",
                            )
                            us_guidance_scale = gr.Slider(
                                1.0, 15.0,
                                label="Guidance",
                                value=5.0,
                                step=0.5,
                                info="Conditioning strength. Usually has minor VRAM/runtime impact versus steps and resolution.",
                            )
                            us_octree_resolution = gr.Slider(
                                256, 1024,
                                label="Octree Res",
                                value=384,
                                step=64,
                                info="Higher octree resolution increases VRAM usage and runtime during geometry decoding.",
                            )
                        with gr.Row():
                            us_num_chunks = gr.Slider(
                                1000, 50000,
                                label="Chunk Size",
                                value=8000,
                                step=1000,
                                info="Larger chunks are usually faster but need more VRAM; smaller chunks reduce VRAM pressure but run slower.",
                            )
                            us_target_face_count = gr.Slider(
                                100000, 2000000,
                                label="Target Faces",
                                value=500000,
                                step=10000,
                                info="Higher face targets increase mesh processing time and can raise memory use in later mesh operations.",
                            )
                            us_num_latents = gr.Slider(
                                0, 131072,
                                label="Num Latents (0=cfg)",
                                value=0,
                                step=1024,
                                info="More latent tokens increase VRAM usage and runtime during voxel conditioning/sampling.",
                            )
                        with gr.Row():
                            us_box_v = gr.Slider(
                                0.5, 2.0,
                                label="Box V",
                                value=1.0,
                                step=0.1,
                                info="Geometric control parameter. Usually negligible direct VRAM/runtime impact.",
                            )
                            us_mc_level = gr.Slider(
                                -0.1, 0.1,
                                label="MC Level",
                                value=0.0,
                                step=0.01,
                                info="Surface threshold parameter. Usually negligible direct VRAM/runtime impact.",
                            )
                            us_normalize_scale = gr.Slider(
                                0.5, 1.0,
                                label="Normalize Scale",
                                value=0.99,
                                step=0.01,
                                info="Input normalization control. Usually negligible direct VRAM/runtime impact.",
                            )
                        with gr.Row():
                            us_num_sharp_points = gr.Slider(
                                10000, 500000,
                                label="Sharp Points",
                                value=204800,
                                step=10000,
                                info="More sharp-edge samples improve detail capture but increase sampling/voxelization runtime and memory.",
                            )
                            us_num_uniform_points = gr.Slider(
                                10000, 500000,
                                label="Uniform Points",
                                value=204800,
                                step=10000,
                                info="More uniform samples improve coverage but increase sampling/voxelization runtime and memory.",
                            )

                with gr.Column(scale=2, min_width=520):
                    us_output = gr.Model3D(
                        label="Refined Mesh Preview",
                        height=724,
                        show_label=True,
                        display_mode="solid",
                        clear_color=(0.25, 0.25, 0.25, 1.0),
                        elem_id="ultrashape_refined_viewer",
                    )
                    us_status_box = gr.Textbox(
                        value="Upload a coarse mesh + reference image, then click Refine Mesh.",
                        lines=20,
                        max_lines=20,
                        interactive=False,
                        show_label=False,
                        container=False,
                        visible=False,
                    )
                    with gr.Row():
                        us_refine_btn = gr.Button("Refine Mesh", variant="primary")
                        us_download_btn = gr.DownloadButton(label="Download Refined Mesh", variant="primary")
                    with gr.Row():
                        us_cancel_btn = gr.Button("Cancel processing", variant="stop")
                        us_open_outputs_btn = gr.Button("Open outputs folder", variant="secondary")
                        us_view_logs_btn = gr.Button("View Logs", variant="secondary")

            us_logs_visible_state = gr.State(False)

            us_mesh_file.change(
                fn=_coerce_file_to_path,
                inputs=[us_mesh_file],
                outputs=[us_output],
                queue=False,
                show_progress="hidden",
            )

            def _us_toggle_logs(current_visible: bool) -> tuple:
                new_visible = not current_visible
                return (
                    gr.update(visible=new_visible),
                    new_visible,
                    gr.update(value="Hide Logs" if new_visible else "View Logs"),
                )

            us_view_logs_btn.click(
                fn=_us_toggle_logs,
                inputs=[us_logs_visible_state],
                outputs=[us_status_box, us_logs_visible_state, us_view_logs_btn],
                queue=False,
                show_progress="hidden",
            )

            def _us_cancel(current_status: str, req: gr.Request) -> str:
                ts = datetime.now().strftime("%H:%M:%S")
                msg = _cancel_now(_session_key(req), scope="all")
                return _append_status(current_status, f"[{ts}] {msg}")

            us_cancel_btn.click(
                fn=_us_cancel,
                inputs=[us_status_box],
                outputs=[us_status_box],
                queue=False,
                show_progress="hidden",
            )

            us_refine_btn.click(
                get_seed,
                inputs=[us_randomize_seed, us_seed],
                outputs=[us_seed],
            ).then(
                lambda: (gr.update(visible=True), True, gr.update(value="Hide Logs")),
                outputs=[us_status_box, us_logs_visible_state, us_view_logs_btn],
            ).then(
                ultrashape_refine_mesh,
                inputs=[
                    us_mesh_file,
                    us_image,
                    us_seed,
                    us_output_format,
                    us_export_formats,
                    us_checkpoint,
                    us_config_name,
                    us_dtype,
                    us_low_vram,
                    us_remove_bg,
                    us_steps,
                    us_guidance_scale,
                    us_octree_resolution,
                    us_num_chunks,
                    us_target_face_count,
                    us_num_latents,
                    us_box_v,
                    us_mc_level,
                    us_normalize_scale,
                    us_num_sharp_points,
                    us_num_uniform_points,
                    subprocess_mode,
                ],
                outputs=[us_output, us_download_btn, us_status_box],
            ).then(
                lambda: (gr.update(visible=False), False, gr.update(value="View Logs")),
                outputs=[us_status_box, us_logs_visible_state, us_view_logs_btn],
            )

            us_open_outputs_btn.click(
                fn=_open_outputs_from_texturing_tab,
                inputs=[us_status_box],
                outputs=[us_status_box],
                queue=False,
                show_progress="hidden",
            )

        # ---------------------------- Tab 4: View 3D Files ----------------------------
        with gr.Tab("View 3D Files"):
            gr.Markdown(
                "Drag & drop / upload a 3D file to preview it.\n\n"
                "**Supported:** `.glb`, `.gltf`, `.obj`, `.ply`, `.stl`"
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=380):
                    view3d_file = gr.File(
                        label="3D File",
                        file_types=[".glb", ".gltf", ".obj", ".ply", ".stl"],
                        file_count="single",
                    )
                    view3d_fullscreen_btn = gr.Button("Fullscreen", variant="secondary")
                with gr.Column(scale=2, min_width=520):
                    view3d_output = gr.Model3D(
                        label="3D Preview",
                        height=724,
                        show_label=True,
                        display_mode="solid",
                        clear_color=(0.25, 0.25, 0.25, 1.0),
                        elem_id="view3d_files_viewer",
                    )

            def _coerce_file_to_path(f):
                if f is None:
                    return None
                if isinstance(f, str):
                    return f
                if isinstance(f, dict):
                    return f.get("name") or f.get("path")
                return getattr(f, "name", None) or str(f)

            view3d_file.change(
                fn=_coerce_file_to_path,
                inputs=[view3d_file],
                outputs=[view3d_output],
                queue=False,
                show_progress="hidden",
            )

            view3d_fullscreen_btn.click(
                fn=None,
                inputs=[],
                outputs=None,
                js="""
() => {
  const root = document.querySelector("#view3d_files_viewer");
  const el = root?.querySelector('[data-testid="model3d"]') || root;
  if (!el) return;
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
  } else {
    el.requestFullscreen?.();
  }
}
""",
            )

        # ---------------------------- Tab 4: Help / Guide ----------------------------
        with gr.Tab("📘 Help / Settings Guide"):
            gr.Markdown(
                """
## Quick start (most people)

1. Go to **Image → 3D**.
2. Upload one or more images in **Upload Input Image (Or Multi-Angle)**.
   If you upload multiple, the first image is treated as the primary view.
3. Keep defaults, click **Generate**.
4. When preview is ready, click **Extract GLB**.
5. Your files are saved into `./outputs/<run_id>/` (for example `./outputs/0007/`).

If you want to stop a run:
- **Subprocess mode ON**: **Cancel processing** will stop everything and terminate the active worker stage immediately.
- **Subprocess mode OFF**: **Cancel processing** will stop **batch only** (in-process jobs can’t be force-killed safely).

---

## What the pipeline does (Image → 3D)

The Image → 3D pipeline is intentionally split into stages so progress can be shown and (in subprocess mode) VRAM can be released between stages:

1. **Preprocess image**: background removal + crop/center.  
   - Goal: give the model a clean, object-focused input.
2. **Encode conditioning**: compute image embeddings (512px and/or 1024px depending on resolution).
3. **Stage 1 — Sparse structure**: generate a sparse 3D structure (where the object exists in space).
4. **Stage 2 — Shape generation**: generate the high-detail geometry latent.
5. **Stage 3 — Material generation** (optional): generate texture/material latent (basecolor/roughness/metallic/opacity).
6. **Preview render**: render multi-view snapshots for the UI preview.
7. **Extract GLB**: convert the latent representation into a mesh and bake textures into a GLB (and optional extra formats).

---

## GLOBAL setting

### Subprocess stage processing (zero leftover VRAM between stages)
**What it is**: When enabled, each major stage runs in a fresh Python subprocess. This keeps the UI process from "holding onto" VRAM between stages.  
**When to enable**:
- Enable if you get CUDA OOM errors, driver resets, or your VRAM stays high after a run.
- Enable if you run large resolutions (1536/2048) or do batch processing.
**When to disable**:
- Disable if you prefer slightly simpler execution and you’re not memory constrained.
**Important**:
- With subprocess mode ON, the **Cancel processing** button can immediately terminate the worker stage.
- With subprocess mode OFF, in-process work can only stop at "safe points" (and we intentionally only cancel batch).

---

## IMAGE → 3D settings (left panel)

### Upload Input Image (Or Multi-Angle)
Upload the image or images you want to convert to 3D.

**Best practices**:
- Use a single main object. Avoid busy backgrounds.
- Center the object and keep it large in the frame.
- If you have a PNG with transparency, that’s ideal.

**Examples**:
- Good: a centered product photo on a plain background.
- Risky: multiple characters, cluttered scenery, tiny object in the distance.

### Multi-image behavior
You can upload one or multiple views of the same object (front/back/side).

**Behavior**:
- If you upload **1 image**, generation runs in single-image mode.
- If you upload **2 or more images**, generation runs in multi-image mode.
- The **first uploaded image** is treated as the primary view.
- `stochastic` mode is faster; `multidiffusion` is usually more consistent but slower.

### Resolution
Choose the target generation quality/speed level. Higher resolutions produce more detail but cost more VRAM/time.

**Options**:
- **512**: fastest and lightest. Great for quick tests and low‑VRAM GPUs.
- **1024**: good default balance (recommended starting point).
- **1536 / 2048**: highest detail, slowest, and most VRAM‑intensive.

**Example decision**:
- "I want fast previews": start with **512** or **1024**.
- "I need maximum detail": try **1536**, and only use **2048** if your GPU has enough VRAM and you can wait.

### Seed
Controls randomness. Same inputs + same seed + same settings = very similar output (useful for reproducibility).

**Example**:
- If you find a good result at seed `12345`, keep that seed to reproduce it later.

### Randomize Seed
If enabled, a new random seed is used each time you click **Generate** (or for each file in batch).

**Use it when**:
- You want to explore variations quickly.

**Turn it off when**:
- You want repeatable results and debugging.

### Decimation Target
Target triangle/face count for mesh simplification during **Extract GLB**.

**What it changes**:
- Lower target → smaller file, faster loading, but less geometric detail.
- Higher target → more detail, larger file, heavier rendering.

**Examples**:
- Game/real‑time: try `100k–300k`.
- DCC / offline: try `500k–1M` (default is high quality).

### Remesh Method
Controls how the surface is reconstructed before export.

**dual_contouring** (default):
- Fast, robust, works everywhere.

**faithful_contouring** (optional):
- Can preserve thin/open structures better, but needs extra dependencies (`faithcontour` + `atom3d`).
- If not installed, the UI hides it or auto-falls back.

### Simplify Method
Controls which mesh simplifier is used during export.

**cumesh**:
- GPU‑accelerated (when available), generally fast.

**meshlib**:
- CPU‑based alternative (requires optional deps), can behave differently on some meshes.

### Prune Invisible Faces
Attempts to remove faces that are not visible / not contributing (can reduce mesh size).

**Enable when**:
- You want smaller exports and cleaner geometry.

**Disable when**:
- You see holes or missing parts after extraction.

### Skip Texture Generation
If enabled, the model will generate **shape only** and skip material/texture generation.

**Why use it**:
- Faster generation
- Lower VRAM/time
- Useful for clay/geometry workflows

**Trade‑off**:
- Exported GLB won’t have rich PBR textures.

### Texture Size
Controls baked texture resolution during extraction (typical values: 1024 / 2048 / 4096).

**Examples**:
- 1024: lightweight, faster, good for previews.
- 2048: balanced quality/performance.
- 4096: default for maximum crispness (heavy VRAM/disk).

### Auto‑save export formats
Select which formats are written under `./outputs/<run_id>/08_final_exports/`.

**Notes**:
- `glb` is always produced for the viewer/download.
- Filenames now include stage meaning:
  - `glb_stage3_shape_only_####.glb`: intermediate mesh before optional re-texture.
  - `glb_final_textured_direct_####.glb`: final direct extract with TRELLIS texture bake.
  - `glb_final_textured_retexture_####.glb`: final output after deferred re-texture pass.
  - `glb_final_shape_only_####.glb`: final geometry-only output (no texture generation).
- Extra formats follow the same suffix, e.g. `obj_final_textured_retexture_####.obj`.
- `extract_artifacts.json` is written in `08_final_exports/` and marks which artifact is final + what it contains.
- If deferred re-texture is enabled, temporary stage files are stored in `09_retexture_work/` (not final deliverables).
- Extra formats (obj/ply/stl/gltf) are best-effort and may fail for some meshes; failures won’t block GLB export.

---

## Preview panel controls (right side)

### Generate
Runs the **Image → 3D** pipeline and builds the preview.

### Extract GLB
Converts the generated latents into an exportable mesh + textures (GLB) and saves to `./outputs/<run_id>/08_final_exports/`.
Also writes `extract_artifacts.json` that documents stage outputs, final artifact, and whether textures are included.

### View Extracted
Switches the UI to show the extracted GLB in the 3D viewer (no re‑compute).

### 📂 Open outputs folder
Opens the `./outputs` folder in your OS file explorer:
- Windows: File Explorer
- Linux: default file manager via `xdg-open` / `gio open`
- macOS: Finder via `open`

### 🛑 Cancel processing (two‑step safety)
This button uses a **two‑click confirmation** to avoid accidental cancels:

1. First click → arms cancellation (no work is stopped yet).
2. Second click within a few seconds → performs cancellation.

**Actual cancel behavior**:
- **If a subprocess stage is running**: cancels everything and terminates the active stage process immediately.
- **If subprocess mode is OFF and no subprocess stage is running**: cancels **batch processing only**.

---

## Batch Processing (accordion)

### Enable batch processing
Must be enabled to unlock **Run Batch**.

### Input folder (required)
Folder that contains images to process.

**Supported extensions**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

**Path examples**:
- Relative (recommended): `./my_images`
- Windows absolute: `D:\\datasets\\my_images`
- Linux absolute: `/home/user/my_images`

Tip: If your path contains spaces, you can wrap it in quotes.

### Output folder (optional)
Where batch results go.

- Leave blank to use `./outputs`
- Each input image is saved into its own subfolder named after the filename (safe‑sanitized).
- If a target folder already exists, that file is **skipped** (safe for resume).

### Run Batch
Processes each image using the **same settings** as a single run (resolution, guidance, extraction options, etc.).

**Seed behavior**:
- If **Randomize Seed** is ON → each image gets a different seed.
- If OFF → all images use the same seed (useful if you want consistent style).

---

## Advanced Settings (what "guidance" means)

These parameters control diffusion sampling behavior. Think of them as "how strongly the model follows its conditioning" and "how the sampler behaves over time".

### Guidance Strength
Higher values usually enforce the conditioning more strongly (often sharper/more literal), but too high can cause artifacts.

**Example tuning**:
- Too bland / not matching image: increase slightly (e.g. +0.5).
- Too many artifacts / distortions: reduce slightly.

### Guidance Rescale
Helps reduce over-saturation / over-contrast artifacts at higher guidance.

**Rule of thumb**:
- If you raise guidance strength a lot, consider raising rescale a bit too.

### Guidance Interval Start / End
Limits guidance to only part of the sampling trajectory (0 → start, 1 → end).

**Examples**:
- `start=0.6, end=1.0` means "apply stronger guidance mostly later".
- Narrower interval can reduce early over-constraint artifacts.

### Sampling Steps
More steps can improve quality but increases time.

**Examples**:
- Fast test: 8–12 steps
- Higher quality: 16–30 steps

### Rescale T
Sampler stability/temperature-like parameter used by this pipeline. Defaults are generally good.

### Max Number of Tokens
Mainly relevant for higher resolutions (cascade). It controls internal token budget / compute.

**If you see OOM at high resolution**:
- Reduce resolution first.
- Then reduce `max_num_tokens`.

---

## Texturing tab settings

### Upload Mesh
Upload an existing mesh (`.ply`, `.obj`, `.glb`, `.gltf`) to texture.

**Tip**: If you upload a scene file with multiple meshes, the app tries to convert it to a single mesh.

### Reference Image
Image that guides the texture appearance (color/material cues).

### Resolution (Texturing)
Controls which internal model path is used. Higher = more detail, more cost.

### Seed / Randomize Seed
Same meaning as Image → 3D: controls randomness and reproducibility.

### Texture Size (Texturing)
Baked texture resolution for the textured output GLB.

### Texturing Advanced Settings
Same concepts as "guidance" above but applied to texture generation.

---

## View 3D Files tab

Upload a 3D file to preview it locally. This does not run the ML pipeline.

---

## Config Presets (Save / Load)

Presets save **all settings** from Image->3D, Texturing, UltraShape Refine, and Rigging tabs, but do **not** include uploaded images/files.

**Where presets are stored**: `./presets/<name>.json`

**Built-in presets**:
- `best`: the tested default path
- `low_vram`: lower-memory variant with chunked/tiled memory-saving options enabled

**Typical workflow**:
- Dial in settings you like → Save preset as `my_high_quality`
- Later → Load preset to restore all sliders/checkboxes instantly
"""
            )

        # ---------------------------- Tab 5: Rigging (UniRig) ----------------------------
        with gr.Tab("🦴 Rigging", id="rigging_tab"):
            rigging_ui = rigging_tab(
                run_skeleton_fn=_run_unirig_skeleton,
                run_skinning_fn=_run_unirig_skinning,
                run_merge_fn=_run_unirig_merge,
                rigging_outputs_dir=OUTPUTS_DIR,
                open_folder_fn=_open_folder,
            )

        # ---------------------------- Tab 6: Animation Player ----------------------------
        with gr.Tab("🎬 Animation Player", id="animation_tab"):
            animation_ui = animation_player_tab(
                list_models_fn=_list_rigged_models,
                rigging_outputs_dir=OUTPUTS_DIR,
                open_folder_fn=_open_folder,
                generate_animation_preview_fn=_regenerate_animation_preview_for_browser,
            )

    # Bridge: Rigging -> Animation browser selection.
    rigging_ui["send_to_animation_btn"].click(
        fn=_send_rig_output_to_animation,
        inputs=[
            rigging_ui["final_output_state"],
            rigging_ui["skinned_path_state"],
            rigging_ui["skeleton_path_state"],
            rigging_ui["rig_status"],
        ],
        outputs=[
            animation_ui["external_select_input"],
            rigging_ui["rig_status"],
            main_tabs,
        ],
        queue=False,
        show_progress="hidden",
    )

    # ---------------------------- Preset Wiring ----------------------------
    _CONFIG_KEYS = [
        ("global", "subprocess_mode"),
        ("image_to_3d", "resolution"),
        ("image_to_3d", "pipeline_strategy"),
        ("image_to_3d", "seed"),
        ("image_to_3d", "randomize_seed"),
        ("image_to_3d", "decimation_target"),
        ("image_to_3d", "remesh_method"),
        ("image_to_3d", "fill_holes_max_perimeter"),
        ("image_to_3d", "repair_method"),
        ("image_to_3d", "simplify_method"),
        ("image_to_3d", "prune_invisible_faces"),
        ("image_to_3d", "merge_vertices_dist"),
        ("image_to_3d", "shade_smooth"),
        ("image_to_3d", "shade_smooth_angle"),
        ("image_to_3d", "force_double_sided"),
        ("image_to_3d", "no_pbr_export"),
        ("image_to_3d", "no_texture_gen"),
        ("image_to_3d", "deferred_texture_after_cleanup"),
        ("image_to_3d", "projection_texture_refine"),
        ("image_to_3d", "projection_view_azimuths"),
        ("image_to_3d", "projection_view_elevations"),
        ("image_to_3d", "projection_blend_exponent"),
        ("image_to_3d", "projection_ortho_scale"),
        ("image_to_3d", "projection_fill_holes"),
        ("image_to_3d", "projection_max_hole_size"),
        ("image_to_3d", "texture_size"),
        ("image_to_3d", "export_formats"),
        ("image_to_3d", "ss_guidance_strength"),
        ("image_to_3d", "ss_guidance_rescale"),
        ("image_to_3d", "ss_guidance_interval_start"),
        ("image_to_3d", "ss_guidance_interval_end"),
        ("image_to_3d", "ss_sampling_steps"),
        ("image_to_3d", "ss_rescale_t"),
        ("image_to_3d", "force_high_res_conditional"),
        ("image_to_3d", "low_vram"),
        ("image_to_3d", "model_variant"),
        ("image_to_3d", "attention_backend"),
        ("image_to_3d", "sampler_type"),
        ("image_to_3d", "use_chunked_processing"),
        ("image_to_3d", "use_tiled_extraction"),
        ("image_to_3d", "extract_use_chunked_processing"),
        ("image_to_3d", "extract_use_tiled_extraction"),
        ("image_to_3d", "shape_slat_guidance_strength"),
        ("image_to_3d", "shape_slat_guidance_rescale"),
        ("image_to_3d", "shape_slat_guidance_interval_start"),
        ("image_to_3d", "shape_slat_guidance_interval_end"),
        ("image_to_3d", "shape_slat_sampling_steps"),
        ("image_to_3d", "shape_slat_rescale_t"),
        ("image_to_3d", "max_num_tokens"),
        ("image_to_3d", "multiview_mode"),
        ("image_to_3d", "tex_slat_guidance_strength"),
        ("image_to_3d", "tex_slat_guidance_rescale"),
        ("image_to_3d", "tex_slat_guidance_interval_start"),
        ("image_to_3d", "tex_slat_guidance_interval_end"),
        ("image_to_3d", "tex_slat_sampling_steps"),
        ("image_to_3d", "tex_slat_rescale_t"),
        ("image_to_3d", "ultrashape_enabled"),
        ("image_to_3d", "ultrashape_retexture_after_refine"),
        ("image_to_3d", "ultrashape_conservative_mode"),
        ("image_to_3d", "ultrashape_checkpoint"),
        ("image_to_3d", "ultrashape_config_name"),
        ("image_to_3d", "ultrashape_dtype"),
        ("image_to_3d", "ultrashape_low_vram"),
        ("image_to_3d", "ultrashape_remove_bg"),
        ("image_to_3d", "ultrashape_steps"),
        ("image_to_3d", "ultrashape_guidance_scale"),
        ("image_to_3d", "ultrashape_octree_resolution"),
        ("image_to_3d", "ultrashape_num_chunks"),
        ("image_to_3d", "ultrashape_target_face_count"),
        ("image_to_3d", "ultrashape_num_latents"),
        ("image_to_3d", "ultrashape_box_v"),
        ("image_to_3d", "ultrashape_mc_level"),
        ("image_to_3d", "ultrashape_normalize_scale"),
        ("image_to_3d", "ultrashape_num_sharp_points"),
        ("image_to_3d", "ultrashape_num_uniform_points"),
        ("texturing", "resolution"),
        ("texturing", "seed"),
        ("texturing", "randomize_seed"),
        ("texturing", "texture_size"),
        ("texturing", "guidance_strength"),
        ("texturing", "guidance_rescale"),
        ("texturing", "guidance_interval_start"),
        ("texturing", "guidance_interval_end"),
        ("texturing", "sampling_steps"),
        ("texturing", "rescale_t"),
        ("texturing", "low_vram"),
        ("texturing", "attention_backend"),
        ("texturing", "sampler_type"),
        ("ultrashape_refine", "seed"),
        ("ultrashape_refine", "randomize_seed"),
        ("ultrashape_refine", "output_format"),
        ("ultrashape_refine", "export_formats"),
        ("ultrashape_refine", "checkpoint"),
        ("ultrashape_refine", "config_name"),
        ("ultrashape_refine", "dtype"),
        ("ultrashape_refine", "low_vram"),
        ("ultrashape_refine", "remove_bg"),
        ("ultrashape_refine", "steps"),
        ("ultrashape_refine", "guidance_scale"),
        ("ultrashape_refine", "octree_resolution"),
        ("ultrashape_refine", "num_chunks"),
        ("ultrashape_refine", "target_face_count"),
        ("ultrashape_refine", "num_latents"),
        ("ultrashape_refine", "box_v"),
        ("ultrashape_refine", "mc_level"),
        ("ultrashape_refine", "normalize_scale"),
        ("ultrashape_refine", "num_sharp_points"),
        ("ultrashape_refine", "num_uniform_points"),
        ("rigging", "seed"),
        ("rigging", "randomize_seed"),
        ("rigging", "export_format"),
        ("rigging", "export_both_formats"),
        ("rigging", "enable_skinning"),
        ("rigging", "auto_merge"),
    ]

    _CONFIG_COMPONENTS = [
        subprocess_mode,
        resolution,
        pipeline_strategy,
        seed,
        randomize_seed,
        decimation_target,
        remesh_method,
        fill_holes_max_perimeter,
        repair_method,
        simplify_method,
        prune_invisible_faces,
        merge_vertices_dist,
        shade_smooth,
        shade_smooth_angle,
        force_double_sided,
        no_pbr_export,
        no_texture_gen,
        deferred_texture_after_cleanup,
        projection_texture_refine,
        projection_view_azimuths,
        projection_view_elevations,
        projection_blend_exponent,
        projection_ortho_scale,
        projection_fill_holes,
        projection_max_hole_size,
        texture_size,
        export_formats,
        ss_guidance_strength,
        ss_guidance_rescale,
        ss_guidance_interval_start,
        ss_guidance_interval_end,
        ss_sampling_steps,
        ss_rescale_t,
        force_high_res_conditional,
        low_vram,
        model_variant,
        attention_backend,
        sampler_type,
        use_chunked_processing,
        use_tiled_extraction,
        extract_use_chunked_processing,
        extract_use_tiled_extraction,
        shape_slat_guidance_strength,
        shape_slat_guidance_rescale,
        shape_slat_guidance_interval_start,
        shape_slat_guidance_interval_end,
        shape_slat_sampling_steps,
        shape_slat_rescale_t,
        max_num_tokens,
        multiview_mode,
        tex_slat_guidance_strength,
        tex_slat_guidance_rescale,
        tex_slat_guidance_interval_start,
        tex_slat_guidance_interval_end,
        tex_slat_sampling_steps,
        tex_slat_rescale_t,
        ultrashape_enabled,
        ultrashape_retexture_after_refine,
        ultrashape_conservative_mode,
        ultrashape_checkpoint,
        ultrashape_config_name,
        ultrashape_dtype,
        ultrashape_low_vram,
        ultrashape_remove_bg,
        ultrashape_steps,
        ultrashape_guidance_scale,
        ultrashape_octree_resolution,
        ultrashape_num_chunks,
        ultrashape_target_face_count,
        ultrashape_num_latents,
        ultrashape_box_v,
        ultrashape_mc_level,
        ultrashape_normalize_scale,
        ultrashape_num_sharp_points,
        ultrashape_num_uniform_points,
        tex_resolution,
        tex_seed,
        tex_randomize_seed,
        tex_texture_size,
        t_guidance_strength,
        t_guidance_rescale,
        t_guidance_interval_start,
        t_guidance_interval_end,
        t_sampling_steps,
        t_rescale_t,
        tex_low_vram,
        tex_attention_backend,
        tex_sampler_type,
        us_seed,
        us_randomize_seed,
        us_output_format,
        us_export_formats,
        us_checkpoint,
        us_config_name,
        us_dtype,
        us_low_vram,
        us_remove_bg,
        us_steps,
        us_guidance_scale,
        us_octree_resolution,
        us_num_chunks,
        us_target_face_count,
        us_num_latents,
        us_box_v,
        us_mc_level,
        us_normalize_scale,
        us_num_sharp_points,
        us_num_uniform_points,
        rigging_ui["seed"],
        rigging_ui["randomize_seed"],
        rigging_ui["export_format"],
        rigging_ui["export_both_formats"],
        rigging_ui["enable_skinning"],
        rigging_ui["auto_merge"],
    ]

    def _values_to_ui_config(*values) -> dict:
        cfg = _default_ui_config()
        for (section, key), val in zip(_CONFIG_KEYS, values):
            cfg[section][key] = val
        return cfg

    def _ui_config_to_values(cfg: dict) -> List[Any]:
        merged = _merge_ui_config(cfg)

        # Light validation/clamping for list-like and choice-like inputs.
        # If a key is invalid, fall back to defaults (keeps UI consistent).
        defaults = _default_ui_config()

        # Image→3D resolution
        if merged["image_to_3d"]["resolution"] not in ["512", "768", "1024", "1280", "1536", "2048"]:
            merged["image_to_3d"]["resolution"] = defaults["image_to_3d"]["resolution"]
        if merged["image_to_3d"].get("pipeline_strategy") not in PIPELINE_STRATEGY_CHOICES:
            merged["image_to_3d"]["pipeline_strategy"] = defaults["image_to_3d"]["pipeline_strategy"]
        # Texturing resolution
        if merged["texturing"]["resolution"] not in ["512", "1024", "1536"]:
            merged["texturing"]["resolution"] = defaults["texturing"]["resolution"]
        # Remesh method (depends on env)
        if merged["image_to_3d"]["remesh_method"] not in REMESH_METHOD_CHOICES:
            merged["image_to_3d"]["remesh_method"] = defaults["image_to_3d"]["remesh_method"]
        # Simplify method
        if merged["image_to_3d"]["simplify_method"] not in SIMPLIFY_METHOD_CHOICES:
            merged["image_to_3d"]["simplify_method"] = defaults["image_to_3d"]["simplify_method"]
        if merged["image_to_3d"].get("repair_method") not in REPAIR_METHOD_CHOICES:
            merged["image_to_3d"]["repair_method"] = defaults["image_to_3d"]["repair_method"]
        # Export formats
        ef = merged["image_to_3d"].get("export_formats")
        if not isinstance(ef, list):
            ef = defaults["image_to_3d"]["export_formats"]
        ef = [str(x) for x in ef if str(x) in {"glb", "gltf", "obj", "ply", "stl"}]
        merged["image_to_3d"]["export_formats"] = ef or defaults["image_to_3d"]["export_formats"]

        if merged["image_to_3d"].get("ultrashape_dtype") not in {"float16", "bfloat16", "float32"}:
            merged["image_to_3d"]["ultrashape_dtype"] = defaults["image_to_3d"]["ultrashape_dtype"]
        if merged["image_to_3d"].get("multiview_mode") not in {"stochastic", "multidiffusion"}:
            merged["image_to_3d"]["multiview_mode"] = defaults["image_to_3d"]["multiview_mode"]
        if merged["image_to_3d"].get("model_variant") not in MODEL_VARIANT_CHOICES:
            merged["image_to_3d"]["model_variant"] = defaults["image_to_3d"]["model_variant"]
        if merged["image_to_3d"].get("attention_backend") not in ATTENTION_BACKEND_CHOICES:
            merged["image_to_3d"]["attention_backend"] = defaults["image_to_3d"]["attention_backend"]
        if merged["image_to_3d"].get("sampler_type") not in SAMPLER_TYPE_CHOICES:
            merged["image_to_3d"]["sampler_type"] = defaults["image_to_3d"]["sampler_type"]
        if merged["texturing"].get("attention_backend") not in ATTENTION_BACKEND_CHOICES:
            merged["texturing"]["attention_backend"] = defaults["texturing"]["attention_backend"]
        if merged["texturing"].get("sampler_type") not in SAMPLER_TYPE_CHOICES:
            merged["texturing"]["sampler_type"] = defaults["texturing"]["sampler_type"]

        # UltraShape tab output format / export formats / dtype
        if merged["ultrashape_refine"].get("output_format") not in {"glb", "obj", "ply", "stl"}:
            merged["ultrashape_refine"]["output_format"] = defaults["ultrashape_refine"]["output_format"]
        uef = merged["ultrashape_refine"].get("export_formats")
        if not isinstance(uef, list):
            uef = defaults["ultrashape_refine"]["export_formats"]
        uef = [str(x) for x in uef if str(x) in {"glb", "obj", "ply", "stl"}]
        merged["ultrashape_refine"]["export_formats"] = uef or defaults["ultrashape_refine"]["export_formats"]
        if merged["ultrashape_refine"].get("dtype") not in {"float16", "bfloat16", "float32"}:
            merged["ultrashape_refine"]["dtype"] = defaults["ultrashape_refine"]["dtype"]
        if merged["rigging"].get("export_format") not in {"fbx", "glb"}:
            merged["rigging"]["export_format"] = defaults["rigging"]["export_format"]

        def _clamp01(v, d):
            try:
                v = float(v)
            except Exception:
                v = float(d)
            return max(0.0, min(1.0, v))

        def _coerce_float(v, d, min_value: Optional[float] = None):
            try:
                v = float(v)
            except Exception:
                v = float(d)
            if min_value is not None:
                v = max(min_value, v)
            return v

        def _coerce_int(v, d, min_value: Optional[int] = None):
            try:
                v = int(v)
            except Exception:
                v = int(d)
            if min_value is not None:
                v = max(min_value, v)
            return v

        def _fix_interval(section: str, start_key: str, end_key: str) -> None:
            s = _clamp01(merged[section].get(start_key), defaults[section][start_key])
            e = _clamp01(merged[section].get(end_key), defaults[section][end_key])
            if s > e:
                s, e = e, s
            merged[section][start_key] = s
            merged[section][end_key] = e

        _fix_interval("image_to_3d", "ss_guidance_interval_start", "ss_guidance_interval_end")
        _fix_interval("image_to_3d", "shape_slat_guidance_interval_start", "shape_slat_guidance_interval_end")
        _fix_interval("image_to_3d", "tex_slat_guidance_interval_start", "tex_slat_guidance_interval_end")
        _fix_interval("texturing", "guidance_interval_start", "guidance_interval_end")
        merged["image_to_3d"]["projection_blend_exponent"] = _coerce_float(
            merged["image_to_3d"].get("projection_blend_exponent"),
            defaults["image_to_3d"]["projection_blend_exponent"],
            0.1,
        )
        merged["image_to_3d"]["projection_ortho_scale"] = _coerce_float(
            merged["image_to_3d"].get("projection_ortho_scale"),
            defaults["image_to_3d"]["projection_ortho_scale"],
            0.05,
        )
        merged["image_to_3d"]["projection_max_hole_size"] = _coerce_int(
            merged["image_to_3d"].get("projection_max_hole_size"),
            defaults["image_to_3d"]["projection_max_hole_size"],
            0,
        )

        return [merged[s][k] for (s, k) in _CONFIG_KEYS]

    def _save_preset_ui(preset_name: str, *values):
        try:
            cfg = _values_to_ui_config(*values)
            saved = _save_ui_preset(preset_name, cfg)
            presets = _list_ui_presets()
            return (
                gr.update(choices=presets, value=saved),
                f"✅ Saved preset **{saved}**",
            )
        except Exception as e:
            return gr.update(), f"[ERROR] Save failed: {e}"

    def _load_preset_ui(preset_name: str):
        if not preset_name:
            cfg = _default_ui_config()
            vals = _ui_config_to_values(cfg)
            return (*vals, "INFO: No preset selected (showing defaults).")

        cfg = _load_ui_preset(preset_name)
        if not cfg:
            cfg = _default_ui_config()
            vals = _ui_config_to_values(cfg)
            return (*vals, f"WARNING: Preset **{preset_name}** not found (loaded defaults).")

        vals = _ui_config_to_values(cfg)
        return (*vals, f"✅ Loaded preset **{preset_name}**")

    def _reset_defaults_ui():
        cfg = _default_ui_config()
        vals = _ui_config_to_values(cfg)
        return (*vals, "✅ Reset to defaults")

    def _delete_preset_ui(preset_name: str):
        if not preset_name:
            return gr.update(), "WARNING: No preset selected"
        if str(preset_name).strip() in _builtin_ui_presets():
            return gr.update(choices=_list_ui_presets(), value=preset_name), f"INFO: Built-in preset **{preset_name}** cannot be deleted"
        ok = _delete_ui_preset(preset_name)
        presets = _list_ui_presets()
        if ok:
            return gr.update(choices=presets, value=""), f"✅ Deleted preset **{preset_name}**"
        return gr.update(choices=presets), f"WARNING: Could not delete preset **{preset_name}**"

    ui_preset_save_btn.click(
        fn=_save_preset_ui,
        inputs=[ui_preset_name] + _CONFIG_COMPONENTS,
        outputs=[ui_preset_dropdown, ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_load_btn.click(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown],
        outputs=_CONFIG_COMPONENTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_dropdown.change(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown],
        outputs=_CONFIG_COMPONENTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_reset_btn.click(
        fn=_reset_defaults_ui,
        inputs=[],
        outputs=_CONFIG_COMPONENTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_delete_btn.click(
        fn=_delete_preset_ui,
        inputs=[ui_preset_dropdown],
        outputs=[ui_preset_dropdown, ui_preset_status],
        queue=False,
        show_progress="hidden",
    )


def _parse_launch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRELLIS.2 Premium (Gradio)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Optional bind address. If omitted, Gradio default host is used.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional port. If omitted, Gradio chooses its default behavior.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the app in a browser (inbrowser is ON by default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PRESETS_DIR, exist_ok=True)
    demo.queue()
    args = _parse_launch_args()
    launch_kwargs = {
        "share": args.share,
        "inbrowser": not args.no_browser,
        "show_error": True,
    }
    if args.host is not None:
        launch_kwargs["server_name"] = args.host
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    try:
        launch_sig = inspect.signature(demo.launch)
        for k, v in (("theme", APP_THEME), ("css", css), ("head", head), ("favicon_path", FAVICON_PATH)):
            if k in launch_sig.parameters:
                launch_kwargs[k] = v
        if "allowed_paths" in launch_sig.parameters:
            launch_kwargs["allowed_paths"] = _discover_allowed_paths_all_drives()
            print(
                f"[launch] Gradio allowed_paths enabled with {len(launch_kwargs['allowed_paths'])} path roots.",
                flush=True,
            )
    except Exception:
        pass
    demo.launch(**launch_kwargs)
