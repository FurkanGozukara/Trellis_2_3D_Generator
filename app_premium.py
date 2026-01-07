import gradio as gr

import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import subprocess
import signal
import importlib
import json
import time
import threading
from pathlib import Path

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
from typing import Tuple, Optional, Dict, List, Any

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


def _is_faithful_contouring_available() -> bool:
    """
    `faithful_contouring` remeshing in `o_voxel.postprocess.to_glb()` depends on optional
    FaithC packages (`faithcontour` + `atom3d`). These are not installed by default in
    many environments (especially Windows).
    """
    try:
        return (
            importlib.util.find_spec("faithcontour") is not None
            and importlib.util.find_spec("atom3d") is not None
        )
    except Exception:
        return False


REMESH_METHOD_CHOICES = ["dual_contouring"]
if _is_faithful_contouring_available():
    REMESH_METHOD_CHOICES.append("faithful_contouring")


# ------------------------------- Paths / Config ------------------------------

MODELS_DIR = os.path.join(APP_DIR, "models")
TMP_DIR = os.path.join(APP_DIR, "tmp")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
PRESETS_DIR = os.path.join(APP_DIR, "presets")
SUBPROCESS_STAGE_SCRIPT = os.path.join(APP_DIR, "subprocess_stage.py")

# Ensure TRELLIS_MODELS_DIR is set (trellis2 code also falls back to ../models).
os.environ.setdefault("TRELLIS_MODELS_DIR", MODELS_DIR)

# Local helpers (not the stdlib `subprocess` module)
from subprocess_utils import allocate_run_dir, next_indexed_path, ensure_dir, safe_relpath  # noqa: E402

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
    root = Path(PRESETS_DIR)
    if not root.exists():
        return []
    return sorted([p.stem for p in root.glob("*.json") if p.is_file()])


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
        return name or None
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
            # Keep deterministic defaults (users can enable randomize for exploration).
            "seed": 99,
            "randomize_seed": False,
            "decimation_target": 1000000,
            "remesh_method": "dual_contouring",
            "simplify_method": "cumesh",
            "prune_invisible_faces": False,
            "no_texture_gen": False,
            "texture_size": 2048,
            "export_formats": ["glb"],
            "low_vram": False,  # Keep models in VRAM for best quality and speed
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
            "max_num_tokens": 49152,  # Restored to original for quality (was 32768)
            "tex_slat_guidance_strength": 1.0,
            "tex_slat_guidance_rescale": 0.0,
            "tex_slat_guidance_interval_start": 0.6,  # Model default: CFG in middle 30% range
            "tex_slat_guidance_interval_end": 0.9,
            "tex_slat_sampling_steps": 12,
            "tex_slat_rescale_t": 3.0,
        },
        "texturing": {
            "resolution": "1024",
            "seed": 99,
            "randomize_seed": False,
            "texture_size": 2048,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "guidance_interval_start": 0.6,
            "guidance_interval_end": 0.9,
            "sampling_steps": 12,
            "rescale_t": 3.0,
        },
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

    for section in ("global", "image_to_3d", "texturing"):
        section_data = cfg.get(section)
        if isinstance(section_data, dict):
            base[section].update(section_data)

    return base


def _save_ui_preset(preset_name: str, config: dict) -> str:
    if not preset_name or not str(preset_name).strip():
        raise ValueError("Preset name cannot be empty.")

    safe_name = _sanitize_preset_name(str(preset_name).strip())
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
    path = _ui_preset_path(preset_name)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


def _append_status(current: str, msg: str) -> str:
    current = current or ""
    msg = msg or ""
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
            lines = ["… (truncated) …"] + lines[-max_lines:]
            status = "\n".join(lines)

    if max_chars and len(status) > max_chars:
        status = "… (truncated) …\n" + status[-max_chars:]

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

    cmd = [
        sys.executable,
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
                lf.write(line)
                lf.flush()
                yield {"type": "log", "text": line.rstrip("\n")}
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


def get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        _image_pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        _image_pipeline.low_vram = False  # Keep models in VRAM for best quality and speed
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
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_guidance_interval_start: float,
    ss_guidance_interval_end: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    force_high_res_conditional: bool,
    low_vram: bool,
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
    max_num_tokens: int,
    decimation_target: int,
    texture_size: int,
    remesh_method: str,
    simplify_method: str,
    prune_invisible_faces: bool,
    export_formats: List[str],
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
            log_lines.append(f"[{ts}] {line}")
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
                    ss_guidance_strength,
                    ss_guidance_rescale,
                    ss_guidance_interval_start,
                    ss_guidance_interval_end,
                    ss_sampling_steps,
                    ss_rescale_t,
                    force_high_res_conditional,
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
                    max_num_tokens,
                    subprocess_mode,
                    req=req,
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
                    str(simplify_method),
                    bool(no_texture_gen),
                    bool(prune_invisible_faces),
                    export_formats,
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

def _get_pipeline_type(resolution_str: str) -> tuple[str, int]:
    """
    Convert resolution string to pipeline type and target resolution.
    
    Returns:
        (pipeline_type, target_resolution)
        
    Examples:
        "512" -> ("512", 512)
        "768" -> ("768_cascade", 768)
        "1024" -> ("1024_cascade", 1024)
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
        # IMPORTANT: Use the cascade pipeline at 1024 to match the reference app (trellis_org2),
        # which generally yields higher-fidelity shapes/textures than the direct 1024 path.
        return "1024_cascade", 1024
    else:
        # Any other resolution uses cascade
        return f"{res}_cascade", res


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    custom_resolution: int,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_guidance_interval_start: float,
    ss_guidance_interval_end: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    force_high_res_conditional: bool,
    low_vram: bool,
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
    max_num_tokens: int,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[dict], str, str]:
    # Stream step-by-step status so users aren't "in the dark" during long runs.
    status = ""
    session = _session_key(req)

    # Mutable container for log file path (set after run_dir is allocated)
    _log_file_path: List[Optional[Path]] = [None]

    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
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


    if image is None:
        raise gr.Error("Please provide an image (upload or pick an example).")

    # Handle custom resolution override
    if custom_resolution and custom_resolution > 0:
        resolution = str(custom_resolution)
        _log(f"Using custom resolution: {resolution}", 0.0)
    
    # Validate and get pipeline type
    try:
        pipeline_type, target_res = _get_pipeline_type(resolution)
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
        image.save(str(input_path))
    except Exception:
        # Don't fail the run just because saving failed; continue.
        pass

    _log(f"Starting Image → 3D generation (resolution: {resolution}, pipeline: {pipeline_type})…", 0.0)
    yield None, empty_html, gr.update(value=_trim_status(status), visible=True)

    if subprocess_mode:
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
                "pipeline_type": pipeline_type,
                "no_texture_gen": bool(no_texture_gen),
                "max_num_tokens": int(max_num_tokens),
                "force_high_res_conditional": bool(force_high_res_conditional),
                "low_vram": bool(low_vram),
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

        # Stage: preprocess image (writes 01_preprocessed.png)
        preprocess_payload = {
            "model_repo": "microsoft/TRELLIS.2-4B",
            "input_image_path": str(input_path),
            "output_image_path": str(preprocessed_path),
        }
        try:
            _ = yield from _stage("preprocess_image", preprocess_payload, 0.05)
        except UserCancelled:
            yield from _cancelled_exit()
            return

        # Stage: encode conditioning
        cond_payload = {
            "model_repo": "microsoft/TRELLIS.2-4B",
            "image_path": str(preprocessed_path),
            "resolution": resolution,
            "cond_512_path": str(cond_512_path),
            "cond_1024_path": str(cond_1024_path) if cond_1024_path is not None else None,
            "force_high_res_conditional": bool(force_high_res_conditional),
        }
        try:
            cond_result = yield from _stage("encode_cond", cond_payload, 0.08)
        except UserCancelled:
            yield from _cancelled_exit()
            return

        # Track RNG across subprocess stages so results match the single-process reference pipeline
        # for a given seed (instead of re-seeding each stage and changing the noise sequence).
        rng_after_sparse_path = run_dir / "04_rng_after_sparse.pt"
        rng_after_shape_path = run_dir / "05_rng_after_shape.pt"

        # Stage: sparse structure
        sparse_payload = {
            "model_repo": "microsoft/TRELLIS.2-4B",
            "seed": int(seed),
            "resolution": resolution,
            "cond_512_path": str(cond_512_path),
            "coords_path": str(coords_path),
            "force_high_res_conditional": bool(force_high_res_conditional),
            "rng_state_out_path": str(rng_after_sparse_path),
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
            "model_repo": "microsoft/TRELLIS.2-4B",
            "seed": int(seed),
            "resolution": resolution,
            "cond_512_path": str(cond_512_path),
            "cond_1024_path": str(cond_1024_path) if cond_1024_path is not None else None,
            "coords_path": str(coords_path),
            "shape_slat_path": str(shape_slat_path),
            "out_res_path": str(shape_res_path),
            "rng_state_in_path": str(rng_after_sparse_path),
            "rng_state_out_path": str(rng_after_shape_path),
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
                "model_repo": "microsoft/TRELLIS.2-4B",
                "seed": int(seed),
                "resolution": resolution,
                "cond_path": tex_cond_path,
                "shape_slat_path": str(shape_slat_path),
                "tex_slat_path": str(tex_slat_path),
                "rng_state_in_path": str(rng_after_shape_path),
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
        preview_payload = {
            "model_repo": "microsoft/TRELLIS.2-4B",
            "shape_slat_path": str(shape_slat_path),
            "tex_slat_path": str(tex_slat_path) if tex_slat_path is not None else None,
            "res": int(res),
            "preview_dir": str(preview_dir),
            "preview_manifest_path": str(preview_manifest_path),
            "use_chunked_processing": bool(use_chunked_processing),
            "use_tiled_extraction": bool(use_tiled_extraction),
        }
        preview_failed = False
        preview_error_msg = ""
        try:
            _ = yield from _stage("render_preview", preview_payload, 0.82)
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
                _log(f"⚠️ Preview rendering failed: {preview_error_msg}", 0.85)
                _log("Latents saved successfully. You can still extract GLB!", 0.86)
            else:
                # No latents, re-raise the error
                raise

        # Handle preview failure - return minimal state that allows GLB extraction
        if preview_failed:
            state = {
                "_mode": "subprocess",
                "_run_id": run_id,
                "_run_dir": str(run_dir),
                "_pipeline_type": pipeline_type,
                "res": int(res),
                "shape_slat_path": str(shape_slat_path),
                "tex_slat_path": str(tex_slat_path) if tex_slat_path is not None else None,
                "preview_manifest_path": None,  # No preview available
                "_preview_failed": True,
                "_preview_error": preview_error_msg,
            }
            # Create an error HTML that shows the logs and allows extraction
            error_html = f"""
            <div class="previewer-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; padding: 20px;">
                <div style="background: rgba(255, 100, 100, 0.15); border: 1px solid rgba(255, 100, 100, 0.4); border-radius: 8px; padding: 20px; max-width: 600px; text-align: center;">
                    <h3 style="color: #ff6b6b; margin: 0 0 10px 0;">⚠️ Preview Rendering Failed</h3>
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
        preview_html_path.write_text(full_html, encoding="utf-8")

        state = {
            "_mode": "subprocess",
            "_run_id": run_id,
            "_run_dir": str(run_dir),
            "_pipeline_type": pipeline_type,
            "res": int(res),
            "shape_slat_path": str(shape_slat_path),
            "tex_slat_path": str(tex_slat_path) if tex_slat_path is not None else None,
            "preview_manifest_path": str(preview_manifest_path),
        }
        _log("Done. You can now click “Extract GLB”.", 1.0)
        # Hide the overlay once preview is ready so users can see the render.
        yield state, full_html, gr.update(value=_trim_status(status), visible=False)
        return

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
    _log("Preprocessing image (background removal / crop)…", 0.06)
    image = pipe.preprocess_image(image)
    try:
        image.save(str(preprocessed_path))
    except Exception:
        pass

    # Run the pipeline step-by-step so we can update status between stages.
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

    # Sparse structure resolution: 32 for most cases, 64 for direct 1024 sampling
    ss_res = 64 if pipeline_type == "1024" else 32
    _log("Stage 1/3: Sampling sparse structure…", 0.18)
    coords = pipe.sample_sparse_structure(cond_512, ss_res, 1, ss_params)
    try:
        torch.save(coords.detach().cpu(), str(coords_path))
    except Exception:
        pass
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
        state["_pipeline_type"] = pipeline_type
        state["shape_slat_path"] = str(shape_slat_path)
        state["tex_slat_path"] = str(tex_slat_path) if tex_slat_path is not None else None
        state["preview_manifest_path"] = str(preview_manifest_path)
    except Exception:
        pass

    _log("Done. You can now click “Extract GLB”.", 1.0)
    # Hide the overlay once preview is ready so users can see the render.
    yield state, full_html, gr.update(value=_trim_status(status), visible=False)


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    remesh_method: str,
    simplify_method: str,
    no_texture_gen: bool,
    prune_invisible_faces: bool,
    export_formats: List[str],
    extract_use_chunked_processing: bool,
    extract_use_tiled_extraction: bool,
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    if state is None:
        # This happens when users click "Extract GLB" before clicking "Generate"
        # (or right after changing the image / clicking an example).
        raise gr.Error("Nothing to extract yet. Click **Generate** first.")

    session = _session_key(req)

    # If the run was generated in subprocess mode (or the checkbox is enabled),
    # do extraction in a short-lived worker process to ensure VRAM goes back to 0.
    if subprocess_mode or (isinstance(state, dict) and state.get("_mode") == "subprocess"):
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

        run_dir = Path(state.get("_run_dir", Path(TMP_DIR) / str(req.session_hash)))
        logs_dir = ensure_dir(run_dir / "logs")
        work_dir = Path(TMP_DIR) / str(req.session_hash) / "subprocess" / str(state.get("_run_id", "extract"))
        work_dir.mkdir(parents=True, exist_ok=True)

        shape_slat_path = state.get("shape_slat_path")
        if not shape_slat_path:
            raise gr.Error("Missing shape latent on disk. Please Generate again with subprocess mode enabled.")
        tex_slat_path = state.get("tex_slat_path")
        res = int(state.get("res"))

        out_dir = run_dir / "08_extract"

        _log("Starting GLB extraction (subprocess)…", 0.0)
        # Show the overlay while extracting.
        yield None, None, gr.update(value=_trim_status(status), visible=True)

        export_formats = export_formats or ["glb"]
        if "glb" not in export_formats:
            export_formats = ["glb"] + list(export_formats)

        requested_remesh_method = str(remesh_method)
        if requested_remesh_method == "faithful_contouring" and not _is_faithful_contouring_available():
            remesh_method = "dual_contouring"
            _log(
                "WARNING: remesh_method='faithful_contouring' requires optional FaithC dependencies "
                "(`faithcontour` + `atom3d`) which are not installed. Falling back to 'dual_contouring'."
            )
            yield None, None, status

        payload = {
            "model_repo": "microsoft/TRELLIS.2-4B",
            "shape_slat_path": str(shape_slat_path),
            "tex_slat_path": str(tex_slat_path) if tex_slat_path else None,
            "res": int(res),
            "decimation_target": int(decimation_target),
            "texture_size": int(texture_size),
            "remesh_method": remesh_method,
            "simplify_method": simplify_method,
            "prune_invisible_faces": bool(prune_invisible_faces),
            "no_texture_gen": bool(no_texture_gen),
            "out_dir": str(out_dir),
            "prefix": "glb",
            "export_formats": list(export_formats),
            "extract_use_chunked_processing": bool(extract_use_chunked_processing),
            "extract_use_tiled_extraction": bool(extract_use_tiled_extraction),
        }

        last_ui_update = 0.0
        log_path = Path(logs_dir) / "extract_glb.log"
        result = None
        try:
            for ev in _iter_subprocess_stage("extract_glb", payload, work_dir, log_path, session=session):
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
            raise gr.Error("Extraction failed (no GLB path returned). See logs in the run folder.")

        glb_path = result["glb_path"]
        _log(f"Saved: {safe_relpath(glb_path, APP_DIR)}", 0.98)
        _log("Done.", 1.0)
        yield glb_path, glb_path, status
        return

    texture_extraction = not no_texture_gen

    run_dir = Path(state.get("_run_dir", os.path.join(TMP_DIR, str(req.session_hash))))
    out_dir = run_dir / "08_extract"
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
    # Show the overlay while extracting.
    yield None, None, gr.update(value=_trim_status(status), visible=True)

    _log("Loading TRELLIS.2 pipeline…", 0.05)
    pipe = get_image_pipeline()
    yield None, None, status
    
    _log("Decoding latent to mesh…", 0.15)
    mesh = pipe.decode_latent(shape_slat, tex_slat, res, extract_use_tiled_extraction, extract_use_chunked_processing)[0]
    yield None, None, status

    _log("Post-processing + baking GLB (this can take a while)…", 0.3)
    yield None, None, status

    requested_remesh_method = str(remesh_method)
    if requested_remesh_method == "faithful_contouring" and not _is_faithful_contouring_available():
        remesh_method = "dual_contouring"
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
        "simplify_method": simplify_method,
        "texture_extraction": texture_extraction,
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
        # Failsafe: if FaithC is missing/unusable, fall back to a built-in remesher
        # instead of crashing extraction.
        if requested_remesh_method == "faithful_contouring" and "Faithful Contouring is not installed" in str(e):
            fallback_method = "dual_contouring"
            _log(f"WARNING: {e} Falling back to remesh_method={fallback_method!r}.")
            to_glb_kwargs["remesh_method"] = fallback_method
            glb = o_voxel.postprocess.to_glb(**to_glb_kwargs)
        else:
            raise
    yield None, None, status

    _log("Saving GLB…", 0.9)
    export_formats = export_formats or ["glb"]
    if "glb" not in export_formats:
        export_formats = ["glb"] + list(export_formats)

    idx, glb_path_p = next_indexed_path(out_dir, prefix="glb", ext="glb", digits=4, start=1)
    glb.export(str(glb_path_p), extension_webp=True)
    glb_path = str(glb_path_p)

    # Optional extra exports (best effort; never fail the main GLB export).
    extras = [f for f in export_formats if f != "glb"]
    for fmt in extras:
        try:
            fmt = str(fmt).lower().strip()
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
            _log(f"Extra export '{fmt}' failed: {type(e).__name__}: {e}", 0.95)

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
    subprocess_mode: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    status = ""
    session = _session_key(req)

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

    if subprocess_mode:
        run = allocate_run_dir(OUTPUTS_DIR, digits=4)
        run_dir = run.run_dir
        run_id = run.run_id
        logs_dir = ensure_dir(run_dir / "logs")
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
                "tex_params": {
                    "steps": int(tex_slat_sampling_steps),
                    "guidance_strength": float(tex_slat_guidance_strength),
                    "guidance_rescale": float(tex_slat_guidance_rescale),
                    "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                    "rescale_t": float(tex_slat_rescale_t),
                },
            },
        )

        _log(f"Subprocess mode ON. Run: {run_id} → {safe_relpath(run_dir, APP_DIR)}", 0.02)
        yield None, None, status

        payload = {
            "model_repo": "microsoft/TRELLIS.2-4B",
            "config_file": "texturing_pipeline.json",
            "mesh_path": str(mesh_copy),
            "image_path": str(img_path),
            "seed": int(seed),
            "resolution": int(resolution),
            "texture_size": int(texture_size),
            "tex_params": {
                "steps": int(tex_slat_sampling_steps),
                "guidance_strength": float(tex_slat_guidance_strength),
                "guidance_rescale": float(tex_slat_guidance_rescale),
                "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                "rescale_t": float(tex_slat_rescale_t),
            },
            "out_dir": str(run_dir / "08_texturing"),
            "prefix": "textured",
        }

        last_ui_update = 0.0
        log_path = Path(logs_dir) / "texture_generate.log"
        result = None
        try:
            for ev in _iter_subprocess_stage("texture_generate", payload, work_dir, log_path, session=session):
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
            "tex_params": {
                "steps": int(tex_slat_sampling_steps),
                "guidance_strength": float(tex_slat_guidance_strength),
                "guidance_rescale": float(tex_slat_guidance_rescale),
                "guidance_interval": [float(tex_slat_guidance_interval_start), float(tex_slat_guidance_interval_end)],
                "rescale_t": float(tex_slat_rescale_t),
            },
        },
    )

    _log(f"Run: {run_id} → {safe_relpath(run_dir, APP_DIR)}", 0.0)
    yield None, None, status

    _log("Loading texturing pipeline (first run can take a while)…", 0.05)
    pipe = get_texturing_pipeline()
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
    out_dir = run_dir / "08_texturing"
    _, glb_path_p = next_indexed_path(out_dir, prefix="textured", ext="glb", digits=4, start=1)
    output.export(str(glb_path_p), extension_webp=True)
    glb_path = str(glb_path_p)
    torch.cuda.empty_cache()
    _log("Done.", 1.0)
    yield glb_path, glb_path, status


# --------------------------------- App UI -----------------------------------

with gr.Blocks(
    title="TRELLIS.2 Premium",
    theme=gr.themes.Soft(
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
    ),
    css=css,
    head=head,
    delete_cache=(600, 600),
) as demo:
    gr.Markdown(
        """
## TRELLIS.2 Premium
Generate a 3D asset from an image, export as GLB, and optionally texture an existing mesh.
"""
    )

    with gr.Row():
        subprocess_mode = gr.Checkbox(
            label="Subprocess stage processing (zero leftover VRAM between stages)",
            value=True,
        )

    demo.load(start_session)
    demo.unload(end_session)

    with gr.Tabs():
        # ---------------------------- Tab 1: Image -> 3D ----------------------------
        with gr.Tab("Image → 3D"):
            with gr.Row():
                with gr.Column(scale=1, min_width=380):
                    image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)

                    with gr.Row():
                        resolution = gr.Radio(["512", "768", "1024", "1280", "1536", "2048"], label="Resolution (Generate)", value="1024", info="Output mesh resolution. Higher = finer detail but more VRAM. 512 uses direct sampling; 768+ use cascade for quality. ⬆Quality ⬆VRAM", scale=3)
                        custom_resolution = gr.Number(label="Custom Resolution", value=0, precision=0, minimum=0, maximum=4096, step=128, info="Set to 0 to use radio selection. Must be ≥512 and divisible by 128. Overrides radio if >0.", scale=1)
                    with gr.Row():
                        seed = gr.Slider(0, MAX_SEED, label="Seed (Generate)", value=99, step=1, scale=4, info="Random seed for reproducibility. Same seed + settings = same output.")
                        randomize_seed = gr.Checkbox(label="Randomize Seed (Generate)", value=False, scale=1, info="Generate random seed each run for variety.")
                    decimation_target = gr.Slider(100000, 9000000, label="Decimation Target (Extract GLB)", value=1000000, step=10000, info="Target polygon count during mesh simplification. Higher = more geometric detail preserved but larger files. ⬆Quality, minimal VRAM impact.")
                    remesh_method = gr.Dropdown(REMESH_METHOD_CHOICES, label="Remesh Method (Extract GLB)", value="dual_contouring", info="Mesh reconstruction algorithm. dual_contouring: fast, good quality. faithful_contouring: higher fidelity (requires extra deps).")
                    if "faithful_contouring" not in REMESH_METHOD_CHOICES:
                        gr.Markdown(
                            "**Note:** `faithful_contouring` remeshing requires optional FaithC dependencies "
                            "(`faithcontour` + `atom3d`). Not detected in this environment, so the option is hidden."
                        )
                    simplify_method = gr.Dropdown(["cumesh", "meshlib"], label="Simplify Method (Extract GLB)", value="cumesh", info="Polygon reduction method. cumesh: GPU-accelerated, fast. meshlib: CPU-based alternative. cumesh uses some GPU VRAM.")
                    prune_invisible_faces = gr.Checkbox(label="Prune Invisible Faces (Extract GLB)", value=False, info="Remove triangles not visible from outside. Reduces polygon count, may affect internal geometry. Slight ⬇VRAM.")
                    no_texture_gen = gr.Checkbox(label="Skip Texture Generation (Generate + Extract GLB)", value=False, info="Output shape-only mesh without PBR textures. Faster processing, significantly ⬇VRAM usage.")
                    texture_size = gr.Slider(1024, 4096, label="Texture Size (Extract GLB)", value=2048, step=1024, info="Resolution of baked texture maps (albedo, normal, etc). Higher = sharper textures. ⬆Quality ⬆VRAM during baking.")
                    export_formats = gr.CheckboxGroup(
                        choices=["glb", "gltf", "obj", "ply", "stl"],
                        value=["glb", "gltf", "obj", "ply", "stl"],
                        label="Export Formats (Extract GLB)",
                    )

                    with gr.Accordion("⚙️ Config Presets (Save / Load)", open=True):
                        gr.Markdown(
                            "Saves/loads **all settings** from **both** tabs (inputs like images/files are not included)."
                        )
                        ui_preset_dropdown = gr.Dropdown(
                            label="Select Preset",
                            choices=_list_ui_presets(),
                            value=_get_last_used_ui_preset() or "",
                            allow_custom_value=False,
                        )
                        with gr.Row():
                            ui_preset_name = gr.Textbox(
                                label="New Preset Name",
                                placeholder="my_settings",
                                scale=3,
                            )
                            ui_preset_save_btn = gr.Button("💾 Save", variant="primary", scale=1)
                        with gr.Row():
                            ui_preset_load_btn = gr.Button("📂 Load Selected", scale=1)
                            ui_preset_reset_btn = gr.Button("🔄 Reset Defaults", variant="secondary", scale=1)
                            ui_preset_delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                        ui_preset_status = gr.Markdown("")

                with gr.Column(scale=3, min_width=680):
                    with gr.Walkthrough(selected=0) as walkthrough:
                        with gr.Step("Preview", id=0):
                            with gr.Column(elem_id="preview_stack"):
                                preview_output = gr.HTML(
                                    empty_html, label="3D Asset Preview", show_label=True, container=True
                                )
                                # Progress shown directly on top of the preview (no separate side panel).
                                status_box = gr.Textbox(
                                    value="Select an image (upload or example), then click Generate.",
                                    lines=20,
                                    max_lines=20,
                                    interactive=False,
                                    show_label=False,
                                    container=False,
                                    elem_id="preview_status_overlay",
                                )
                            with gr.Row():
                                generate_btn = gr.Button("Generate", variant="primary")
                                extract_btn = gr.Button("Extract GLB", interactive=False)
                                view_extract_btn = gr.Button("View Extracted", interactive=False)
                            cancel_confirm_state = gr.State({"armed": False, "armed_at": 0.0, "scope": ""})
                            with gr.Row():
                                open_outputs_top_btn = gr.Button("📂 Open outputs folder", variant="secondary")
                                view_logs_btn = gr.Button("📄 View Logs", variant="secondary")
                                cancel_processing_btn = gr.Button("🛑 Cancel processing", variant="stop")
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
                            with gr.Accordion(label="Advanced Settings (Generate)", open=True):
                                gr.Markdown("**Stage 1: Sparse Structure Generation (Generate)**")
                                with gr.Row():
                                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1, info="CFG scale - how strongly model follows image. Higher = more faithful but can oversaturate. 7.5 default. Slight ⬆VRAM (2 forward passes).")
                                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01, info="Reduces over-exposure from high CFG by normalizing variance. 0.7 recommended. No VRAM impact.")
                                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1, info="Denoising iterations. More = cleaner but slower. 12 is efficient. ⬆Quality, no per-step VRAM increase.")
                                with gr.Row():
                                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1, info="Time schedule warping. Higher = more steps on coarse structure. 5.0 default improves structure. No VRAM impact.")
                                    ss_guidance_interval_start = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01, info="⚠️ ADVANCED: Model default is 0.6. Only apply CFG in final refinement phase. Changing may reduce quality!")
                                    ss_guidance_interval_end = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval End", value=1.0, step=0.01, info="⚠️ ADVANCED: Model default is 1.0. Keep at 1.0 unless you know what you're doing.")
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
                                gr.Markdown("**Mesh Extraction Optimizations (Generate)**")
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
                                gr.Markdown("**Mesh Extraction Optimizations (Extract GLB)**")
                                with gr.Row():
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
                                        0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01, info="⚠️ ADVANCED: Model default is 0.6. Only apply CFG in final refinement phase. Changing may reduce quality!")
                                    shape_slat_guidance_interval_end = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval End", value=1.0, step=0.01, info="⚠️ ADVANCED: Model default is 1.0. Keep at 1.0 unless you know what you're doing.")
                                    max_num_tokens = gr.Slider(
                                        10000, 200000,
                                        label="Max Tokens (Generate - VRAM vs Quality)",
                                        value=49152,
                                        step=1000,
                                        info="KEY VRAM CONTROL. Max voxel tokens in cascade. 49K = original quality. Lower = less VRAM but may auto-reduce resolution. 10K min, 49K default, 65K+ for 2048. ⬆Quality ⬆VRAM (linear).")

                                gr.Markdown("**Stage 3: Material Generation (Generate)**")
                                with gr.Row():
                                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1, info="CFG for texture. Low (1.0) works well since shape provides strong conditioning. Slight ⬆VRAM if >1.")
                                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01, info="Variance normalization. 0.0 = disabled (not needed at low guidance). No VRAM impact.")
                                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1, info="Steps for texture generation. 12 is efficient. ⬆Quality, no per-step VRAM increase.")
                                with gr.Row():
                                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1, info="Time warping for texture. 3.0 default. No VRAM impact.")
                                    tex_slat_guidance_interval_start = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01, info="⚠️ ADVANCED: Model default is 0.6. Apply CFG in middle refinement phase. Changing may reduce quality!")
                                    tex_slat_guidance_interval_end = gr.Slider(
                                        0.0, 1.0, label="Guidance Interval End", value=0.9, step=0.01, info="⚠️ ADVANCED: Model default is 0.9. Texture uses 0.6-0.9 range. Changing may reduce quality!")
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

            gr.Markdown("### Examples")
            examples = gr.Examples(
                examples=[
                    os.path.join(APP_DIR, "assets", "example_image", image)
                    for image in os.listdir(os.path.join(APP_DIR, "assets", "example_image"))
                ],
                inputs=[image_prompt],
                examples_per_page=18,
            )

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
                        value="Select an image (upload or example), then click Generate.",
                    ),  # status_box
                    True,  # logs_visible_state
                    gr.update(value="📄 Hide Logs"),  # view_logs_btn
                )

            # Note: We intentionally do not auto-preprocess on upload/example click.
            # Both in-process and subprocess pipelines do preprocessing as part of "Generate",
            # and each run saves both the raw input + the preprocessed image under outputs/<run_id>/.

            # Any time the input image changes (upload, example click, clear), invalidate previous results.
            image_prompt.change(
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
                    image_prompt,
                    seed,
                    resolution,
                    custom_resolution,
                    ss_guidance_strength,
                    ss_guidance_rescale,
                    ss_guidance_interval_start,
                    ss_guidance_interval_end,
                    ss_sampling_steps,
                    ss_rescale_t,
                    force_high_res_conditional,
                    low_vram,
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
                    max_num_tokens,
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
                    simplify_method,
                    no_texture_gen,
                    prune_invisible_faces,
                    export_formats,
                    extract_use_chunked_processing,
                    extract_use_tiled_extraction,
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
                label = "⚠️ CONFIRM cancel (click again)"
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
                    ss_guidance_strength,
                    ss_guidance_rescale,
                    ss_guidance_interval_start,
                    ss_guidance_interval_end,
                    ss_sampling_steps,
                    ss_rescale_t,
                    force_high_res_conditional,
                    low_vram,
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
                    max_num_tokens,
                    decimation_target,
                    texture_size,
                    remesh_method,
                    simplify_method,
                    prune_invisible_faces,
                    export_formats,
                    subprocess_mode,
                ],
                outputs=[batch_status_box],
            )

        # ---------------------------- Tab 2: Texturing -------------------------------
        with gr.Tab("Texturing"):
            with gr.Row():
                with gr.Column(scale=1, min_width=380):
                    mesh_file = gr.File(label="Upload Mesh", file_types=[".ply", ".obj", ".glb", ".gltf"], file_count="single")
                    tex_image = gr.Image(label="Reference Image", format="png", image_mode="RGBA", type="pil", height=400)

                    tex_resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
                    with gr.Row():
                        tex_seed = gr.Slider(0, MAX_SEED, label="Seed", value=99, step=1, scale=4)
                        tex_randomize_seed = gr.Checkbox(label="Randomize Seed", value=False, scale=1)
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
                        with gr.Row():
                            t_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                            t_guidance_interval_start = gr.Slider(
                                0.0, 1.0, label="Guidance Interval Start", value=0.6, step=0.01
                            )
                            t_guidance_interval_end = gr.Slider(
                                0.0, 1.0, label="Guidance Interval End", value=0.9, step=0.01
                            )

                with gr.Column(scale=2, min_width=520):
                    textured_glb_output = gr.Model3D(label="Textured GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    with gr.Row():
                        textured_download_btn = gr.DownloadButton(label="Download Textured GLB", variant="primary")
                        tex_open_outputs_btn = gr.Button("Open outputs folder", variant="secondary")

            gr.Markdown("### Examples")
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
                    t_guidance_interval_start,
                    t_guidance_interval_end,
                    t_sampling_steps,
                    t_rescale_t,
                    subprocess_mode,
                ],
                outputs=[textured_glb_output, textured_download_btn, tex_status_box],
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

        # ---------------------------- Tab 3: View 3D Files ----------------------------
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
2. Upload an image in **Image Prompt** (best: one object, centered, clear silhouette).
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
**What it is**: When enabled, each major stage runs in a fresh Python subprocess. This keeps the UI process from “holding onto” VRAM between stages.  
**When to enable**:
- Enable if you get CUDA OOM errors, driver resets, or your VRAM stays high after a run.
- Enable if you run large resolutions (1536/2048) or do batch processing.
**When to disable**:
- Disable if you prefer slightly simpler execution and you’re not memory constrained.
**Important**:
- With subprocess mode ON, the **Cancel processing** button can immediately terminate the worker stage.
- With subprocess mode OFF, in-process work can only stop at “safe points” (and we intentionally only cancel batch).

---

## IMAGE → 3D settings (left panel)

### Image Prompt (upload)
Upload the input image you want to convert to 3D.

**Best practices**:
- Use a single main object. Avoid busy backgrounds.
- Center the object and keep it large in the frame.
- If you have a PNG with transparency, that’s ideal.

**Examples**:
- Good: a centered product photo on a plain background.
- Risky: multiple characters, cluttered scenery, tiny object in the distance.

### Resolution
Choose the target generation quality/speed level. Higher resolutions produce more detail but cost more VRAM/time.

**Options**:
- **512**: fastest and lightest. Great for quick tests and low‑VRAM GPUs.
- **1024**: good default balance (recommended starting point).
- **1536 / 2048**: highest detail, slowest, and most VRAM‑intensive.

**Example decision**:
- “I want fast previews”: start with **512** or **1024**.
- “I need maximum detail”: try **1536**, and only use **2048** if your GPU has enough VRAM and you can wait.

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
- 2048: default sweet spot.
- 4096: maximum crispness, heavy VRAM/disk.

### Auto‑save export formats
Select which formats are written under `./outputs/<run_id>/08_extract/`.

**Notes**:
- `glb` is always produced for the viewer/download.
- Extra formats (obj/ply/stl/gltf) are best-effort and may fail for some meshes; failures won’t block GLB export.

---

## Preview panel controls (right side)

### Generate
Runs the **Image → 3D** pipeline and builds the preview.

### Extract GLB
Converts the generated latents into an exportable mesh + textures (GLB) and saves to `./outputs/<run_id>/08_extract/`.

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

## Advanced Settings (what “guidance” means)

These parameters control diffusion sampling behavior. Think of them as “how strongly the model follows its conditioning” and “how the sampler behaves over time”.

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
- `start=0.6, end=1.0` means “apply stronger guidance mostly later”.
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
Same concepts as “guidance” above but applied to texture generation.

---

## View 3D Files tab

Upload a 3D file to preview it locally. This does not run the ML pipeline.

---

## Config Presets (Save / Load)

Presets save **all settings** from **both tabs**, but do **not** include uploaded images/files.

**Where presets are stored**: `./presets/<name>.json`

**Typical workflow**:
- Dial in settings you like → Save preset as `my_high_quality`
- Later → Load preset to restore all sliders/checkboxes instantly
"""
            )

    # ---------------------------- Preset Wiring ----------------------------
    _CONFIG_KEYS = [
        ("global", "subprocess_mode"),
        ("image_to_3d", "resolution"),
        ("image_to_3d", "seed"),
        ("image_to_3d", "randomize_seed"),
        ("image_to_3d", "decimation_target"),
        ("image_to_3d", "remesh_method"),
        ("image_to_3d", "simplify_method"),
        ("image_to_3d", "prune_invisible_faces"),
        ("image_to_3d", "no_texture_gen"),
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
        ("image_to_3d", "tex_slat_guidance_strength"),
        ("image_to_3d", "tex_slat_guidance_rescale"),
        ("image_to_3d", "tex_slat_guidance_interval_start"),
        ("image_to_3d", "tex_slat_guidance_interval_end"),
        ("image_to_3d", "tex_slat_sampling_steps"),
        ("image_to_3d", "tex_slat_rescale_t"),
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
    ]

    _CONFIG_COMPONENTS = [
        subprocess_mode,
        resolution,
        seed,
        randomize_seed,
        decimation_target,
        remesh_method,
        simplify_method,
        prune_invisible_faces,
        no_texture_gen,
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
        tex_slat_guidance_strength,
        tex_slat_guidance_rescale,
        tex_slat_guidance_interval_start,
        tex_slat_guidance_interval_end,
        tex_slat_sampling_steps,
        tex_slat_rescale_t,
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
        if merged["image_to_3d"]["resolution"] not in ["512", "1024", "1536", "2048"]:
            merged["image_to_3d"]["resolution"] = defaults["image_to_3d"]["resolution"]
        # Texturing resolution
        if merged["texturing"]["resolution"] not in ["512", "1024", "1536"]:
            merged["texturing"]["resolution"] = defaults["texturing"]["resolution"]
        # Remesh method (depends on env)
        if merged["image_to_3d"]["remesh_method"] not in REMESH_METHOD_CHOICES:
            merged["image_to_3d"]["remesh_method"] = defaults["image_to_3d"]["remesh_method"]
        # Simplify method
        if merged["image_to_3d"]["simplify_method"] not in ["cumesh", "meshlib"]:
            merged["image_to_3d"]["simplify_method"] = defaults["image_to_3d"]["simplify_method"]
        # Export formats
        ef = merged["image_to_3d"].get("export_formats")
        if not isinstance(ef, list):
            ef = defaults["image_to_3d"]["export_formats"]
        ef = [str(x) for x in ef if str(x) in {"glb", "gltf", "obj", "ply", "stl"}]
        merged["image_to_3d"]["export_formats"] = ef or defaults["image_to_3d"]["export_formats"]

        def _clamp01(v, d):
            try:
                v = float(v)
            except Exception:
                v = float(d)
            return max(0.0, min(1.0, v))

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
            return gr.update(), f"❌ Save failed: {e}"

    def _load_preset_ui(preset_name: str):
        if not preset_name:
            cfg = _default_ui_config()
            vals = _ui_config_to_values(cfg)
            return (*vals, "ℹ️ No preset selected (showing defaults).")

        cfg = _load_ui_preset(preset_name)
        if not cfg:
            cfg = _default_ui_config()
            vals = _ui_config_to_values(cfg)
            return (*vals, f"⚠️ Preset **{preset_name}** not found (loaded defaults).")

        vals = _ui_config_to_values(cfg)
        return (*vals, f"✅ Loaded preset **{preset_name}**")

    def _reset_defaults_ui():
        cfg = _default_ui_config()
        vals = _ui_config_to_values(cfg)
        return (*vals, "✅ Reset to defaults")

    def _delete_preset_ui(preset_name: str):
        if not preset_name:
            return gr.update(), "⚠️ No preset selected"
        ok = _delete_ui_preset(preset_name)
        presets = _list_ui_presets()
        if ok:
            return gr.update(choices=presets, value=""), f"✅ Deleted preset **{preset_name}**"
        return gr.update(choices=presets), f"⚠️ Could not delete preset **{preset_name}**"

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
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--server-port", type=int, default=7860, help="Port to listen on (default: 7860)")
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
    demo.launch(
        share=args.share,
        inbrowser=True,
        show_error=True,
    )
