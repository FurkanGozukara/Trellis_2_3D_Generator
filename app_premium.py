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
from typing import Tuple

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


def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
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


def preprocess_image(image: Image.Image) -> Image.Image:
    pipe = get_image_pipeline()
    return pipe.preprocess_image(image)


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
) -> Tuple[dict, str]:
    pipe = get_image_pipeline()
    envmap = _get_envmap()
    _ensure_mode_icons()

    outputs, latents = pipe.run(
        image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
            "2048": "2048_cascade",
        }[resolution],
        return_latent=True,
        max_num_tokens=max_num_tokens,
        no_texture_gen=no_texture_gen,
    )
    mesh = outputs[0]
    mesh.simplify(16777216)  # nvdiffrast limit
    images = render_utils.render_snapshot(mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=envmap)
    state = pack_state(latents)
    torch.cuda.empty_cache()

    # Build HTML (48 images)
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
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

    return state, full_html


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
) -> Tuple[str, str]:
    texture_extraction = not no_texture_gen

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shape_slat, tex_slat, res = unpack_state(state)
    pipe = get_image_pipeline()
    mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]
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
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f"sample_{timestamp}.glb")
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path


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
) -> Tuple[str, str]:
    pipe = get_texturing_pipeline()

    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    output = pipe.run(
        mesh,
        image,
        seed=seed,
        # Let the pipeline handle preprocessing so Examples also work correctly.
        preprocess_image=True,
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        resolution=int(resolution),
        texture_size=texture_size,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f"textured_{timestamp}.glb")
    output.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path


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

                    generate_btn = gr.Button("Generate", variant="primary")

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
                            extract_btn = gr.Button("Extract GLB")
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

            image_prompt.upload(
                preprocess_image,
                inputs=[image_prompt],
                outputs=[image_prompt],
            )

            generate_btn.click(
                get_seed,
                inputs=[randomize_seed, seed],
                outputs=[seed],
            ).then(
                lambda: gr.Walkthrough(selected=0), outputs=walkthrough
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
                outputs=[output_buf, preview_output],
            )

            extract_btn.click(
                lambda: gr.Walkthrough(selected=1), outputs=walkthrough
            ).then(
                extract_glb,
                inputs=[output_buf, decimation_target, texture_size, remesh_method, simplify_method, no_texture_gen, prune_invisible_faces],
                outputs=[glb_output, download_btn],
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
                outputs=[textured_glb_output, textured_download_btn],
            )


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


