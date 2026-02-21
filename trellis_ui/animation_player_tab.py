"""
Animation Player Tab - Browse and view previously rigged models.
Displays rigged models with metadata and download options.
"""
import gradio as gr
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

from subprocess_utils import allocate_run_dir


def animation_player_tab(list_models_fn, rigging_outputs_dir, open_folder_fn):
    """
    Create the Animation Player tab interface.
    
    Args:
        list_models_fn: Function to list rigged models
        rigging_outputs_dir: Path to rigging outputs directory
        open_folder_fn: Function to open folder in file explorer
    
    Features:
    - Browse previously rigged models
    - View models in 3D viewer
    - Display rigging metadata (bone count, file info)
    - Download rigged models
    """
    
    with gr.Row():
        # Left Column: Browser and Controls
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("## Rigged Models Browser")
            
            # Model selection
            rigged_models_dropdown = gr.Dropdown(
                label="Select Rigged Model",
                choices=list_models_fn(),
                interactive=True,
                info="Choose from previously rigged models"
            )
            
            refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")

            upload_model_file = gr.File(
                label="Upload Model To Browser",
                file_types=[".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"],
                file_count="single",
            )
            
            # Metadata display
            gr.Markdown("## Model Information")
            metadata_display = gr.JSON(
                label="Metadata",
                show_label=False,
            )
            
            # File info
            file_info_text = gr.Textbox(
                label="File Details",
                lines=5,
                interactive=False,
                show_label=False,
                placeholder="Select a model to view details..."
            )
            
            # Actions
            gr.Markdown("## Actions")
            download_model_btn = gr.DownloadButton(
                label="⬇️ Download Model",
                variant="primary",
                size="lg",
                visible=False
            )
            
            open_folder_btn = gr.Button("📁 Open Outputs Folder", variant="secondary")
            clear_viewer_btn = gr.Button("🗑️ Clear Viewer", variant="secondary")
        
        # Right Column: 3D Viewer and Info
        with gr.Column(scale=2, min_width=520):
            gr.Markdown("## 3D Viewer")
            model_viewer = gr.Model3D(
                label="Rigged Model Viewer",
                height=600,
                show_label=False,
                display_mode="solid",
                clear_color=[0.2, 0.2, 0.25, 1.0]
            )
            
            # Info panel
            gr.Markdown("""
            ### ℹ️ Animation Player Information
            
            **Viewing Rigged Models:**
            - Select a model from the dropdown to view it in 3D
            - Rotate: Click and drag
            - Zoom: Scroll wheel
            - Pan: Shift + Click and drag
            
            **Animation Notes:**
            - Embedded GLB animations auto-play in this viewer when present
            - FBX files are previewed via generated GLB fallback when available
            - For full rig editing, still use external DCC tools:
              - **Blender** (recommended - free, full rigging support)
              - **Unity** / **Unreal Engine**
              - **Maya/3ds Max**
            
            **Supported Formats:**
            - `.fbx` - Best for most 3D software (preserves skeleton hierarchy)
            - `.glb` - Good for web/real-time applications
            """)
    
    # State to track selected model path
    selected_model_path = gr.State(None)
    external_select_input = gr.Textbox(visible=False, label="External Model Path")

    def _safe_filename(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
        safe = safe.strip("._")
        return safe or "model"

    def _uploaded_file_to_path(file):
        if file is None:
            return None
        if isinstance(file, str):
            return file
        if isinstance(file, dict):
            return file.get("path") or file.get("name")
        return getattr(file, "name", None)

    def _import_model_to_browser(src_path: str) -> str:
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {src_path}")

        run = allocate_run_dir(Path(rigging_outputs_dir), digits=4)
        work_dir = run.run_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        dst = work_dir / _safe_filename(src.name)
        shutil.copy2(src, dst)
        return str(dst.relative_to(Path(rigging_outputs_dir)))

    def _resolve_to_rel_model_path(path_value: str) -> str:
        if not path_value:
            raise ValueError("No model path provided.")

        root = Path(rigging_outputs_dir).resolve()
        path = Path(path_value)

        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            abs_path = path.resolve()
            try:
                return str(abs_path.relative_to(root))
            except ValueError:
                return _import_model_to_browser(str(abs_path))

        rel_candidate = root / path
        if rel_candidate.exists():
            return str(path)

        raise FileNotFoundError(f"Model file not found: {path_value}")

    def _normalize_model3d_path(path_value: str):
        """
        Normalize a file path for Gradio Model3D.
        Uses absolute path + forward slashes for consistent browser loading on Windows.
        """
        if not path_value:
            return None
        try:
            path = Path(path_value).resolve()
            if not path.exists() or not path.is_file():
                return None
            return path.as_posix()
        except Exception:
            return None

    def load_model(model_rel_path):
        """Load selected model and extract metadata."""
        if not model_rel_path:
            return (
                gr.update(value=None),  # viewer
                None,  # metadata
                "No model selected",  # file_info
                None,  # selected_model_path
                gr.update(visible=False),  # download_btn
            )
        
        try:
            # Construct full path
            full_path = Path(rigging_outputs_dir) / model_rel_path
            
            if not full_path.exists():
                return (
                    gr.update(value=None),
                    {"error": "File not found"},
                    f"❌ File not found: {model_rel_path}",
                    None,
                    gr.update(visible=False),
                )
            
            # Get file info
            file_size = full_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            file_ext = full_path.suffix.lower()
            file_modified = full_path.stat().st_mtime
            
            from datetime import datetime
            modified_date = datetime.fromtimestamp(file_modified).strftime("%Y-%m-%d %H:%M:%S")
            
            file_info = f"""**File:** {full_path.name}
**Size:** {file_size_mb:.2f} MB
**Format:** {file_ext}
**Modified:** {modified_date}
**Path:** {model_rel_path}"""
            
            # Try to extract metadata from adjacent files
            metadata = {
                "filename": full_path.name,
                "format": file_ext,
                "size_mb": round(file_size_mb, 2),
            }

            # Decide what to show in Model3D. Prefer animation previews for FBX.
            previewable_exts = {".glb", ".gltf", ".obj", ".ply", ".stl"}
            viewer_path = str(full_path) if file_ext in previewable_exts else None
            viewer_note = ""

            metadata_path = None
            run_meta = {}
            for parent in [full_path.parent, *full_path.parents]:
                cand = parent / "run_metadata.json"
                if cand.exists():
                    metadata_path = cand
                    break

            if metadata_path is not None:
                try:
                    data = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        run_meta = data
                except Exception:
                    run_meta = {}

            if viewer_path is None:
                paths_meta = run_meta.get("paths", {}) if isinstance(run_meta, dict) else {}
                if "_skeleton" in full_path.stem.lower():
                    preferred_preview = paths_meta.get("skeleton_preview") or paths_meta.get("animation_preview")
                else:
                    preferred_preview = paths_meta.get("animation_preview") or paths_meta.get("skeleton_preview")
                if preferred_preview and os.path.exists(preferred_preview):
                    viewer_path = str(Path(preferred_preview))
                    viewer_note = f"Viewer: metadata preview ({Path(preferred_preview).name})"

            if viewer_path is None:
                sibling_candidates = [
                    full_path.with_name(f"{full_path.stem}_anim_preview.glb"),
                    full_path.with_name(f"{full_path.stem}_animation_preview.glb"),
                    full_path.with_suffix(".glb"),
                    full_path.with_suffix(".gltf"),
                ]
                for cand in sibling_candidates:
                    if cand.exists():
                        viewer_path = str(cand)
                        viewer_note = f"Viewer fallback: sibling {cand.name}"
                        break

            if viewer_path is None:
                input_meta = run_meta.get("input", {}) if isinstance(run_meta, dict) else {}
                fallback = input_meta.get("preview_path") or input_meta.get("copied_input_path")
                if fallback and os.path.exists(fallback):
                    fallback_ext = Path(fallback).suffix.lower()
                    if fallback_ext in previewable_exts:
                        viewer_path = str(Path(fallback))
                        viewer_note = f"Viewer fallback: source mesh ({Path(fallback).name})"

            if viewer_path is None and file_ext == ".fbx":
                viewer_note = "FBX preview is not supported by Model3D unless a GLB fallback exists."

            normalized_viewer = _normalize_model3d_path(viewer_path)
            if viewer_path and not normalized_viewer:
                viewer_note = f"Viewer path was not accessible: {viewer_path}"
                viewer_path = None
            else:
                viewer_path = normalized_viewer
            
            # Look for log file or metadata file
            log_file = full_path.parent / (full_path.stem + "_log.txt")
            if log_file.exists():
                metadata["has_log"] = True
            
            # Check for skeleton vs skinned
            if "skeleton" in full_path.stem.lower():
                metadata["type"] = "Skeleton Only"
            elif "skinned" in full_path.stem.lower() or "rigged" in full_path.stem.lower():
                metadata["type"] = "Rigged (Skinned)"
            else:
                metadata["type"] = "Unknown"

            metadata["viewer_path"] = viewer_path
            if viewer_note:
                metadata["viewer_note"] = viewer_note
                file_info += f"\n**Viewer:** {viewer_note}"
            
            return (
                gr.update(value=viewer_path),  # viewer (load previewable model or fallback)
                metadata,
                file_info,
                str(full_path),  # selected_model_path
                gr.update(visible=True, value=str(full_path)),  # download_btn
            )
        
        except Exception as e:
            return (
                gr.update(value=None),
                {"error": str(e)},
                f"❌ Error loading model: {e}",
                None,
                gr.update(visible=False),
            )
    
    # Load model when dropdown changes
    rigged_models_dropdown.change(
        fn=load_model,
        inputs=[rigged_models_dropdown],
        outputs=[
            model_viewer,
            metadata_display,
            file_info_text,
            selected_model_path,
            download_model_btn,
        ],
        queue=False,
        show_progress="minimal"
    )

    def _load_external_model(path_value: str):
        if not path_value:
            return (
                gr.update(),
                gr.update(value=None),
                None,
                "No model selected",
                None,
                gr.update(visible=False),
            )

        try:
            model_rel_path = _resolve_to_rel_model_path(path_value)
            models = list_models_fn()
            if model_rel_path not in models:
                models = sorted(set(models + [model_rel_path]))
            viewer, metadata, file_info, selected_path, download_update = load_model(model_rel_path)
            return (
                gr.update(choices=models, value=model_rel_path),
                viewer,
                metadata,
                file_info,
                selected_path,
                download_update,
            )
        except Exception as e:
            return (
                gr.update(),
                gr.update(value=None),
                {"error": str(e)},
                f"❌ Error loading model: {e}",
                None,
                gr.update(visible=False),
            )

    external_select_input.change(
        fn=_load_external_model,
        inputs=[external_select_input],
        outputs=[
            rigged_models_dropdown,
            model_viewer,
            metadata_display,
            file_info_text,
            selected_model_path,
            download_model_btn,
        ],
        queue=False,
        show_progress="minimal",
    )

    def _upload_model_to_browser(file):
        path = _uploaded_file_to_path(file)
        if not path:
            return (
                gr.update(),
                gr.update(value=None),
                {"error": "No file uploaded"},
                "❌ Please upload a valid model file.",
                None,
                gr.update(visible=False),
            )
        return _load_external_model(path)

    upload_model_file.change(
        fn=_upload_model_to_browser,
        inputs=[upload_model_file],
        outputs=[
            rigged_models_dropdown,
            model_viewer,
            metadata_display,
            file_info_text,
            selected_model_path,
            download_model_btn,
        ],
        queue=False,
        show_progress="minimal",
    )
    
    # Refresh model list
    def refresh_models():
        models = list_models_fn()
        return gr.update(choices=models, value=None)
    
    refresh_btn.click(
        fn=refresh_models,
        outputs=[rigged_models_dropdown],
        queue=False,
        show_progress="hidden"
    )
    
    # Open outputs folder
    def open_outputs():
        try:
            open_folder_fn(rigging_outputs_dir)
        except Exception as e:
            gr.Warning(f"Failed to open folder: {e}")
    
    open_folder_btn.click(
        fn=open_outputs,
        queue=False,
        show_progress="hidden"
    )
    
    # Clear viewer
    def clear_viewer():
        return (
            None,  # dropdown
            gr.update(value=None),  # viewer
            None,  # metadata
            "Select a model to view details...",  # file_info
            None,  # selected_model_path
            gr.update(visible=False),  # download_btn
        )
    
    clear_viewer_btn.click(
        fn=clear_viewer,
        outputs=[
            rigged_models_dropdown,
            model_viewer,
            metadata_display,
            file_info_text,
            selected_model_path,
            download_model_btn,
        ],
        queue=False,
        show_progress="hidden"
    )

    return {
        "external_select_input": external_select_input,
        "rigged_models_dropdown": rigged_models_dropdown,
    }
