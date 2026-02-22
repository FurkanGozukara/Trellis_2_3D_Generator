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


def animation_player_tab(
    list_models_fn,
    rigging_outputs_dir,
    open_folder_fn,
    generate_animation_preview_fn=None,
):
    """
    Create the Animation Player tab interface.
    
    Args:
        list_models_fn: Function to list rigged models
        rigging_outputs_dir: Path to rigging outputs directory
        open_folder_fn: Function to open folder in file explorer
        generate_animation_preview_fn: Optional callback to regenerate animation preview
    
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

            animation_style = gr.Dropdown(
                label="Animation Style",
                choices=["dance", "walk", "idle"],
                value="dance",
                info="Generate stronger rig animation previews.",
            )
            animation_strength = gr.Slider(
                label="Animation Strength",
                minimum=0.4,
                maximum=2.5,
                step=0.1,
                value=1.5,
            )
            animation_frames = gr.Slider(
                label="Animation Frames",
                minimum=45,
                maximum=240,
                step=5,
                value=120,
            )
            regenerate_animation_btn = gr.Button(
                "🎞️ Generate Animation Preview",
                variant="secondary",
                interactive=bool(generate_animation_preview_fn),
            )
            animation_action_status = gr.Textbox(
                label="Animation Status",
                lines=3,
                interactive=False,
                show_label=False,
                placeholder="Select a model then generate animation preview.",
            )

            open_folder_btn = gr.Button("📁 Open Outputs Folder", variant="secondary")
            clear_viewer_btn = gr.Button("🗑️ Clear Viewer", variant="secondary")
        
        # Right Column: 3D Viewer and Info
        with gr.Column(scale=2, min_width=520):
            gr.Markdown("## 3D Viewer")
            with gr.Tabs() as preview_tabs:
                with gr.Tab("Animated", id="preview_tab_animated"):
                    animated_viewer = gr.Model3D(
                        label="Animated Preview",
                        height=600,
                        show_label=False,
                        display_mode="solid",
                        clear_color=[0.2, 0.2, 0.25, 1.0],
                    )
                with gr.Tab("Textured", id="preview_tab_textured"):
                    textured_viewer = gr.Model3D(
                        label="Textured Preview",
                        height=600,
                        show_label=False,
                        display_mode="solid",
                        clear_color=[0.2, 0.2, 0.25, 1.0],
                    )
                with gr.Tab("Skeleton", id="preview_tab_skeleton"):
                    skeleton_viewer = gr.Model3D(
                        label="Skeleton Preview",
                        height=600,
                        show_label=False,
                        display_mode="solid",
                        clear_color=[0.2, 0.2, 0.25, 1.0],
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
            - Animated tab: generated animation preview
            - Textured tab: merged textured rig
            - Skeleton tab: skeleton preview overlay
            - Embedded GLB animations auto-play in this viewer when present
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
        previewable_exts = {".glb", ".gltf", ".obj", ".ply", ".stl", ".splat"}
        if not path_value:
            return None
        try:
            path = Path(path_value).resolve()
            if not path.exists() or not path.is_file():
                return None
            if path.suffix.lower() not in previewable_exts:
                return None
            return path.as_posix()
        except Exception:
            return None

    def _load_run_metadata(full_path: Path):
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
        return run_meta

    def _candidate_stems(stem: str):
        stems = []
        if stem:
            stems.append(stem)
        suffix_tokens = (
            "_textured_anim_preview",
            "_anim_preview",
            "_animation_preview",
            "_textured_preview",
            "_skeleton_preview",
            "_skinned",
            "_skeleton",
            "_rigged",
        )
        for token in suffix_tokens:
            if stem.endswith(token):
                trimmed = stem[: -len(token)]
                if trimmed:
                    stems.append(trimmed)
        # keep order but unique
        dedup = []
        for item in stems:
            if item not in dedup:
                dedup.append(item)
        return dedup

    def _first_previewable(candidates):
        for candidate in candidates:
            normalized = _normalize_model3d_path(candidate)
            if normalized:
                return normalized
        return None

    def _collect_preview_paths(full_path: Path, run_meta):
        paths_meta = run_meta.get("paths", {}) if isinstance(run_meta, dict) else {}
        input_meta = run_meta.get("input", {}) if isinstance(run_meta, dict) else {}
        stems = _candidate_stems(full_path.stem)

        animated_candidates = [
            paths_meta.get("textured_animation_preview"),
            paths_meta.get("animation_preview"),
        ]
        textured_candidates = [
            paths_meta.get("textured_preview"),
            paths_meta.get("final_output"),
            input_meta.get("preview_path"),
            input_meta.get("copied_input_path"),
        ]
        skeleton_candidates = [
            paths_meta.get("skeleton_preview"),
        ]

        # Include selected model when it already matches one of preview classes.
        stem_l = full_path.stem.lower()
        if "_anim_preview" in stem_l or "_animation_preview" in stem_l:
            animated_candidates.insert(0, str(full_path))
        if "_textured_preview" in stem_l or "_rigged" in stem_l:
            textured_candidates.insert(0, str(full_path))
        if "_skeleton_preview" in stem_l:
            skeleton_candidates.insert(0, str(full_path))

        for stem in stems:
            animated_candidates.extend(
                [
                    str(full_path.with_name(f"{stem}_textured_anim_preview.glb")),
                    str(full_path.with_name(f"{stem}_skinned_anim_preview.glb")),
                    str(full_path.with_name(f"{stem}_anim_preview.glb")),
                    str(full_path.with_name(f"{stem}_animation_preview.glb")),
                ]
            )
            textured_candidates.extend(
                [
                    str(full_path.with_name(f"{stem}_skinned_textured_preview.glb")),
                    str(full_path.with_name(f"{stem}_textured_preview.glb")),
                    str(full_path.with_name(f"{stem}_rigged.glb")),
                    str(full_path.with_name(f"{stem}_rigged.gltf")),
                ]
            )
            skeleton_candidates.append(str(full_path.with_name(f"{stem}_skeleton_preview.glb")))

        animated_path = _first_previewable(animated_candidates)
        textured_path = _first_previewable(textured_candidates)
        skeleton_path = _first_previewable(skeleton_candidates)

        # Reasonable fallback for textured view if none was generated.
        if not textured_path:
            textured_path = _first_previewable([str(full_path)])

        if animated_path:
            selected_tab = "preview_tab_animated"
        elif textured_path:
            selected_tab = "preview_tab_textured"
        elif skeleton_path:
            selected_tab = "preview_tab_skeleton"
        else:
            selected_tab = "preview_tab_animated"

        return {
            "animated": animated_path,
            "textured": textured_path,
            "skeleton": skeleton_path,
            "selected_tab": selected_tab,
        }

    def load_model(model_rel_path):
        """Load selected model and extract metadata."""
        if not model_rel_path:
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.Tabs(selected="preview_tab_animated"),
                None,
                "No model selected",
                None,
                gr.update(visible=False),
            )
        
        try:
            # Construct full path
            full_path = Path(rigging_outputs_dir) / model_rel_path
            
            if not full_path.exists():
                return (
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.Tabs(selected="preview_tab_animated"),
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

            run_meta = _load_run_metadata(full_path)
            previews = _collect_preview_paths(full_path, run_meta)
            viewer_note = ""
            if not previews["animated"] and not previews["textured"] and not previews["skeleton"]:
                if file_ext == ".fbx":
                    viewer_note = "FBX preview is not supported by Model3D unless a GLB fallback exists."
                else:
                    viewer_note = "No previewable 3D assets were found for this run."
            
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

            metadata["preview_paths"] = {
                "animated": previews["animated"],
                "textured": previews["textured"],
                "skeleton": previews["skeleton"],
            }
            if viewer_note:
                metadata["viewer_note"] = viewer_note
                file_info += f"\n**Viewer:** {viewer_note}"
            
            return (
                gr.update(value=previews["animated"]),
                gr.update(value=previews["textured"]),
                gr.update(value=previews["skeleton"]),
                gr.Tabs(selected=previews["selected_tab"]),
                metadata,
                file_info,
                str(full_path),  # selected_model_path
                gr.update(visible=True, value=str(full_path)),  # download_btn
            )
        
        except Exception as e:
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.Tabs(selected="preview_tab_animated"),
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
            animated_viewer,
            textured_viewer,
            skeleton_viewer,
            preview_tabs,
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
                gr.update(value=None),
                gr.update(value=None),
                gr.Tabs(selected="preview_tab_animated"),
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
            (
                animated_update,
                textured_update,
                skeleton_update,
                tab_update,
                metadata,
                file_info,
                selected_path,
                download_update,
            ) = load_model(model_rel_path)
            return (
                gr.update(choices=models, value=model_rel_path),
                animated_update,
                textured_update,
                skeleton_update,
                tab_update,
                metadata,
                file_info,
                selected_path,
                download_update,
            )
        except Exception as e:
            return (
                gr.update(),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.Tabs(selected="preview_tab_animated"),
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
            animated_viewer,
            textured_viewer,
            skeleton_viewer,
            preview_tabs,
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
                gr.update(value=None),
                gr.update(value=None),
                gr.Tabs(selected="preview_tab_animated"),
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
            animated_viewer,
            textured_viewer,
            skeleton_viewer,
            preview_tabs,
            metadata_display,
            file_info_text,
            selected_model_path,
            download_model_btn,
        ],
        queue=False,
        show_progress="minimal",
    )

    def _regenerate_animation_preview(selected_path, model_rel_path, style, strength, frames, req: gr.Request):
        if not generate_animation_preview_fn:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "Animation regeneration is not available in this build.",
            )

        try:
            target_path = selected_path
            if not target_path and model_rel_path:
                target_path = str((Path(rigging_outputs_dir) / model_rel_path).resolve())
            if not target_path:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "Select a model first.",
                )

            _preview_path, status = generate_animation_preview_fn(
                target_path,
                str(style),
                float(strength),
                int(frames),
                req,
            )

            rel_to_reload = model_rel_path or _resolve_to_rel_model_path(target_path)
            (
                animated_update,
                textured_update,
                skeleton_update,
                tab_update,
                metadata,
                file_info,
                selected_path_out,
                download_update,
            ) = load_model(rel_to_reload)

            return (
                animated_update,
                textured_update,
                skeleton_update,
                tab_update,
                metadata,
                file_info,
                selected_path_out,
                download_update,
                status,
            )
        except Exception as e:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                f"❌ Failed to regenerate animation: {type(e).__name__}: {e}",
            )

    regenerate_animation_btn.click(
        fn=_regenerate_animation_preview,
        inputs=[
            selected_model_path,
            rigged_models_dropdown,
            animation_style,
            animation_strength,
            animation_frames,
        ],
        outputs=[
            animated_viewer,
            textured_viewer,
            skeleton_viewer,
            preview_tabs,
            metadata_display,
            file_info_text,
            selected_model_path,
            download_model_btn,
            animation_action_status,
        ],
        queue=True,
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
            gr.update(value=None),  # animated_viewer
            gr.update(value=None),  # textured_viewer
            gr.update(value=None),  # skeleton_viewer
            gr.Tabs(selected="preview_tab_animated"),  # preview_tabs
            None,  # metadata
            "Select a model to view details...",  # file_info
            None,  # selected_model_path
            gr.update(visible=False),  # download_btn
            "Cleared.",
        )
    
    clear_viewer_btn.click(
        fn=clear_viewer,
        outputs=[
            rigged_models_dropdown,
            animated_viewer,
            textured_viewer,
            skeleton_viewer,
            preview_tabs,
            metadata_display,
            file_info_text,
            selected_model_path,
            download_model_btn,
            animation_action_status,
        ],
        queue=False,
        show_progress="hidden"
    )

    return {
        "external_select_input": external_select_input,
        "rigged_models_dropdown": rigged_models_dropdown,
    }
