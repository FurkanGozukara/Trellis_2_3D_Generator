"""
Rigging Tab - UniRig integration for automatic skeleton generation and skinning.
Allows users to upload 3D models and rig them using the UniRig system.
"""
import gradio as gr
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# Import helper functions from parent directory
import sys
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from subprocess_utils import allocate_run_dir
from trellis_ui.common_ui import open_folder, append_status, trim_status


def rigging_tab(
    run_skeleton_fn,
    run_skinning_fn,
    run_merge_fn,
    rigging_outputs_dir,
    open_folder_fn,
):
    """
    Create the UniRig rigging tab interface.
    
    Args:
        run_skeleton_fn: Function to run skeleton generation
        run_skinning_fn: Function to run skinning prediction
        run_merge_fn: Function to run merge operation
        rigging_outputs_dir: Path to rigging outputs directory
        open_folder_fn: Function to open folder in file explorer
    
    Features:
    - Upload mesh files (.obj, .fbx, .glb, .vrm)
   - Generate skeleton with seed control
    - Add skinning weights
    - Merge and export rigged models
    - Preview rigged models in 3D viewer
    """
    
    
    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("## Upload 3D Model")
            mesh_upload = gr.File(
                label="Upload Mesh",
                file_types=[".obj", ".fbx", ".glb", ".vrm"],
                file_count="single"
            )
            
            # Skeleton Settings
            with gr.Accordion("Skeleton Settings", open=True):
                with gr.Row():
                    rig_seed = gr.Number(label="Seed", value=12345, precision=0)
                    rig_randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                
                gr.Markdown("""
                **Skeleton Generation** uses UniRig's autoregressive model to predict a topologically valid skeleton structure for your 3D model.
                
                - Supports humans, animals, and various object types
                - Seed controls the skeleton variation (try different seeds for different results)
                """)
            
            # Skinning Settings
            with gr.Accordion("Skinning Settings", open=True):
                enable_skinning = gr.Checkbox(
                    label="Enable Skinning",
                    value=True,
                    info="Automatically add skinning weights after skeleton generation"
                )
                
                gr.Markdown("""
                **Skinning** predicts per-vertex weights that bind the mesh to the skeleton bones.
                
                - Essential for animation
                - Can be skipped if you only need the skeleton structure
                """)
            
            # Export Settings
            with gr.Accordion("Export Settings", open=True):
                export_format = gr.Radio(
                    choices=["fbx", "glb"],
                    label="Export Format",
                    value="fbx",
                    info="FBX preserves skeleton hierarchy better for most 3D software"
                )
                export_both_formats = gr.Checkbox(
                    label="Also export the other format",
                    value=True,
                    info="Saves both FBX and GLB so users can pick the best file for DCC tools or preview."
                )
                
                auto_merge = gr.Checkbox(
                    label="Auto-merge with original mesh",
                    value=True,
                    info="Automatically combine rig with original mesh textures/materials"
                )
            
            # Action Buttons
            gr.Markdown("## Actions")
            generate_skeleton_btn = gr.Button("🦴 Generate Skeleton", variant="primary", size="lg")
            add_skinning_btn = gr.Button("🎨 Add Skinning", variant="secondary", size="lg")
            export_rigged_btn = gr.Button("💾 Export Rigged Model", variant="secondary", size="lg")
            send_to_animation_btn = gr.Button("➡ Open In Animation Browser", variant="secondary", size="lg")
            
            with gr.Row():
                open_outputs_btn = gr.Button("📁 Open Outputs Folder", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Right Column: Preview and Status
        with gr.Column(scale=2, min_width=520):
            gr.Markdown("## Preview")
            rigged_model_preview = gr.Model3D(
                label="Rigged Model Preview",
                height=600,
                show_label=False,
                display_mode="solid",
                clear_color=[0.2, 0.2, 0.25, 1.0]
            )
            
            # Status Section
            gr.Markdown("## Status")
            rig_status = gr.Textbox(
                label="Processing Status",
                lines=12,
                max_lines=20,
                show_label=False,
                placeholder="Upload a mesh and click 'Generate Skeleton' to begin...",
                interactive=False
            )
            
            # Download button (hidden until export is complete)
            download_btn = gr.DownloadButton(
                label="⬇️ Download Rigged Model",
                visible=False,
                variant="primary"
            )
    
    # State variables to track workflow
    skeleton_path_state = gr.State(None)
    skinned_path_state = gr.State(None)
    final_output_state = gr.State(None)
    original_mesh_state = gr.State(None)
    upload_run_dir_state = gr.State(None)
    
    # Helper to update seed when randomize is toggled
    def randomize_seed_fn(randomize: bool, current_seed: int):
        if randomize:
            import numpy as np
            return np.random.randint(0, 2**31 - 1)
        return current_seed
    
    rig_randomize_seed.change(
        fn=randomize_seed_fn,
        inputs=[rig_randomize_seed, rig_seed],
        outputs=[rig_seed],
        queue=False,
        show_progress="hidden"
    )
    
    # Store uploaded mesh path and show immediate preview when supported.
    def _uploaded_file_to_path(file):
        if file is None:
            return None
        if isinstance(file, str):
            return file
        if isinstance(file, dict):
            return file.get("path") or file.get("name")
        return getattr(file, "name", None)

    def _safe_filename(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
        safe = safe.strip("._")
        return safe or "upload_mesh"

    def _create_upload_workspace(upload_path: str):
        root = Path(rigging_outputs_dir)
        root.mkdir(parents=True, exist_ok=True)
        run = allocate_run_dir(root, digits=4)
        work_dir = run.run_dir
        input_dir = work_dir / "inputs"
        logs_dir = work_dir / "logs"
        tmp_npz_dir = work_dir / "tmp_npz"
        preview_dir = work_dir / "preview"
        input_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        tmp_npz_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

        src = Path(upload_path)
        dst = input_dir / _safe_filename(src.name)
        shutil.copy2(src, dst)

        metadata_path = work_dir / "run_metadata.json"
        metadata = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
            "work_dir": str(work_dir),
            "input": {
                "original_upload_path": str(src),
                "copied_input_path": str(dst),
                "filename": src.name,
            },
            "paths": {
                "logs_dir": str(logs_dir),
                "outputs_dir": str(work_dir),
                "tmp_npz_dir": str(tmp_npz_dir),
                "full_log_path": str(logs_dir / "run_full.log"),
            },
            "stages": {},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return str(dst), str(work_dir)

    def _normalize_model3d_path(path_value: str):
        if not path_value:
            return None
        try:
            path = Path(path_value).resolve()
            if not path.exists() or not path.is_file():
                return None
            return path.as_posix()
        except Exception:
            return None

    def _save_preview_metadata(work_dir: str, preview_path: str, preview_note: str):
        metadata_path = Path(work_dir) / "run_metadata.json"
        try:
            metadata = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("input", {})
            if preview_path:
                metadata["input"]["preview_path"] = preview_path
            if preview_note:
                metadata["input"]["preview_note"] = preview_note
            metadata["last_updated_at"] = datetime.now().isoformat()
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _build_preview_asset(copied_path: str, work_dir: str):
        suffix = Path(copied_path).suffix.lower()
        previewable_suffixes = {".glb", ".gltf", ".obj", ".ply", ".stl"}
        if suffix not in previewable_suffixes:
            return None, f"Preview unavailable for '{suffix}' in Model3D."

        # Fallback: direct uploaded mesh path.
        direct_preview = _normalize_model3d_path(copied_path)
        note = "Preview source: uploaded mesh"

        # Compatibility conversion: export a lightweight GLB preview.
        # Heavy meshes can fail/blank in browser WebGL viewers, so we cap face count.
        try:
            import trimesh

            scene = trimesh.load(copied_path, force="scene")
            if isinstance(scene, trimesh.Scene):
                mesh = scene.dump(concatenate=True)
            elif isinstance(scene, trimesh.Trimesh):
                mesh = scene
            else:
                mesh = None

            if mesh is None or len(mesh.faces) == 0:
                return direct_preview, "Preview conversion skipped; using uploaded mesh."

            target_faces = 120_000
            original_faces = int(len(mesh.faces))
            simplified_faces = original_faces

            if original_faces > target_faces:
                simplified_mesh = None
                try:
                    # Optional dependency path (if available).
                    simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
                except Exception:
                    simplified_mesh = None

                if simplified_mesh is None or len(simplified_mesh.faces) == 0:
                    # Deterministic fallback: subsample faces for viewer-friendly preview.
                    step = max(1, original_faces // target_faces)
                    simplified_mesh = trimesh.Trimesh(
                        vertices=mesh.vertices.copy(),
                        faces=mesh.faces[::step].copy(),
                        process=False,
                    )
                    simplified_mesh.remove_unreferenced_vertices()

                mesh = simplified_mesh
                simplified_faces = int(len(mesh.faces))

            preview_dir = Path(work_dir) / "preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_glb = preview_dir / f"{Path(copied_path).stem}_preview.glb"
            exported = trimesh.Scene(mesh).export(file_type="glb")
            if isinstance(exported, (bytes, bytearray)):
                preview_glb.write_bytes(bytes(exported))
                converted_preview = _normalize_model3d_path(str(preview_glb))
                if converted_preview:
                    return (
                        converted_preview,
                        f"Preview source: converted {preview_glb.name} ({original_faces}→{simplified_faces} faces)",
                    )
        except Exception as e:
            note = f"Preview conversion failed ({type(e).__name__}); using uploaded mesh."

        return direct_preview, note

    def store_mesh(file):
        path = _uploaded_file_to_path(file)
        if not path:
            return (
                None,
                gr.update(value=None),
                "Upload a mesh and click 'Generate Skeleton' to begin...",
                None,
            )
        if not os.path.exists(path):
            return (
                None,
                gr.update(value=None),
                "❌ Uploaded mesh file was not found. Please upload again.",
                None,
            )

        try:
            copied_path, work_dir = _create_upload_workspace(path)
        except Exception as e:
            return (
                None,
                gr.update(value=None),
                f"❌ Failed to prepare upload workspace: {type(e).__name__}: {e}",
                None,
            )

        preview_path, preview_note = _build_preview_asset(copied_path, work_dir)
        _save_preview_metadata(work_dir, preview_path, preview_note)
        suffix = Path(copied_path).suffix.lower()

        if preview_path:
            status = (
                f"✅ Mesh uploaded: {Path(copied_path).name}\n"
                f"Workspace: {work_dir}\n"
                f"Viewer path: {preview_path}\n"
                f"{preview_note}\n"
                "Preview loaded. Click 'Generate Skeleton' to begin."
            )
        else:
            status = (
                f"✅ Mesh uploaded: {Path(copied_path).name}\n"
                f"Workspace: {work_dir}\n"
                f"{preview_note}\n"
                "You can still run rigging."
            )
        return (copied_path, gr.update(value=preview_path), status, work_dir)
    
    mesh_upload.change(
        fn=store_mesh,
        inputs=[mesh_upload],
        outputs=[original_mesh_state, rigged_model_preview, rig_status, upload_run_dir_state],
        queue=False,
        show_progress="hidden"
    )
    
    def auto_skin_after_skeleton(enable_skin: bool, skeleton_path: str, seed: int, prior_status: str, req: gr.Request):
        """Automatically run skinning after skeleton generation when enabled."""
        base_status = prior_status or ""

        if not skeleton_path:
            msg = "❌ Skeleton generation did not produce an output. Auto-skinning skipped."
            yield (None, None, (base_status + "\n" + msg).strip() if base_status else msg)
            return

        if not enable_skin:
            msg = "ℹ️ Auto-skinning is disabled. Skeleton was saved."
            yield (None, None, (base_status + "\n" + msg).strip() if base_status else msg)
            return

        # Stream the underlying skinning stage and keep previous logs visible.
        for skinned_path, preview_path, skin_status in run_skinning_fn(skeleton_path, seed, req):
            if skin_status:
                merged_status = (base_status + "\n" + skin_status).strip() if base_status else skin_status
            else:
                merged_status = base_status
            yield (skinned_path, preview_path, merged_status)

    # Generate Skeleton (+ optional auto-skinning chained after skeleton success)
    _skeleton_evt = generate_skeleton_btn.click(
        fn=run_skeleton_fn,
        inputs=[original_mesh_state, rig_seed, upload_run_dir_state],
        outputs=[skeleton_path_state, rigged_model_preview, rig_status]
    )
    _skeleton_evt.then(
        fn=auto_skin_after_skeleton,
        inputs=[enable_skinning, skeleton_path_state, rig_seed, rig_status],
        outputs=[skinned_path_state, rigged_model_preview, rig_status],
    )
    
    # Add Skinning
    add_skinning_btn.click(
        fn=run_skinning_fn,
        inputs=[skeleton_path_state, rig_seed],
        outputs=[skinned_path_state, rigged_model_preview, rig_status]
    )
    
    # Export Rigged Model
    def prepare_export(skinned_path, skeleton_path, original_mesh, export_fmt, export_both, auto_merge_enabled, req):
        """Determine which file to use for merge and call merge function."""
        # Use skinned if available, otherwise skeleton
        source = skinned_path if skinned_path else skeleton_path
        
        if not source:
            return (None, None, "❌ Please generate skeleton or skinning first.", gr.update(visible=False))
        
        if not original_mesh:
            return (None, None, "❌ Original mesh not found.", gr.update(visible=False))
        
        if auto_merge_enabled:
            chosen_output = None
            chosen_download = None

            def _download_update(path):
                if path:
                    return gr.update(visible=True, value=path)
                return gr.update(visible=False)

            # Primary format export (selected by user).
            for output in run_merge_fn(source, original_mesh, export_fmt, req):
                out_path, out_download, out_status = output
                if out_path and out_download and chosen_output is None:
                    chosen_output = out_path
                    chosen_download = out_download
                yield (chosen_output, chosen_download, out_status, _download_update(chosen_download))

            # Secondary format export (optional) so both FBX+GLB are saved.
            if export_both:
                other_fmt = "glb" if export_fmt == "fbx" else "fbx"
                for output in run_merge_fn(source, original_mesh, other_fmt, req):
                    out_path, out_download, out_status = output
                    if out_path and out_download and chosen_output is None:
                        chosen_output = out_path
                        chosen_download = out_download
                    yield (chosen_output, chosen_download, out_status, _download_update(chosen_download))
        else:
            # Just export the source file without merging
            final_status = f"✅ Rigged model ready (no merge):\n{source}"
            yield (source, source, final_status, gr.update(visible=True, value=source))
    
    export_rigged_btn.click(
        fn=prepare_export,
        inputs=[
            skinned_path_state,
            skeleton_path_state,
            original_mesh_state,
            export_format,
            export_both_formats,
            auto_merge,
        ],
        outputs=[final_output_state, download_btn, rig_status, download_btn]
    )
    
    # Open Outputs Folder
    def open_outputs():
        try:
            open_folder_fn(rigging_outputs_dir)
            return "✅ Opened outputs folder"
        except Exception as e:
            return f"❌ Failed to open folder: {e}"
    
    open_outputs_btn.click(
        fn=open_outputs,
        outputs=[rig_status],
        queue=False,
        show_progress="hidden"
    )
    
    # Clear
    def clear_all():
        return (
            None,  # mesh_upload
            gr.update(value=None),  # rigged_model_preview
            "Cleared. Upload a new mesh to begin.",  # rig_status
            None,  # skeleton_path_state
            None,  # skinned_path_state
            None,  # final_output_state
            None,  # original_mesh_state
            None,  # upload_run_dir_state
            gr.update(visible=False),  # download_btn
        )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[
            mesh_upload,
            rigged_model_preview,
            rig_status,
            skeleton_path_state,
            skinned_path_state,
            final_output_state,
            original_mesh_state,
            upload_run_dir_state,
            download_btn,
        ],
        queue=False,
        show_progress="hidden"
    )

    return {
        "send_to_animation_btn": send_to_animation_btn,
        "final_output_state": final_output_state,
        "skinned_path_state": skinned_path_state,
        "skeleton_path_state": skeleton_path_state,
        "rig_status": rig_status,
    }
