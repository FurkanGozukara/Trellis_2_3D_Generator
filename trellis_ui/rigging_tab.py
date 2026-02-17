"""
Rigging Tab - UniRig integration for automatic skeleton generation and skinning.
Allows users to upload 3D models and rig them using the UniRig system.
"""
import gradio as gr
import os
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
    
    # Store uploaded mesh path
    def store_mesh(file):
        return file.name if file else None
    
    mesh_upload.change(
        fn=store_mesh,
        inputs=[mesh_upload],
        outputs=[original_mesh_state],
        queue=False,
        show_progress="hidden"
    )
    
    # Generate Skeleton
    generate_skeleton_btn.click(
        fn=run_skeleton_fn,
        inputs=[mesh_upload, rig_seed],
        outputs=[skeleton_path_state, rigged_model_preview, rig_status]
    )
    
    # Add Skinning
    add_skinning_btn.click(
        fn=run_skinning_fn,
        inputs=[skeleton_path_state, rig_seed],
        outputs=[skinned_path_state, rigged_model_preview, rig_status]
    )
    
    # Export Rigged Model
    def prepare_export(skinned_path, skeleton_path, original_mesh, export_fmt, auto_merge_enabled, req):
        """Determine which file to use for merge and call merge function."""
        # Use skinned if available, otherwise skeleton
        source = skinned_path if skinned_path else skeleton_path
        
        if not source:
            return (None, None, "❌ Please generate skeleton or skinning first.", gr.update(visible=False))
        
        if not original_mesh:
            return (None, None, "❌ Original mesh not found.", gr.update(visible=False))
        
        if auto_merge_enabled:
            for output in run_merge_fn(source, original_mesh, export_fmt, req):
                if output[0] and output[1]:  # final_output_state, download_btn_file, status, download_btn_update
                    yield (output[0], output[1], output[2], gr.update(visible=True, value=output[1]))
                else:
                    yield (output[0], output[1], output[2], gr.update(visible=False))
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
            None,  # rigged_model_preview
            "Cleared. Upload a new mesh to begin.",  # rig_status
            None,  # skeleton_path_state
            None,  # skinned_path_state
            None,  # final_output_state
            None,  # original_mesh_state
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
            download_btn,
        ],
        queue=False,
        show_progress="hidden"
    )
