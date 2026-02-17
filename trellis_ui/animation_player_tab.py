"""
Animation Player Tab - Browse and view previously rigged models.
Displays rigged models with metadata and download options.
"""
import gradio as gr
import os
import json
from pathlib import Path


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
            
            **Limitations:**
            - Gradio's Model3D component does **not** support animation playback
            - Download the model to view animations in:
              - **Blender** (recommended - free, full rigging support)
              - **Unity** (for game development)
              - **Unreal Engine** (for real-time applications)
              - **Maya/3ds Max** (professional 3D software)
            
            **Supported Formats:**
            - `.fbx` - Best for most 3D software (preserves skeleton hierarchy)
            - `.glb` - Good for web/real-time applications
            """)
    
    # State to track selected model path
    selected_model_path = gr.State(None)
    
    def load_model(model_rel_path):
        """Load selected model and extract metadata."""
        if not model_rel_path:
            return (
                None,  # viewer
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
                    None,
                    {"error": "File not found"},
                    f"❌ File not found: {model_rel_path}",
                    None,
                    gr.update(visible=False),
                )
            
            # Get file info
            file_size = full_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            file_ext = full_path.suffix
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
            
            return (
                str(full_path),  # viewer (load model)
                metadata,
                file_info,
                str(full_path),  # selected_model_path
                gr.update(visible=True, value=str(full_path)),  # download_btn
            )
        
        except Exception as e:
            return (
                None,
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
            None,  # viewer
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
