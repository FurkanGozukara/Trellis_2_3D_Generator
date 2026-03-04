import argparse
import os
import sys

# Set environment variables relative to app.py configuration
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
import o_voxel

def parse_args():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Inference CLI")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Path(s) to input image(s)")
    parser.add_argument("--output", type=str, required=True, help="Path to output GLB file")
    
    # Model and Generation Settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--randomize_seed", action="store_true", help="Randomize seed")
    parser.add_argument("--resolution", type=str, default="1024", choices=["512", "1024", "1536", "2048"], help="Generation resolution")
    parser.add_argument("--no_texture_gen", action="store_true", help="Skip texture generation")
    parser.add_argument("--max_num_tokens", type=int, default=49152, help="Max number of tokens")

    # Stage 1: Sparse Structure
    parser.add_argument("--ss_guidance_strength", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    parser.add_argument("--ss_sampling_steps", type=int, default=12)
    parser.add_argument("--ss_rescale_t", type=float, default=5.0)

    # Stage 2: Shape Generation
    parser.add_argument("--shape_slat_guidance_strength", type=float, default=7.5)
    parser.add_argument("--shape_slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--shape_slat_sampling_steps", type=int, default=12)
    parser.add_argument("--shape_slat_rescale_t", type=float, default=3.0)

    # Stage 3: Texture Generation
    parser.add_argument("--tex_slat_guidance_strength", type=float, default=1.0)
    parser.add_argument("--tex_slat_guidance_rescale", type=float, default=0.0)
    parser.add_argument("--tex_slat_sampling_steps", type=int, default=12)
    parser.add_argument("--tex_slat_rescale_t", type=float, default=3.0)

    # Export Settings
    parser.add_argument("--decimation_target", type=int, default=500000, help="Target face count for decimation")
    parser.add_argument("--texture_size", type=int, default=2048, choices=[1024, 2048, 4096], help="Texture size")
    parser.add_argument("--remesh_method", type=str, default="dual_contouring", choices=["dual_contouring", "faithful_contouring", "none"], help="Remesh method")
    parser.add_argument("--simplify_method", type=str, default="cumesh", choices=["cumesh", "meshlib"], help="Simplify method")
    parser.add_argument("--prune_invisible_faces", type=bool, default=True, help="Prune invisible faces")

    return parser.parse_args()

def main():
    args = parse_args()

    # Seed
    seed = np.random.randint(0, np.iinfo(np.int32).max) if args.randomize_seed else args.seed
    
    # Load Pipeline
    print(f"Loading pipeline (Texture Models: {not args.no_texture_gen})...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B', load_texture_models=not args.no_texture_gen)
    pipeline.cuda()

    # Load and Preprocess Images
    images = []
    for img_path in args.images:
        print(f"Processing image: {img_path}")
        img = Image.open(img_path)
        img = pipeline.preprocess_image(img)
        images.append(img)

    # Run Pipeline
    print("Running generation...")
    outputs, latents = pipeline.run(
        images,
        seed=seed,
        preprocess_image=False, # Already done
        sparse_structure_sampler_params={
            "steps": args.ss_sampling_steps,
            "guidance_strength": args.ss_guidance_strength,
            "guidance_rescale": args.ss_guidance_rescale,
            "rescale_t": args.ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": args.shape_slat_sampling_steps,
            "guidance_strength": args.shape_slat_guidance_strength,
            "guidance_rescale": args.shape_slat_guidance_rescale,
            "rescale_t": args.shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": args.tex_slat_sampling_steps,
            "guidance_strength": args.tex_slat_guidance_strength,
            "guidance_rescale": args.tex_slat_guidance_rescale,
            "rescale_t": args.tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
            "2048": "2048_cascade",
        }[args.resolution],
        return_latent=True,
        max_num_tokens=args.max_num_tokens,
        no_texture_gen=args.no_texture_gen,
    )

    # Extract GLB
    print("Extracting GLB...")
    shape_slat, tex_slat, res = latents
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]

    requested_remesh_method = str(args.remesh_method)
    remesh_method = requested_remesh_method
    remesh_fallback_reason = None
    if remesh_method == "faithful_contouring":
        try:
            import importlib

            importlib.import_module("faithcontour")
            importlib.import_module("atom3d")
        except Exception as e:
            print(
                "[warn] remesh_method='faithful_contouring' requires optional FaithC dependencies "
                f"(`faithcontour` + `atom3d`). Missing/unusable ({type(e).__name__}: {e}). "
                "Falling back to 'dual_contouring'.",
                file=sys.stderr,
            )
            remesh_method = "dual_contouring"
            remesh_fallback_reason = "missing_faithc_dependencies_precheck"
    print(
        f"[extract] remesh audit: requested={requested_remesh_method!r}, stage_input={remesh_method!r}",
        flush=True,
    )
    
    # Prune config
    to_glb_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipeline.pbr_attr_layout,
        "grid_size": res,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": args.decimation_target,
        "simplify_method": args.simplify_method,
        "texture_extraction": not args.no_texture_gen,
        "texture_size": args.texture_size,
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0,
        "remesh_method": remesh_method,
        "prune_invisible": args.prune_invisible_faces,
        "use_tqdm": True,
    }
    try:
        glb = o_voxel.postprocess.to_glb(**to_glb_kwargs)
    except Exception as e:
        can_fallback = requested_remesh_method == "faithful_contouring"
        is_missing_faithc = isinstance(e, ImportError) and ("Faithful Contouring is not installed" in str(e))
        is_oom = isinstance(e, torch.OutOfMemoryError) or (
            "out of memory" in str(e).lower() and "cuda" in str(e).lower()
        )
        if can_fallback and (is_missing_faithc or is_oom):
            fallback_method = "dual_contouring"
            if is_missing_faithc:
                warn_msg = f"{e} Falling back to remesh_method={fallback_method!r}."
                remesh_fallback_reason = "faithc_missing_during_to_glb"
            else:
                warn_msg = (
                    "faithful_contouring ran out of GPU memory. "
                    f"Retrying with remesh_method={fallback_method!r}."
                )
                remesh_fallback_reason = "oom_in_faithful_contouring"
            print(f"[warn] {warn_msg}", file=sys.stderr)
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
        print(
            f"[extract] remesh audit result: requested={requested_remesh_method!r}, "
            f"effective={effective_remesh_method!r}, "
            f"fallback_reason={(remesh_fallback_reason or 'unknown')!r}",
            flush=True,
        )
    else:
        print(
            f"[extract] remesh audit result: requested={requested_remesh_method!r}, "
            f"effective={effective_remesh_method!r}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    glb.export(args.output, extension_webp=False)
    print(f"Saved GLB to {args.output}")

if __name__ == "__main__":
    main()
