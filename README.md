![](assets/teaser.webp)

# Trellis 2 Ultimate

Trellis 2 Ultimate is a productized, workflow-focused package built on top of the official `TRELLIS.2` backend. The core TRELLIS.2 image-to-3D and texturing models are still the foundation, but this package expands them into a much larger end-user application centered around `app_premium.py`, richer export and cleanup flows, rigging and animation utilities, cloud install helpers, and a much easier dependency path.

## What this README is comparing

This document was prepared by scanning and comparing:

- Official upstream demo and backend baseline: `TRELLIS.2/app.py`, `TRELLIS.2/app_texturing.py`, and the official `TRELLIS.2/trellis2` tree
- Ultimate app surface: `app_premium.py`
- Ultimate runtime/workflow additions: `subprocess_stage.py`, `subprocess_utils.py`, `inference.py`, `trellis_ui/*`, `ultrashape_integration.py`
- Ultimate backend extensions: `trellis2/runtime_options.py`, `trellis2/pipelines/trellis2_multiview.py`, `trellis2/projection/texture_projection_multiview.py`, and modified pipeline/base files
- Package-level install and model-management helpers shipped with this build: `Windows_Install_or_Update.bat`, `Windows_Model_Download_Resume.bat`, `Windows_Start_App.bat`, `RunPod_Install_Trellis.sh`, `Massed_Compute_Install.sh`, `HF_model_downloader.py`, `requirements_trellis.txt`

Important scope note:

- This comparison is about the end-user app, inference workflow, export pipeline, rigging/animation, runtime stability, and installation experience.
- The official repo still contains research-oriented training and data-toolkit code that this package does not try to replace.
- In other words, Trellis 2 Ultimate is not a training-code superset of the official repo. It is a much larger inference application and deployment package.

## Executive summary

Official `TRELLIS.2/app.py` is a clean research demo:

- one image in
- one preview flow
- one GLB export flow
- one minimal UI tab

Trellis 2 Ultimate turns the same core model family into a broader 3D production workflow:

- a 7-tab premium app instead of a single demo tab
- single-image and multi-angle generation
- more generation resolutions and more pipeline strategies
- selectable model/runtime backends and samplers
- low-VRAM controls, staged subprocess execution, and safer cancellation
- richer extraction, cleanup, repair, re-texturing, and multi-format export
- standalone texturing, standalone UltraShape refinement, and local 3D file viewing
- integrated UniRig rigging and animation browsing
- batch processing, presets, logs, manifests, and structured numbered output folders
- Windows/cloud install helpers with precompiled wheel usage instead of forcing users into a compile-heavy setup

## At-a-glance comparison

| Area | Official TRELLIS.2 | Trellis 2 Ultimate |
| --- | --- | --- |
| Main app | Single `Image to 3D` Gradio demo | Full `app_premium.py` workflow app |
| Tabs | 1 main generation tab | 7 tabs: `Image -> 3D`, `Texturing`, `UltraShape Refine`, `View 3D Files`, `Help / Settings Guide`, `Rigging`, `Animation Player` |
| Inputs | Single image in the main demo | Single image or reordered multi-angle image set |
| Multi-view generation | Not exposed in official app | Dedicated multi-view pipeline with `stochastic` and `multidiffusion` fusion modes |
| Exposed resolutions | `512`, `1024`, `1536` | `512`, `768`, `1024`, `1280`, `1536`, `2048`, plus custom divisible-by-128 override |
| Pipeline strategies | Fixed app path | `reference_auto`, `direct_1024`, `hybrid_512g_1024t` |
| Runtime controls | Basic stage sliders | FP8 option, attention backend choice, sampler choice, max token budget, low-VRAM, chunked/tiled extraction, high-res conditioning toggle |
| Export flow | Extract one GLB | GLB plus optional `gltf`, `obj`, `ply`, `stl`, with manifests and mesh audit reports |
| Cleanup/refine | Minimal extract options | Remesh, simplify, repair, prune, shade smooth, vertex merge, force double-sided, basecolor-only export, deferred re-texture, projection-texture refine, UltraShape refine |
| Texturing | Separate minimal `app_texturing.py` | Integrated texturing tab with advanced runtime controls and logs |
| Rigging / animation | Not present | UniRig skeleton, skinning, merge/export, animation browser, animated preview generation |
| Batch / presets | Not present | Folder batch processing, skip-if-exists resume behavior, built-in and custom presets |
| Output management | Session temp files | Structured numbered run folders, logs, run metadata, final export manifests |
| Install experience | Linux-first, compile-heavy setup path | Windows starter scripts, cloud install scripts, resumable downloader, precompiled dependency strategy |

## Official baseline: what the upstream app actually gives you

The official `TRELLIS.2/app.py` is intentionally simple. It exposes:

- one uploaded RGBA image
- `512`, `1024`, or `1536` resolution
- seed and randomize-seed
- decimation target
- texture size
- basic Stage 1 / Stage 2 / Stage 3 sampling sliders
- `Generate`
- `Extract GLB`
- a preview walkthrough

The official repo also ships a separate `app_texturing.py`, but that is also a focused demo: upload a mesh, upload a reference image, set a few texturing sliders, generate a textured GLB, download it.

That is a good research demo surface. Trellis 2 Ultimate goes far beyond that.

## 1. Premium multi-tab application instead of a single demo

The Ultimate UI is centered around `app_premium.py`, which is far larger and more workflow-oriented than the official app surface.

Included tabs:

- `Image -> 3D`: the main generation and extraction workflow
- `Texturing`: texture an existing mesh from an image
- `UltraShape Refine`: refine an existing coarse mesh against a reference image
- `View 3D Files`: local viewer for already-existing 3D assets
- `Help / Settings Guide`: in-app operating guide for users
- `Rigging`: UniRig skeleton + skinning workflow
- `Animation Player`: browse rigged outputs and generate animation previews

This alone is a major productization step over the official single-task demo.

## 2. Image -> 3D generation improvements

### Multi-image / multi-angle generation

The official app is single-image only.

Ultimate adds:

- multiple uploaded input images in one run
- drag-reordering of input views
- explicit primary-view behavior
- dedicated multi-view fusion modes:
  - `stochastic`
  - `multidiffusion`

This is backed by the added `trellis2/pipelines/trellis2_multiview.py` pipeline rather than just UI sugar.

### More resolution options

Official app UI:

- `512`
- `1024`
- `1536`

Ultimate app UI:

- `512`
- `768`
- `1024`
- `1280`
- `1536`
- `2048`
- custom resolution override when divisible by `128`

This gives users finer control over quality, VRAM pressure, and turnaround time.

### More pipeline strategies

Ultimate exposes three image-to-3D pipeline strategies:

- `reference_auto`
- `direct_1024`
- `hybrid_512g_1024t`

What that means in practice:

- users can stay on the high-quality default path
- users can force the more direct `1024` path
- users can use a lower-memory hybrid path with `512` geometry plus `1024` texturing when VRAM is tighter

### More runtime and model controls

Ultimate surfaces controls that are not exposed in the official app UI:

- `standard` and `fp8` model variant selection
- attention backend selection:
  - `auto`
  - `flash_attn`
  - `flash_attn_3`
  - `xformers`
  - `sdpa`
- sampler selection:
  - `heun`
  - `euler`
  - `rk4`
  - `rk5`
- max token budget control for higher resolutions
- force high-resolution conditioning toggle
- low-VRAM mode
- chunked triangle processing for generation
- tiled mesh extraction for generation
- separate chunked/tiled toggles for final extraction

The official backend already contains some lower-level capability in the model stack, but Ultimate is what actually exposes and orchestrates these knobs as an end-user workflow.

### Better session control and quality-of-life features

Ultimate adds:

- input preview switching for one image vs galleries for many images
- top-level logs and output-folder buttons
- `View Extracted` step jump
- fullscreen extracted-view control
- in-app guide text for settings

These are workflow improvements the official demo does not try to provide.

## 3. Batch processing and presets

Official TRELLIS.2 app:

- no folder batch mode
- no settings presets

Ultimate adds both.

### Batch processing

Users can point the app at a folder of images and run the same configured pipeline over all of them.

Verified capabilities:

- input folder selection
- optional custom output folder
- supported image extension filtering
- per-file seed handling
- ETA / processed / skipped / failed progress reporting
- safe resume behavior:
  - if the target output folder for an image already exists, that image is skipped
- per-image output directories
- optional preview skipping during batch to save time and memory

### Presets

Ultimate has a real preset system stored under `presets/`.

It supports:

- save current UI settings
- load selected preset
- reset to defaults
- delete custom presets
- remember the last used preset
- built-in presets:
  - `best`
  - `low_vram`

Presets cover settings across:

- `Image -> 3D`
- `Texturing`
- `UltraShape Refine`
- `Rigging`

That is much closer to a real user product than a demo UI.

## 4. Extraction, cleanup, and export pipeline upgrades

The official app's extract path is simple: decode the latent, run `to_glb`, export one GLB.

Ultimate turns extraction into a much more configurable post-processing pipeline.

### More extraction controls

Ultimate exposes:

- remesh method selection
- simplify method selection
- repair method selection
- hole-fill perimeter threshold
- invisible-face pruning
- merge-vertices distance
- shade-smooth toggle
- shade-smooth angle
- force double-sided materials
- basecolor-only material export
- texture-size control
- skip-texture-generation mode

### Multiple remesh / simplify / repair paths

Supported and conditionally exposed options include:

- remesh:
  - `dual_contouring`
  - `dual_contouring_vb` when available from the CuMesh build
  - `faithful_contouring` when FaithC dependencies are available
- simplify:
  - `cumesh`
  - `meshlib`
  - `none`
- repair:
  - `disabled`
  - `cumesh`
  - `meshlib`
  - `pymeshfix` when its full runtime stack is installed

Ultimate also contains fallback handling so unsupported or failing paths can fall back more safely, especially around `faithful_contouring`.

### Deferred texture rebuild after cleanup

Ultimate can:

- first clean/remesh/simplify the extracted mesh
- then run a separate final TRELLIS texturing pass on that cleaned mesh

Why this matters:

- the mesh users keep is often not the raw direct extract
- remesh/simplify can change the surface enough that a deferred final texture bake produces a better alignment than simply keeping the original stage texture

### Projection texture refinement from known views

This is a major addition.

Ultimate adds a projection-based texture path in `trellis2/projection/texture_projection_multiview.py` and exposes it in the UI.

Users can provide:

- multiple input views
- per-view azimuths
- per-view elevations
- blend exponent
- orthographic scale
- projection hole filling
- maximum hole size

This is useful when users actually know the view order/angles and want the final mesh texture to be projected from those real images instead of relying only on the TRELLIS texture latent.

### Multi-format export

Official app is centered on one GLB export.

Ultimate can write:

- `glb`
- `gltf`
- `obj`
- `ply`
- `stl`

This is especially useful for:

- DCC cleanup
- game-tool ingestion
- mesh inspection
- pipelines that need non-GLB interchange

### Structured export metadata

Ultimate writes export-side metadata files the official demo does not:

- `extract_artifacts.json`
- `mesh_audit.json`

These document:

- which artifacts were created
- which file is the final output
- whether PBR textures are present
- whether remesh fallback happened
- artifact roles such as intermediate shape-only output vs final re-textured output
- mesh statistics for final and intermediate outputs

This is useful for debugging, repeatability, and automation.

## 5. Structured run folders, logs, and reproducibility

Official app writes session outputs into a temp-style folder with timestamped file names.

Ultimate organizes work into numbered run folders such as:

- `outputs/0001/`
- `outputs/0002/`

Per-run files include, depending on workflow:

- raw inputs
- preprocessed inputs
- intermediate condition files / latent files
- preview assets
- `run.json`
- run logs
- `08_final_exports/`
- `09_retexture_work/`

This is a serious operational upgrade because users can:

- inspect what happened after a failure
- keep full per-run history
- revisit and re-export later
- understand which stage produced which file

## 6. Dedicated texturing workflow upgrades

Official repo has a separate lightweight `app_texturing.py`.

Ultimate keeps texturing, but expands it significantly inside the premium app:

- integrated `Texturing` tab
- mesh upload plus reference image
- low-VRAM toggle
- selectable attention backend
- selectable sampler
- full texturing guidance interval controls
- logs visibility toggle
- cancel button
- open-output-folder action
- example assets for testing

This makes the texturing path feel like a first-class workflow, not an isolated demo.

## 7. UltraShape integration

UltraShape support is one of the biggest functional additions in the package.

Ultimate adds:

- `ultrashape_integration.py`
- a dedicated `UltraShape Refine` tab
- an optional UltraShape refinement stage inside `Extract GLB`

### What UltraShape adds

UltraShape is used here as an image-guided mesh refinement step on top of coarse or already-generated geometry.

Ultimate exposes:

- checkpoint selection
- config selection
- dtype selection
- low-VRAM mode
- background removal for the reference image
- diffusion steps
- guidance scale
- octree resolution
- chunk size
- target face count
- latent count
- box and marching-cubes thresholds
- normalization scale
- sharp-point and uniform-point sampling controls

### Two Ultimate workflows using UltraShape

Ultimate supports both:

- standalone refinement of an existing uploaded mesh
- refinement inserted into the main Image -> 3D extraction path

It also supports:

- conservative mode to reduce geometry drift
- optional TRELLIS re-texturing after UltraShape changes

That is a very large step beyond the official TRELLIS.2 demo experience.

## 8. Rigging and animation features via UniRig

Official TRELLIS.2 has no rigging or animation browser.

Ultimate adds a real rigging workflow through UniRig.

### Rigging tab

The `Rigging` tab supports:

- mesh upload for `.obj`, `.fbx`, `.glb`, `.vrm`
- skeleton generation
- optional skinning
- merge back with original mesh/materials
- export as `fbx` or `glb`
- export both formats
- animated, textured, and skeleton preview tabs
- download button
- output-folder access
- send-to-animation-browser handoff

### Animation Player tab

The `Animation Player` tab supports:

- browsing previously rigged outputs
- importing external rigged models into the browser
- metadata display
- file info display
- animated preview
- textured preview
- skeleton preview
- animation preview generation controls:
  - style: `dance`, `walk`, `idle`
  - strength
  - frame count

This turns Trellis 2 Ultimate into something much closer to a complete 3D asset workflow rather than only a generator.

## 9. View 3D Files utility tab

Official app assumes you are always generating a fresh model.

Ultimate adds a `View 3D Files` tab for direct local preview of:

- `.glb`
- `.gltf`
- `.obj`
- `.ply`
- `.stl`

This is small compared with the bigger features above, but it matters for real users because it means the app can also work as a lightweight local inspection tool.

## 10. Runtime engineering and stability improvements

One of the biggest practical upgrades is not just new features, but how the app manages memory and failures.

### Subprocess stage processing

Ultimate exposes a global option:

- `Subprocess stage processing (zero leftover VRAM between stages)`

When enabled, major stages run in fresh Python workers. This gives users:

- less leftover VRAM between stages
- better odds of surviving long sessions
- easier recovery from CUDA memory fragmentation
- a real process to terminate when canceling

This is a strong operational improvement over a simple monolithic demo execution flow.

### Safer cancellation

Ultimate implements:

- two-step cancel confirmation
- cancellation state tracking
- subprocess termination for active worker stages
- separate batch-cancel handling when subprocess mode is off

### Better runtime switching

Ultimate adds runtime orchestration around:

- attention backend availability detection
- dense/sparse attention backend matching
- sampler switching for image-to-3D
- sampler switching for texturing

This logic lives in `trellis2/runtime_options.py` and related pipeline modifications.

### Offline-friendly local model loading

Ultimate modifies the pipeline loader so it prefers a local model mirror under `models/` using `TRELLIS_MODELS_DIR` before hitting Hugging Face.

Why this matters:

- easier offline use after the first download
- easier packaged distribution
- easier cloud persistence-volume use
- better reuse of downloaded models across runs

## 11. Dedicated CLI inference entrypoint

Ultimate adds `inference.py`, which gives a scriptable entrypoint separate from the web UI.

It supports:

- one or more input images
- output path selection
- image-to-3D generation settings
- extract settings
- remesh and simplify selection
- direct file export

This is useful for:

- automation
- integration into larger toolchains
- headless or scripted usage

## 12. Installation and deployment improvements

This is one of the clearest places where Trellis 2 Ultimate goes beyond the official package.

The official README is Linux-first and expects users to run a compile-oriented `setup.sh` path with CUDA toolkit requirements.

Ultimate ships a much more packaging-oriented install story.

### Windows package scripts

Companion scripts shipped with this build include:

- `Windows_Install_or_Update.bat`
- `Windows_Model_Download_Resume.bat`
- `Windows_Start_App.bat`

What the Windows install script does:

- clones `Trellis_2_3D_Generator`
- clones `UniRig`
- creates a Python `3.11` virtual environment
- installs `uv`
- installs the dependency stack from `requirements_trellis.txt`
- installs additional wheel-only heavy packages such as:
  - `flex_gemm`
  - `o_voxel`
  - `spconv`
- clones and installs `FaithC`
- launches the model downloader

What the Windows start script does:

- activates the app virtual environment
- sets useful runtime environment variables
- points `HF_HOME` to `models`
- launches `app_premium.py`

What the resume script does:

- reruns the downloader without forcing a full restart
- lets interrupted model downloads continue

### Cloud / remote install scripts

Ultimate also ships:

- `RunPod_Install_Trellis.sh`
- `Massed_Compute_Install.sh`

These scripts:

- create a Python `3.11` venv
- install `uv`
- install the app requirements
- install wheel-based heavy CUDA packages
- install `FaithC`
- run `HF_model_downloader.py`

There are also companion instruction files for these environments:

- `RunPod_SimplePod_Trellis_Instructions_READ.txt`
- `Massed_Compute_Instructions_READ.txt`

### Why this is a real upgrade over official installation

Compared with the official TRELLIS.2 installation path, this package explicitly tries to reduce user pain by:

- targeting Windows as well as Linux/cloud usage
- favoring wheel installs over local compilation when possible
- bundling start/resume scripts
- bundling a model downloader that understands resume and verification
- making persistent local/offline model folders part of the design

## 13. Precompiled library strategy

The package does not just add scripts. It also adds an installation strategy built around precompiled wheel delivery for many of the heavy dependencies that usually hurt users the most.

### Heavy wheel-based packages referenced by `requirements_trellis.txt`

Attention and runtime acceleration:

- `flash_attn`
- `xformers`
- `sageattention`
- `triton` / `triton-windows`

Geometry, rasterization, sparse ops, and core CUDA stack:

- `nvdiffrast`
- `cumesh`
- `cumm`
- `torch_scatter`
- `torch_cluster`
- `cubvh`
- `atom3d`

Packages installed explicitly by the helper scripts:

- `flex_gemm`
- `o_voxel`
- `spconv`

Optional contouring support installed from source repo:

- `FaithC`

Why this matters:

- far fewer users have to build these packages locally
- Windows installs are much more realistic
- cloud instances come up faster
- support burden is lower because the dependency path is more standardized

## 14. Model downloader and offline model management

`HF_model_downloader.py` is another major quality-of-life upgrade.

### What it does

It downloads models into:

- `Trellis_2_3D_Generator/models`

It supports:

- mirrored bundled model packs
- UltraShape bundle download
- DINOv2 bundle download
- direct official repo download paths
- BiRefNet download
- optional Hugging Face token usage
- SHA256 verification
- verified-file caching
- resumable downloads
- multi-connection downloading

### Verified downloader behavior

The downloader is built around:

- `16` connections
- HTTP range support when available
- retry and backoff logic
- `sha256_cache.json`
- `verified_files_cache.json`

This is far better than a "download everything again from scratch" experience.

### Why offline loading works better here

The Ultimate pipeline loader prefers local model mirrors under `models/<org>--<repo>/...`.

That means once the model store is populated:

- the app can reuse it
- cloud persistent volumes are more useful
- repeated runs do not have to rediscover everything from the network

## 15. The practical result

The official TRELLIS.2 repo gives you the core research implementation and demo surface.

Trellis 2 Ultimate turns that into a broader user-facing application with:

- more generation paths
- more export paths
- more recovery paths
- more memory-management paths
- more deployment paths
- more workflow stages after generation

That includes features the official app simply does not attempt to cover:

- multi-angle fusion
- batch processing
- preset management
- deferred re-texturing
- projection-based texture refinement
- UltraShape refinement
- UniRig rigging
- animation preview browsing
- Windows start/install scripts
- resumable model downloading
- precompiled heavy dependency strategy

## Bottom line

If the official `TRELLIS.2` app is the research demo, Trellis 2 Ultimate is the workflow build.

It keeps the official TRELLIS.2 backbone, but adds the layers real users usually ask for:

- easier installation
- better VRAM survivability
- more export control
- multi-view handling
- better post-processing
- rigging and animation utilities
- batchability
- offline-friendly model management

That is the real value of this package.
