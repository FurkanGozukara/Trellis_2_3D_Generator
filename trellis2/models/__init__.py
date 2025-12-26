import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _resolve_local_model_path(path: str) -> str:
    """
    Resolve a Hugging Face-style model path (e.g. "org/repo/subpath/to/ckpt")
    to a local path inside TRELLIS_MODELS_DIR if available.

    This lets the code run fully offline once model repos are downloaded into:
      TRELLIS_MODELS_DIR/<org>--<repo>/...

    Note: this function is intentionally conservative; it only rewrites when the
    expected local files exist.
    """
    import os

    models_dir = os.environ.get("TRELLIS_MODELS_DIR")
    if not models_dir:
        # Default to ../models relative to this repo layout: Trellis_2_3D_Generator/models
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        if not os.path.isdir(models_dir):
            return path

    # If the given path already points to local files, keep it.
    if os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors"):
        return path

    # Map "org/repo/..." -> "<models_dir>/org--repo/..."
    parts = path.split("/")
    if len(parts) < 3:
        return path  # Not a trellis2-style hf path.

    repo_id = f"{parts[0]}/{parts[1]}"
    subpath = "/".join(parts[2:])
    local_repo_dir = os.path.join(models_dir, repo_id.replace("/", "--"))
    local_path = os.path.join(local_repo_dir, subpath)

    if os.path.exists(f"{local_path}.json") and os.path.exists(f"{local_path}.safetensors"):
        return local_path

    return path


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file

    path = _resolve_local_model_path(path)
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
