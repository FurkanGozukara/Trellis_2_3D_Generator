from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image
import os


def _resolve_hf_repo_or_path(model_name: str) -> str:
    """
    If `model_name` is a HF repo id and a local copy exists under TRELLIS_MODELS_DIR,
    return that local folder path. Otherwise return the original string.
    """
    if not isinstance(model_name, str):
        return model_name

    models_dir = os.environ.get("TRELLIS_MODELS_DIR")
    if not models_dir:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        if not os.path.isdir(models_dir):
            models_dir = None
    if not models_dir:
        return model_name

    # If the caller already passed a local path, keep it.
    if os.path.exists(model_name):
        return model_name

    if "/" not in model_name:
        return model_name

    parts = model_name.split("/")
    if len(parts) < 2:
        return model_name

    repo_id = f"{parts[0]}/{parts[1]}"
    local_repo_dir = os.path.join(models_dir, repo_id.replace("/", "--"))
    if os.path.isdir(local_repo_dir):
        return local_repo_dir

    return model_name


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Keep torch.hub downloads inside our models folder if configured.
        models_dir = os.environ.get("TRELLIS_MODELS_DIR")
        if not models_dir:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            if not os.path.isdir(models_dir):
                models_dir = None
        if models_dir:
            try:
                torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))
            except Exception:
                pass
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        resolved_name = _resolve_hf_repo_or_path(model_name)
        self.model = DINOv3ViTModel.from_pretrained(resolved_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.extract_features(image)
        return features
