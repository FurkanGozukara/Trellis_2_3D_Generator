from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import os


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # Prefer a local copy if it exists under TRELLIS_MODELS_DIR/<org>--<repo>/
        resolved_name = model_name
        models_dir = os.environ.get("TRELLIS_MODELS_DIR")
        if not models_dir:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
            if not os.path.isdir(models_dir):
                models_dir = None
        if models_dir and isinstance(model_name, str) and "/" in model_name and not os.path.exists(model_name):
            parts = model_name.split("/")
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                local_repo_dir = os.path.join(models_dir, repo_id.replace("/", "--"))
                if os.path.isdir(local_repo_dir):
                    resolved_name = local_repo_dir

        self.model = AutoModelForImageSegmentation.from_pretrained(
            resolved_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    