"""
Multi-view extension for TRELLIS.2 Image-to-3D Pipeline.

This adds tuning-free multi-view aggregation during denoising:
- stochastic: cycle conditioning views over steps
- multidiffusion: average predictions across all views each step
"""

from typing import *
from contextlib import contextmanager
import numpy as np
import torch
from PIL import Image

from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from .samplers import FlowEulerSampler
from ..representations import MeshWithVoxel


class Trellis2MultiViewPipeline(Trellis2ImageTo3DPipeline):
    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ):
        old_inference_model = sampler._inference_model
        setattr(sampler, "_old_inference_model", old_inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images ({num_images}) is greater "
                    f"than number of steps ({num_steps}). This may reduce quality.\033[0m"
                )

            cond_indices = (np.arange(num_steps) % max(1, num_images)).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0) if len(cond_indices) > 0 else 0
                cond_i = cond[cond_idx : cond_idx + 1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":

            def _new_inference_model(self, model, x_t, t, cond, neg_cond=None, guidance_strength=1.0, **kwargs):
                guidance_interval = kwargs.pop("guidance_interval", (0.0, 1.0))
                kwargs.pop("guidance_rescale", None)
                in_guidance_interval = guidance_interval[0] <= t <= guidance_interval[1]

                preds = []
                for i in range(len(cond)):
                    cond_i = cond[i : i + 1]
                    pred = FlowEulerSampler._inference_model(self, model, x_t, t, cond_i, **kwargs)
                    preds.append(pred)
                pred_avg = sum(preds) / len(preds)

                if in_guidance_interval and guidance_strength != 1 and neg_cond is not None:
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return guidance_strength * pred_avg + (1 - guidance_strength) * neg_pred
                return pred_avg

        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'stochastic' or 'multidiffusion'.")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
        try:
            yield
        finally:
            sampler._inference_model = old_inference_model
            delattr(sampler, "_old_inference_model")

    def get_cond_multi(self, images: List[Image.Image], resolution: int, include_neg_cond: bool = True) -> dict:
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(images)
        if self.low_vram:
            self.image_cond_model.cpu()

        if not include_neg_cond:
            return {"cond": cond}

        neg_cond = torch.zeros_like(cond[:1])
        return {"cond": cond, "neg_cond": neg_cond}

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
        no_texture_gen: bool = False,
    ) -> List[MeshWithVoxel]:
        if len(images) < 2:
            print("Warning: run_multi_image called with fewer than 2 images. Consider using run() instead.")

        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == "512":
            assert "shape_slat_flow_model_512" in self.models
            if not no_texture_gen:
                assert "tex_slat_flow_model_512" in self.models
        elif pipeline_type == "1024":
            assert "shape_slat_flow_model_1024" in self.models
            if not no_texture_gen:
                assert "tex_slat_flow_model_1024" in self.models
        elif pipeline_type in {"1024_cascade", "1536_cascade", "2048_cascade"}:
            assert "shape_slat_flow_model_512" in self.models
            assert "shape_slat_flow_model_1024" in self.models
            if not no_texture_gen:
                assert "tex_slat_flow_model_1024" in self.models
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]

        torch.manual_seed(seed)
        num_images = len(images)

        cond_512 = self.get_cond_multi(images, 512)
        cond_1024 = self.get_cond_multi(images, 1024) if pipeline_type != "512" else None

        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get("steps", 12)
        shape_steps = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}.get("steps", 12)
        tex_steps = {**self.tex_slat_sampler_params, **tex_slat_sampler_params}.get("steps", 12)

        ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32, "2048_cascade": 32}[pipeline_type]

        with self.inject_sampler_multi_image(self.sparse_structure_sampler, num_images, ss_steps, mode):
            coords = self.sample_sparse_structure(cond_512, ss_res, num_samples, sparse_structure_sampler_params)

        if pipeline_type == "512":
            with self.inject_sampler_multi_image(self.shape_slat_sampler, num_images, shape_steps, mode):
                shape_slat = self.sample_shape_slat(
                    cond_512,
                    self.models["shape_slat_flow_model_512"],
                    coords,
                    shape_slat_sampler_params,
                )
            res = 512
        elif pipeline_type == "1024":
            with self.inject_sampler_multi_image(self.shape_slat_sampler, num_images, shape_steps, mode):
                shape_slat = self.sample_shape_slat(
                    cond_1024,
                    self.models["shape_slat_flow_model_1024"],
                    coords,
                    shape_slat_sampler_params,
                )
            res = 1024
        else:
            target_res = int(pipeline_type.split("_")[0])
            with self.inject_sampler_multi_image(self.shape_slat_sampler, num_images, shape_steps * 2, mode):
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512,
                    cond_1024,
                    self.models["shape_slat_flow_model_512"],
                    self.models["shape_slat_flow_model_1024"],
                    512,
                    target_res,
                    coords,
                    shape_slat_sampler_params,
                    max_num_tokens,
                )

        if no_texture_gen:
            tex_slat = None
        else:
            tex_cond = cond_512 if pipeline_type == "512" else cond_1024
            tex_model = self.models["tex_slat_flow_model_512"] if pipeline_type == "512" else self.models["tex_slat_flow_model_1024"]
            with self.inject_sampler_multi_image(self.tex_slat_sampler, num_images, tex_steps, mode):
                tex_slat = self.sample_tex_slat(tex_cond, tex_model, shape_slat, tex_slat_sampler_params)

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh
