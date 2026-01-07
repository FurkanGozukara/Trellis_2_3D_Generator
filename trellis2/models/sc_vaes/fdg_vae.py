from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)
from ...representations import Mesh
from o_voxel.convert import flexible_dual_grid_to_mesh, tiled_flexible_dual_grid_to_mesh


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
    
    
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        self.use_tiled_extraction = False
        self.use_chunked_processing = False
        
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution
    
    def set_tiled_extraction(self, use_tiled_extraction: bool) -> None:
        self.use_tiled_extraction = use_tiled_extraction
    
    def set_chunked_processing(self, use_chunked_processing: bool) -> None:
        self.use_chunked_processing = use_chunked_processing
        
    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs):
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            # In training mode, always use standard extraction (tiled not supported in training)
            mesh = [Mesh(flexible_dual_grid_to_mesh(
                h.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True,
                use_chunked_processing=self.use_chunked_processing
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected = h.replace(h.feats[..., 3:6] > 0)
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            # Choose extraction method based on settings
            if self.use_tiled_extraction:
                # Tiled extraction for reduced memory usage
                mesh = [Mesh(*tiled_flexible_dual_grid_to_mesh(
                    h.coords[:, 1:], v.feats, i.feats, q.feats,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    grid_size=self.resolution,
                    tile_size=128,
                    overlap=1,
                    train=False
                )) for v, i, q in zip(vertices, intersected, quad_lerp)]
            else:
                # Standard extraction with optional chunked processing
                mesh = [Mesh(*flexible_dual_grid_to_mesh(
                    h.coords[:, 1:], v.feats, i.feats, q.feats,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    grid_size=self.resolution,
                    train=False,
                    use_chunked_processing=self.use_chunked_processing
                )) for v, i, q in zip(vertices, intersected, quad_lerp)]
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
