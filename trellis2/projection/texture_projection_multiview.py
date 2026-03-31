from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def _parse_float_csv(value: str) -> List[float]:
    if not value or not str(value).strip():
        return []
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def default_projection_views(view_count: int) -> Tuple[List[float], List[float]]:
    count = max(1, int(view_count))
    presets = {
        1: ([0.0], [0.0]),
        2: ([0.0, 180.0], [0.0, 0.0]),
        3: ([0.0, 90.0, 270.0], [0.0, 0.0, 0.0]),
        4: ([0.0, 180.0, 90.0, 270.0], [0.0, 0.0, 0.0, 0.0]),
        5: ([0.0, 180.0, 90.0, 270.0, 0.0], [0.0, 0.0, 0.0, 0.0, 90.0]),
        6: ([0.0, 180.0, 90.0, 270.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 90.0, -90.0]),
    }
    if count in presets:
        return presets[count]
    azimuths = [float((idx * (360.0 / count)) % 360.0) for idx in range(count)]
    elevations = [0.0] * count
    return azimuths, elevations


def resolve_projection_views(
    image_paths: Sequence[Path | str],
    azimuth_text: str = "",
    elevation_text: str = "",
) -> Tuple[List[float], List[float]]:
    view_count = len(list(image_paths))
    azimuths = _parse_float_csv(azimuth_text)
    elevations = _parse_float_csv(elevation_text)
    if not azimuths and not elevations:
        return default_projection_views(view_count)
    if len(azimuths) != view_count or len(elevations) != view_count:
        raise ValueError(
            f"Projection view angles must match the number of images ({view_count}). "
            f"Got {len(azimuths)} azimuths and {len(elevations)} elevations."
        )
    return azimuths, elevations


def get_camera_vectors(azimuth: float, elevation: float, device: str = "cuda"):
    import torch

    az = math.radians(azimuth)
    el = math.radians(elevation)

    cx = math.sin(az) * math.cos(el)
    cy = math.sin(el)
    cz = math.cos(az) * math.cos(el)

    look = torch.tensor([-cx, -cy, -cz], device=device, dtype=torch.float32)
    look = look / (look.norm() + 1e-8)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    right = torch.cross(look, world_up, dim=-1)
    if right.norm() < 1e-6:
        world_fwd = torch.tensor([0.0, 0.0, 1.0 if look[1] > 0.0 else -1.0], device=device, dtype=torch.float32)
        right = torch.cross(look, world_fwd, dim=-1)
    right = right / (right.norm() + 1e-8)

    up = torch.cross(right, look, dim=-1)
    up = up / (up.norm() + 1e-8)
    return look, right, up


def build_ortho_clip_verts(vertices, right, up, look, ortho_scale):
    import torch

    scale = max(float(ortho_scale), 1e-6)
    x = (vertices * right).sum(-1) / (scale / 2.0)
    y = (vertices * up).sum(-1) / (scale / 2.0)
    depth = -(vertices * look).sum(-1)
    d_min = depth.min()
    d_span = (depth.max() - d_min).clamp(min=1e-6)
    z = 1.0 - 2.0 * (depth - d_min) / d_span
    return torch.stack([x, y, z, torch.ones_like(x)], dim=-1).unsqueeze(0)


def texture_mesh_with_multiview(
    high_poly_mesh: "trimesh.Trimesh",
    images: Sequence["Image.Image"],
    azimuths: Sequence[float],
    elevations: Sequence[float],
    *,
    texture_size: int = 4096,
    blend_texture: bool = True,
    blend_exponent: float = 2.0,
    ortho_scale: float = 1.1,
    norm_size: float = 1.15,
    fill_holes: bool = True,
    max_hole_size: int = 20,
    use_metallic: bool = True,
    depth_eps: float = 0.01,
    low_poly_mesh: Optional["trimesh.Trimesh"] = None,
) -> Tuple["trimesh.Trimesh", "Image.Image", "Image.Image"]:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(
            "Projection texture refinement requires OpenCV (`cv2`) in the active environment."
        ) from e
    try:
        import cumesh as CuMesh
    except Exception as e:
        raise RuntimeError(
            "Projection texture refinement requires `cumesh` for UV unwrapping when meshes do not already contain UVs."
        ) from e
    try:
        import nvdiffrast.torch as dr
    except Exception as e:
        raise RuntimeError(
            "Projection texture refinement requires `nvdiffrast` in the active environment."
        ) from e

    import numpy as np
    import torch
    import torch.nn.functional as F
    import trimesh
    from PIL import Image

    if not (len(images) == len(azimuths) == len(elevations)):
        raise ValueError("images, azimuths, and elevations must have the same length")

    mesh = low_poly_mesh if low_poly_mesh is not None else high_poly_mesh
    device = "cuda"

    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        out_verts = torch.from_numpy(mesh.vertices).float().to(device)
        out_faces = torch.from_numpy(mesh.faces).int().to(device)
        out_uvs = torch.from_numpy(mesh.visual.uv).float().to(device)
        if not hasattr(mesh, "vertex_normals") or len(mesh.vertex_normals) == 0:
            mesh.face_normals = mesh.generate_face_normals()
            mesh.vertex_normals = mesh.generate_vertex_normals()
        out_normals = torch.from_numpy(mesh.vertex_normals).float().to(device)
    else:
        cumesh = CuMesh.CuMesh()
        cumesh.init(torch.from_numpy(mesh.vertices).float().to(device), torch.from_numpy(mesh.faces).int().to(device))
        out_verts, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(return_vmaps=True, verbose=True)
        out_verts = out_verts.to(device)
        out_faces = out_faces.to(device)
        out_uvs = out_uvs.to(device)
        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps.to(device)]
        out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)

    bbox_min = out_verts.min(dim=0)[0]
    bbox_max = out_verts.max(dim=0)[0]
    center = (bbox_min + bbox_max) / 2.0
    out_verts = out_verts - center
    current_max_radius = torch.sqrt((out_verts**2).sum(dim=-1).max())
    scale_factor = float(norm_size) / (current_max_radius * 2.0)
    out_verts = out_verts * scale_factor
    out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)

    ctx = dr.RasterizeCudaContext()
    uvs_clip = torch.cat(
        [out_uvs * 2.0 - 1.0, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])],
        dim=-1,
    ).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, uvs_clip, out_faces.int(), resolution=[texture_size, texture_size])
    uv_hit_mask = rast[0, :, :, 3] > 0
    tex_pos = dr.interpolate(out_verts.unsqueeze(0), rast, out_faces.int())[0][0]
    tex_normals = dr.interpolate(out_normals.unsqueeze(0), rast, out_faces.int())[0][0]
    tex_normals = tex_normals / (tex_normals.norm(dim=-1, keepdim=True) + 1e-8)

    acc_color = torch.zeros(texture_size, texture_size, 3, device=device)
    acc_weight = torch.zeros(texture_size, texture_size, device=device)

    for img, az, el in zip(images, azimuths, elevations):
        img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        img_h, img_w = img_np.shape[:2]
        img_rgba = np.array(img.convert("RGBA"))
        if img.mode != "RGBA" or img_rgba[:, :, 3].min() == 255:
            bg_color = img_rgba[0, 0, :3]
            color_diff = np.abs(img_rgba[:, :, :3].astype(int) - bg_color.astype(int)).sum(axis=-1)
            fg_mask = (color_diff > 10).astype(np.uint8)
        else:
            fg_mask = (img_rgba[:, :, 3] > 127).astype(np.uint8)

        scale = max(img_h, img_w) / 1024.0
        dilate_px = int(max(5, 5 * scale))
        kernel = np.ones((3, 3), np.uint8)
        dilated_fg = cv2.dilate(fg_mask, kernel, iterations=dilate_px)
        band_to_fill = cv2.bitwise_and(dilated_fg, cv2.bitwise_not(fg_mask))
        img_rgb_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        if int(band_to_fill.sum()) > 0:
            for channel in range(3):
                img_rgb_u8[..., channel] = cv2.inpaint(img_rgb_u8[..., channel], band_to_fill, 3, cv2.INPAINT_NS)
        img_t = torch.from_numpy(img_rgb_u8.astype(np.float32) / 255.0).to(device).permute(2, 0, 1).unsqueeze(0)

        look, right, up = get_camera_vectors(float(az), float(el), device=device)
        occ_res = min(2048, img_h, img_w)
        cam_clip = build_ortho_clip_verts(out_verts, right, up, look, ortho_scale)
        cam_rast, _ = dr.rasterize(ctx, cam_clip, out_faces.int(), resolution=[occ_res, occ_res])
        cam_hit = cam_rast[0, :, :, 3] > 0
        cam_hit_img = cam_hit.float().unsqueeze(0).unsqueeze(0)
        cam_depth = dr.interpolate((-(out_verts * look).sum(-1)).unsqueeze(0).unsqueeze(-1), cam_rast, out_faces.int())[0][0]
        cam_depth = cam_depth.permute(2, 0, 1).unsqueeze(0)
        inf_depth = torch.full_like(cam_depth, torch.finfo(cam_depth.dtype).max)
        cam_depth_occ = torch.where(cam_hit.unsqueeze(0).unsqueeze(0), cam_depth, inf_depth)
        cam_depth_occ = -F.max_pool2d(-cam_depth_occ, kernel_size=3, stride=1, padding=1)

        u_clip = (tex_pos * right).sum(-1) / (max(float(ortho_scale), 1e-6) / 2.0)
        v_clip = (tex_pos * up).sum(-1) / (max(float(ortho_scale), 1e-6) / 2.0)

        cam_clip_verts = cam_clip[0]
        mesh_u_min = cam_clip_verts[:, 0].min()
        mesh_u_max = cam_clip_verts[:, 0].max()
        mesh_v_min = cam_clip_verts[:, 1].min()
        mesh_v_max = cam_clip_verts[:, 1].max()
        mesh_u_span = (mesh_u_max - mesh_u_min).clamp(min=1e-6)
        mesh_v_span = (mesh_v_max - mesh_v_min).clamp(min=1e-6)

        coords = np.argwhere(fg_mask)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
        else:
            y_min, x_min = 0, 0
            y_max, x_max = img_h - 1, img_w - 1

        u_norm = (u_clip - mesh_u_min) / mesh_u_span
        x_pixel = u_norm * float(x_max - x_min + 1) + float(x_min) - 0.5
        u_samp = ((x_pixel + 0.5) / float(img_w)) * 2.0 - 1.0

        v_norm = (v_clip - mesh_v_min) / mesh_v_span
        y_pixel = (1.0 - v_norm) * float(y_max - y_min + 1) + float(y_min) - 0.5
        v_samp = ((y_pixel + 0.5) / float(img_h)) * 2.0 - 1.0

        eps = 1e-4
        u_samp = u_samp.clamp(-1 + eps, 1 - eps)
        v_samp = v_samp.clamp(-1 + eps, 1 - eps)

        grid_occ = torch.stack([u_clip, v_clip], dim=-1).unsqueeze(0)
        sampled_hit = F.grid_sample(cam_hit_img, grid_occ, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0] > 0.25
        sampled_depth = F.grid_sample(cam_depth_occ, grid_occ, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
        tex_depth = -(tex_pos * look).sum(-1)
        depth_match = tex_depth >= sampled_depth - float(depth_eps)
        in_bounds = (u_samp.abs() <= 1.0) & (v_samp.abs() <= 1.0)
        visible = uv_hit_mask & in_bounds & sampled_hit & depth_match

        grid_col = torch.stack([u_samp, v_samp], dim=-1).unsqueeze(0)
        sampled_colors = F.grid_sample(img_t, grid_col, mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
        dot = (tex_normals * (-look)).sum(-1).clamp(min=0.0) ** 1.5
        weights = (dot ** float(blend_exponent)) * visible.float()
        acc_color += sampled_colors * weights.unsqueeze(-1)
        acc_weight += weights

    valid_mask = acc_weight > 1e-6
    projected_color = torch.zeros_like(acc_color)
    projected_color[valid_mask] = acc_color[valid_mask] / acc_weight[valid_mask].unsqueeze(-1)
    confidence = acc_weight / acc_weight.max().clamp(min=1e-6)
    confidence = confidence * confidence * (3.0 - 2.0 * confidence)
    confidence = confidence ** 0.5
    confidence3 = confidence.unsqueeze(-1)

    existing_base = None
    try:
        material = high_poly_mesh.visual.material
        existing_base = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    except Exception:
        existing_base = None

    if existing_base is not None and blend_texture:
        existing_base = existing_base.convert("RGBA")
        if existing_base.size != (texture_size, texture_size):
            existing_base = existing_base.resize((texture_size, texture_size), Image.LANCZOS)
        existing_np = np.flip(np.array(existing_base).astype(np.float32) / 255.0, axis=0).copy()
        existing_rgb = torch.from_numpy(existing_np[..., :3]).to(device)
        existing_alpha = torch.from_numpy(existing_np[..., 3:4]).to(device)
        blended_rgb = projected_color * confidence3 + existing_rgb * (1.0 - confidence3)
        color_np = (blended_rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        alpha_np = (existing_alpha.squeeze(-1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    else:
        color_np = (projected_color.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        conf_np = confidence.cpu().numpy()
        alpha_np = (conf_np * 255).clip(0, 255).astype(np.uint8)
        alpha_np[conf_np <= 0.01] = 0

    if fill_holes:
        uv_mask_np = uv_hit_mask.cpu().numpy().astype(np.uint8)
        valid_mask_np = valid_mask.cpu().numpy().astype(np.uint8)
        internal_hole_mask = cv2.bitwise_and(cv2.bitwise_not(valid_mask_np), uv_mask_np)
        if int(max_hole_size) > 0:
            _, labels, stats, _ = cv2.connectedComponentsWithStats(internal_hole_mask, connectivity=8)
            areas = stats[:, cv2.CC_STAT_AREA]
            valid_labels_mask = areas <= int(max_hole_size)
            valid_labels_mask[0] = False
            filtered_internal_holes = valid_labels_mask[labels].astype(np.uint8)
        else:
            filtered_internal_holes = internal_hole_mask
        kernel = np.ones((3, 3), np.uint8)
        dilated_uv = cv2.dilate(uv_mask_np, kernel, iterations=5)
        seam_padding_mask = cv2.bitwise_and(dilated_uv, cv2.bitwise_not(uv_mask_np))
        final_inpaint_mask = cv2.bitwise_or(filtered_internal_holes, seam_padding_mask)
        if int(final_inpaint_mask.sum()) > 0:
            for channel in range(3):
                color_np[..., channel] = cv2.inpaint(color_np[..., channel], final_inpaint_mask, 3, cv2.INPAINT_NS)
            alpha_np = cv2.inpaint(alpha_np, final_inpaint_mask, 3, cv2.INPAINT_NS)

    base_color_texture = Image.fromarray(np.dstack([color_np, alpha_np]))
    metallic_roughness_texture = None
    if (
        hasattr(high_poly_mesh, "visual")
        and hasattr(high_poly_mesh.visual, "material")
        and isinstance(high_poly_mesh.visual.material, trimesh.visual.material.PBRMaterial)
    ):
        metallic_roughness_texture = getattr(high_poly_mesh.visual.material, "metallicRoughnessTexture", None)
    if metallic_roughness_texture is None or not use_metallic:
        mr_np = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        mr_np[..., 1] = 230
        metallic_roughness_texture = Image.fromarray(mr_np)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=metallic_roughness_texture,
        metallicFactor=0.0,
        roughnessFactor=0.9,
        alphaMode="OPAQUE",
        doubleSided=True,
    )

    verts_np = out_verts.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    uvs_np[:, 1] = 1.0 - uvs_np[:, 1]
    textured_mesh = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )
    return textured_mesh, base_color_texture, metallic_roughness_texture
