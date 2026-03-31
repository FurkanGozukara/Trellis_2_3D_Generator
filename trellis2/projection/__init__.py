def default_projection_views(view_count):
    from .texture_projection_multiview import default_projection_views as _default_projection_views

    return _default_projection_views(view_count)


def resolve_projection_views(image_paths, azimuth_text="", elevation_text=""):
    from .texture_projection_multiview import resolve_projection_views as _resolve_projection_views

    return _resolve_projection_views(image_paths, azimuth_text, elevation_text)


def texture_mesh_with_multiview(*args, **kwargs):
    from .texture_projection_multiview import texture_mesh_with_multiview as _texture_mesh_with_multiview

    return _texture_mesh_with_multiview(*args, **kwargs)

__all__ = [
    "default_projection_views",
    "resolve_projection_views",
    "texture_mesh_with_multiview",
]
