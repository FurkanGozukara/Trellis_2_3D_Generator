from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional


DEFAULT_ATTENTION_BACKEND = "auto"
DEFAULT_SAMPLER_TYPE = "heun"
DEFAULT_MODEL_VARIANT = "standard"

ATTENTION_BACKEND_CHOICES = [
    "auto",
    "flash_attn",
    "flash_attn_3",
    "xformers",
    "sdpa",
]
SAMPLER_TYPE_CHOICES = ["heun", "euler", "rk4", "rk5"]
MODEL_VARIANT_CHOICES = ["standard", "fp8"]

_ALL_DENSE_BACKENDS = ("flash_attn_3", "flash_attn", "xformers", "sdpa", "naive")
_SPARSE_BACKENDS = ("flash_attn", "xformers")
_SAMPLER_PREFIX = {
    "euler": "Euler",
    "heun": "Heun",
    "rk4": "RK4",
    "rk5": "RK5",
}
_SAMPLER_SUFFIXES = (
    "MultiViewGuidanceIntervalSampler",
    "MultiViewSampler",
    "GuidanceIntervalSampler",
    "CfgSampler",
    "Sampler",
)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def backend_availability() -> Dict[str, bool]:
    return {
        "flash_attn_3": _module_available("flash_attn_interface"),
        "flash_attn": _module_available("flash_attn"),
        "xformers": _module_available("xformers"),
        "sdpa": True,
        "naive": True,
    }


def normalize_attention_backend(requested: Optional[str]) -> str:
    value = str(requested or DEFAULT_ATTENTION_BACKEND).strip().lower()
    if value in _ALL_DENSE_BACKENDS or value == "auto":
        return value
    return DEFAULT_ATTENTION_BACKEND


def normalize_sampler_type(requested: Optional[str]) -> str:
    value = str(requested or DEFAULT_SAMPLER_TYPE).strip().lower()
    if value in _SAMPLER_PREFIX:
        return value
    return DEFAULT_SAMPLER_TYPE


def normalize_model_variant(requested: Optional[str]) -> str:
    value = str(requested or DEFAULT_MODEL_VARIANT).strip().lower()
    if value in MODEL_VARIANT_CHOICES:
        return value
    return DEFAULT_MODEL_VARIANT


def resolve_model_variant(requested: Optional[str]) -> Dict[str, str]:
    variant = normalize_model_variant(requested)
    if variant == "fp8":
        return {
            "variant": "fp8",
            "model_repo": "visualbruno/TRELLIS.2-4B-FP8",
            "config_file": "pipeline_fp8.json",
        }
    return {
        "variant": "standard",
        "model_repo": "microsoft/TRELLIS.2-4B",
        "config_file": "pipeline.json",
    }


def resolve_attention_backends(requested: Optional[str]) -> Dict[str, Any]:
    requested_backend = normalize_attention_backend(requested)
    availability = backend_availability()

    if requested_backend == "auto":
        dense_backend = next(
            (backend for backend in ("flash_attn_3", "flash_attn", "xformers", "sdpa", "naive") if availability.get(backend)),
            "naive",
        )
    else:
        if not availability.get(requested_backend, False):
            raise RuntimeError(
                f"Requested attention backend '{requested_backend}' is not available in the active environment."
            )
        dense_backend = requested_backend

    if dense_backend in _SPARSE_BACKENDS:
        sparse_backend = dense_backend
    else:
        sparse_backend = next((backend for backend in _SPARSE_BACKENDS if availability.get(backend)), None)
        if sparse_backend is None:
            raise RuntimeError(
                "No supported sparse attention backend is available. Install 'flash-attn' or 'xformers'."
            )

    return {
        "requested_backend": requested_backend,
        "dense_backend": dense_backend,
        "sparse_backend": sparse_backend,
        "availability": availability,
    }


def apply_runtime_backends(requested: Optional[str]) -> Dict[str, Any]:
    resolved = resolve_attention_backends(requested)

    from trellis2.modules.attention import config as dense_config
    from trellis2.modules.sparse import config as sparse_config

    dense_config.set_backend(resolved["dense_backend"])
    sparse_config.set_attn_backend(resolved["sparse_backend"])

    os.environ["ATTN_BACKEND"] = resolved["dense_backend"]
    os.environ["SPARSE_ATTN_BACKEND"] = resolved["sparse_backend"]

    return resolved


def sampler_class_name(base_name: str, sampler_type: Optional[str]) -> str:
    sampler_key = normalize_sampler_type(sampler_type)
    prefix = _SAMPLER_PREFIX[sampler_key]
    for suffix in _SAMPLER_SUFFIXES:
        if str(base_name).endswith(suffix):
            return f"Flow{prefix}{suffix}"
    raise ValueError(f"Unsupported sampler class name: {base_name}")
