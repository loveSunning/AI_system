"""Local model aliases and vLLM quantization choices, without importing GPU libraries.

Profile labels describe weight formats. ``dtype`` controls activation/nonquantized
layers; it never changes packed AWQ or FP8 weights into BF16 weights on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

PROFILE_NAMES = ("bf16", "awq", "fp8-local")
MODEL_DIRECTORIES = {
    "bf16": "Qwen3-8B",
    "awq": "Qwen3-8B-AWQ",
    "fp8-local": "Qwen3-8B-FP8-local",
}
_ALIASES = {
    "bf16": "bf16", "qwen3-8b": "bf16",
    "awq": "awq", "qwen3-8b-awq": "awq",
    "fp8": "fp8-local", "fp8-local": "fp8-local", "qwen3-8b-fp8-local": "fp8-local",
}


def canonical_profile(value):
    """Return a built-in format label for a case-insensitive profile/model alias."""
    try:
        return _ALIASES[str(value).strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown model profile {value!r}; choose {', '.join(PROFILE_NAMES)}") from None


def resolve_model_alias(value):
    """Expand known aliases under MODEL_ROOT; leave explicit local paths unchanged."""
    try:
        profile = canonical_profile(value)
    except ValueError:
        return str(value)
    return str(Path(os.environ.get("MODEL_ROOT", "/workspace/weights_pub")) / MODEL_DIRECTORIES[profile])


def quant_method(config):
    quant = config.get("quantization_config") or {}
    return str(quant.get("quant_method", "")).lower().replace("_", "-")


def inspect_model_config(config):
    """Identify supported unquantized, AWQ/W4A16 or FP8/W8A8 metadata.

    This validates format declarations, not tensor contents or kernel availability.
    A compressed-tensors INT4/INT8 recipe must not be mislabeled as local FP8.
    """
    if not isinstance(config, dict):
        raise ValueError("Model config must contain an object")
    quant = config.get("quantization_config")
    if quant is None or quant == {}:
        if config.get("compression_config"):
            raise ValueError("Unsupported legacy compression_config; use quantization_config metadata")
        return "bf16"
    if not isinstance(quant, dict):
        raise ValueError("quantization_config must contain an object")
    method = quant_method(config)
    if method in ("awq", "awq-marlin"):
        if quant.get("bits", 4) != 4:
            raise ValueError("AWQ profile requires 4-bit weights (W4A16)")
        return "awq"
    if method == "fp8":
        if str(quant.get("fmt", "e4m3")).lower() != "e4m3":
            raise ValueError("FP8 profile requires E4M3 weight metadata")
        return "fp8-local"
    if method == "compressed-tensors":
        groups = quant.get("config_groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("compressed-tensors FP8 requires nonempty config_groups")
        for scheme in groups.values():
            if not isinstance(scheme, dict):
                raise ValueError("Invalid compressed-tensors FP8 config group")
            weight = scheme.get("weights") or {}
            acts = scheme.get("input_activations") or {}
            if not isinstance(weight, dict) or not isinstance(acts, dict):
                raise ValueError("Invalid compressed-tensors FP8 tensor scheme")
            if (weight.get("num_bits") != 8 or weight.get("type") != "float"
                    or acts.get("num_bits") != 8 or acts.get("type") != "float"
                    or acts.get("dynamic") is not True):
                raise ValueError("compressed-tensors profile requires FP8 weights and dynamic FP8 activations")
        return "fp8-local"
    raise ValueError(f"Unsupported quantization method {method!r}; supported profiles: {', '.join(PROFILE_NAMES)}")


def resolve_runtime_options(config, dtype="auto", quantization="auto"):
    """Choose matching vLLM 0.11 options for the supported weight formats.

    AWQ-Marlin accepts BF16 on SM80+ (including 4090D); original ``awq`` only
    accepts FP16. Compressed-tensors FP8 retains its real loader metadata.
    Actual CUDA kernel support is checked later by vLLM during model loading.
    """
    profile = inspect_model_config(config)
    requested = "auto" if quantization is None else str(quantization).lower()
    normalized = requested.replace("_", "-")
    method = quant_method(config) or None
    if profile == "bf16":
        if normalized != "auto":
            raise ValueError("Unquantized weights require --quantization auto in this script; use a prequantized model")
    elif profile == "awq":
        if normalized == "auto":
            requested = "awq_marlin"
        elif normalized not in ("awq", "awq-marlin"):
            raise ValueError("AWQ weights require --quantization auto, awq_marlin or awq")
        else:
            requested = normalized.replace("-", "_")
        if requested == "awq":
            if dtype == "auto":
                dtype = "float16"
            if dtype != "float16":
                raise ValueError("Original AWQ supports float16 only; use --dtype float16 or --quantization awq_marlin")
        else:
            if dtype == "auto":
                dtype = "bfloat16"
            if dtype not in ("float16", "bfloat16"):
                raise ValueError("AWQ-Marlin supports float16/bfloat16 activations")
    else:
        if normalized not in ("auto", method):
            raise ValueError(f"FP8 checkpoint uses {method}; keep --quantization auto or explicitly select {method}")
        if normalized != "auto":
            requested = method
        if dtype == "auto":
            dtype = "bfloat16"
        if dtype not in ("float16", "bfloat16"):
            raise ValueError("FP8 inference uses float16/bfloat16 activations outside quantized GEMMs")
    return {"profile": profile, "dtype": dtype, "quantization": requested, "quant_method": method}
