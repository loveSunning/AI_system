"""Local model aliases and vLLM quantization choices, without importing GPU libraries.

Profile labels identify comparison deployments and validate their weight formats. ``dtype`` controls activation/nonquantized
layers; it never changes packed AWQ or FP8 weights into BF16 weights on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

PROFILE_NAMES = ("bf16", "awq", "fp8-official", "fp8-local")
MODEL_DIRECTORIES = {
    "bf16": "Qwen3-8B",
    "awq": "Qwen3-8B-AWQ",
    "fp8-official": "Qwen3-8B-FP8",
    "fp8-local": "Qwen3-8B-FP8-local",
}
_ALIASES = {
    "bf16": "bf16", "qwen3-8b": "bf16",
    "awq": "awq", "qwen3-8b-awq": "awq",
    "fp8-official": "fp8-official", "qwen3-8b-fp8": "fp8-official",
    # Preserve the historical short alias; select official FP8 explicitly.
    "fp8": "fp8-local", "fp8-local": "fp8-local", "qwen3-8b-fp8-local": "fp8-local",
}


def canonical_profile(value):
    """Return a deployment label for a case-insensitive profile/model alias."""
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


def profile_from_model_name(value):
    """Read a deployment label from an explicit alias or known directory basename."""
    for name in (str(value), Path(str(value)).name):
        try:
            return canonical_profile(name)
        except ValueError:
            pass
    return None


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


def validate_model_profile(config, profile):
    """Validate a selected deployment without inferring official provenance from dtype.

    The legacy inspector describes the FP8 format family as ``fp8-local``.
    Official Qwen FP8 is a separate deployment using the native fp8 loader;
    local native-FP8 deployments remain supported for historical configurations.
    Weight hashes and source records, rather than this label, identify contents.
    """
    selected = canonical_profile(profile)
    detected = inspect_model_config(config)
    if selected == "fp8-official":
        quant = config.get("quantization_config") or {}
        if (detected != "fp8-local" or quant_method(config) != "fp8"
                or quant.get("fmt") != "e4m3" or quant.get("activation_scheme") != "dynamic"
                or quant.get("weight_block_size") != [128, 128]):
            raise ValueError("Official FP8 profile requires native fp8, E4M3, dynamic activations "
                             "and weight_block_size=[128, 128]")
    elif selected != detected:
        raise ValueError("Selected profile does not match actual model quantization config")
    return selected


def resolve_runtime_options(config, dtype="auto", quantization="auto"):
    """Choose matching options for the pinned legacy and unified vLLM stacks.

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
