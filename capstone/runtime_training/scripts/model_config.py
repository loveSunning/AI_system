"""Local model checks for the standalone W31 service; no GPU imports.

Only unquantized causal-LM checkpoints and embedded AWQ INT4/GEMM metadata are
supported in this week. Validation checks declared configuration and file names;
it does not inspect tensor contents or prove that a GPU backend can execute them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


MODEL_ALIASES = {
    "qwen3-8b": ("Qwen3-8B", "unquantized"),
    "bf16": ("Qwen3-8B", "unquantized"),
    "qwen3-8b-awq": ("Qwen3-8B-AWQ", "awq"),
    "awq": ("Qwen3-8B-AWQ", "awq"),
}
DTYPES = ("auto", "bfloat16", "float16", "float32")
QUANTIZATIONS = ("auto", "awq", "awq_marlin")


def resolve_model_alias(value, model_root=None):
    """Resolve project-owned aliases; explicit local paths stay local paths."""
    text = str(value)
    alias = text.strip().lower()
    if alias == "fp8" or alias.startswith("qwen3-8b-fp8") or alias.startswith("fp8-"):
        raise ValueError("W31 supports unquantized and AWQ INT4 checkpoints only; FP8 is not supported by this launcher.")
    if alias in MODEL_ALIASES:
        root = Path(model_root if model_root is not None else os.environ.get("MODEL_ROOT", "/workspace/weights_pub"))
        return (root.expanduser() / MODEL_ALIASES[alias][0]).resolve()
    return Path(text).expanduser().resolve()


def _read_object(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def inspect_quantization(model_config):
    """Classify supported checkpoint metadata and reject partial AWQ records."""
    quant = model_config.get("quantization_config")
    if quant is None or quant == {}:
        if model_config.get("compression_config"):
            raise ValueError("Legacy compression_config is not supported by the W31 launcher.")
        return "unquantized", None
    if not isinstance(quant, dict):
        raise ValueError("quantization_config must be a JSON object")
    method = str(quant.get("quant_method", "")).lower()
    if method != "awq":
        raise ValueError(f"W31 supports unquantized and AWQ INT4 checkpoints only; unsupported quant_method={method!r}.")
    if type(quant.get("bits")) is not int or quant["bits"] != 4:
        raise ValueError("W31 AWQ requires bits=4")
    if type(quant.get("group_size")) is not int or quant["group_size"] not in (-1, 32, 64, 128):
        raise ValueError("W31 AWQ requires group_size in [-1, 32, 64, 128]")
    if not isinstance(quant.get("zero_point"), bool):
        raise ValueError("W31 AWQ requires a boolean zero_point")
    if str(quant.get("version", "GEMM")).lower() != "gemm":
        raise ValueError("W31 AWQ supports GEMM checkpoint layout only")
    return "awq", quant


def resolve_runtime_options(profile, dtype, quantization, model_config=None):
    """Choose explicit activation dtype and loader without changing weights."""
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if quantization not in QUANTIZATIONS:
        raise ValueError("quantization must be auto, awq, or awq_marlin")
    if profile == "unquantized":
        if quantization != "auto":
            raise ValueError("An unquantized checkpoint requires quantization=auto; this launcher does not quantize weights.")
        if dtype == "auto":
            declared = (model_config or {}).get("dtype", (model_config or {}).get("torch_dtype"))
            dtype = declared if declared in DTYPES[1:] else "bfloat16"
        return dtype, "auto"
    if profile != "awq":
        raise ValueError(f"Unsupported checkpoint profile: {profile}")
    loader = "awq_marlin" if quantization == "auto" else quantization
    if loader == "awq":
        dtype = "float16" if dtype == "auto" else dtype
        if dtype != "float16":
            raise ValueError("The original awq loader accepts float16 only; use dtype=float16 or quantization=awq_marlin.")
    else:
        dtype = "bfloat16" if dtype == "auto" else dtype
        if dtype not in ("float16", "bfloat16"):
            raise ValueError("awq_marlin requires float16 or bfloat16 activations")
    return dtype, loader


def _weight_inventory(path, require_files):
    files = sorted(p.name for pattern in ("*.safetensors", "*.bin") for p in path.glob(pattern) if p.is_file())
    if require_files and not files:
        raise ValueError(f"No local .safetensors/.bin model weights found: {path}")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / name
        if not index_path.is_file():
            continue
        index = _read_object(index_path)
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping or not all(isinstance(v, str) and v for v in mapping.values()):
            raise ValueError(f"{name} requires a nonempty weight_map of filenames")
        if require_files:
            for filename in set(mapping.values()):
                shard = (path / filename).resolve()
                if not shard.is_relative_to(path) or not shard.is_file():
                    raise ValueError(f"Missing or invalid model shard declared in {name}: {filename}")
    return files


def prepare_model(config, *, require_files=False):
    """Return resolved runtime options and evidence for the launch manifest.

    A dry run may omit downloaded files. Existing model metadata is still
    validated; real serving also requires local model/tokenizer files.
    """
    model_path = resolve_model_alias(config["model"])
    tokenizer_path = resolve_model_alias(config.get("tokenizer", str(model_path)))
    config_path = model_path / "config.json"
    model_config = _read_object(config_path) if config_path.is_file() else None
    alias = str(config["model"]).strip().lower()
    expected = MODEL_ALIASES.get(alias, (None, None))[1]
    if require_files:
        if not model_path.is_dir():
            raise ValueError(f"Local model directory does not exist: {model_path}; set MODEL_ROOT or use an absolute --model path.")
        if model_config is None:
            raise ValueError(f"Missing local model config: {config_path}")
        if not isinstance(model_config.get("model_type"), str) or not model_config["model_type"]:
            raise ValueError("config.json requires a nonempty model_type")
    if model_config is not None:
        profile, quant_metadata = inspect_quantization(model_config)
        if expected is not None and profile != expected:
            raise ValueError(f"Model alias {config['model']!r} expects {expected} metadata, found {profile}.")
    else:
        profile = expected or ("awq" if config["quantization"] in ("awq", "awq_marlin") else "unquantized")
        quant_metadata = None
    dtype, quantization = resolve_runtime_options(profile, config["dtype"], config["quantization"], model_config)
    weights = _weight_inventory(model_path, require_files)
    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    tokenizer_config = _read_object(tokenizer_config_path) if tokenizer_config_path.is_file() else None
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt") if (tokenizer_path / name).is_file()]
    if require_files and (not tokenizer_path.is_dir() or tokenizer_config is None or not tokenizer_files):
        raise ValueError(f"Tokenizer requires a local directory with tokenizer_config.json and tokenizer/vocabulary files: {tokenizer_path}")
    resolved = {**config, "model": str(model_path), "tokenizer": str(tokenizer_path), "dtype": dtype, "quantization": quantization}
    metadata = {
        "requested_model": config["model"],
        "resolved_model_path": str(model_path),
        "resolved_tokenizer_path": str(tokenizer_path),
        "directory_exists": model_path.is_dir(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if model_config is not None else None,
        "tokenizer_config_sha256": hashlib.sha256(tokenizer_config_path.read_bytes()).hexdigest() if tokenizer_config is not None else None,
        "profile": profile,
        "quantization_config": quant_metadata,
        "weight_files": weights,
        "tokenizer_files": tokenizer_files,
        "validation": "local_files_checked" if require_files else "planning_only",
        "note": "Declared metadata and filenames only; tensor contents, GPU compatibility, and attention backend require actual execution.",
    }
    return resolved, metadata
