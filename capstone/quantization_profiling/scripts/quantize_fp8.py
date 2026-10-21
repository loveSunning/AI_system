#!/usr/bin/env python3
"""Quantize local dense Qwen3 weights with LLM Compressor FP8_BLOCK.

Exported format is genuinely compressed-tensors: static E4M3 128x128 weights
and dynamic per-token groups of 128 activations. No dataset or vLLM is needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN"}
FLOAT_DTYPES = {"BF16", "F16", "F32"}
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
PACKAGES = ("torch", "transformers", "tokenizers", "llmcompressor", "compressed-tensors", "safetensors")
REFERENCE = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
             "weight_block_size": [128, 128]}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def versions():
    values = {}
    for name in PACKAGES:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def tensor_catalog(directory):
    """Read safetensors headers only; never allocate model tensors here."""
    catalog = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors prefix: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if length <= 0 or length > 64 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(length))
        start, size = 8 + length, path.stat().st_size
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in catalog:
                raise ValueError("Duplicate tensor: " + name)
            shape, offsets = item.get("shape"), item.get("data_offsets")
            if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= size - start):
                raise ValueError("Invalid tensor metadata: " + name)
            catalog[name] = dict(item, file=path.name, data_start=start)
    if not catalog:
        raise ValueError("No safetensors tensors found: " + str(directory))
    index_path = Path(directory) / "model.safetensors.index.json"
    if index_path.exists():
        mapping = read_json(index_path)["weight_map"]
        if set(mapping) != set(catalog) or any(catalog[n]["file"] != f for n, f in mapping.items()):
            raise ValueError("Safetensors index does not match actual shards")
    return catalog


def separate_paths(source, output):
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")


def inspect_source(source):
    config = read_json(source / "config.json")
    if config.get("model_type") != "qwen3":
        raise ValueError("This entry point supports dense Qwen3 (model_type=qwen3)")
    if config.get("quantization_config") or config.get("compression_config"):
        raise ValueError("Input is already quantized; use the original BF16 model")
    catalog = tensor_catalog(source)
    if any(t["dtype"] not in FLOAT_DTYPES for t in catalog.values()):
        raise ValueError("Input contains quantized/nonfloating tensors; use original weights")
    expected = [f"model.layers.{i}.{projection}.weight"
                for i in range(config["num_hidden_layers"]) for projection in PROJECTIONS]
    required = expected + ["model.embed_tokens.weight", "lm_head.weight"]
    if any(n not in catalog for n in required):
        raise ValueError("Incomplete Qwen3 weights: " + ", ".join(n for n in required if n not in catalog))
    for name in expected:
        if len(catalog[name]["shape"]) != 2:
            raise ValueError("Projection must be 2D: " + name)
    if not (source / "tokenizer_config.json").is_file() or not (source / "tokenizer.json").is_file():
        raise ValueError("Local tokenizer_config.json and tokenizer.json are required")
    return config, catalog, expected


def validate_scheme(scheme):
    """Check the installed preset rather than trusting a preset name alone."""
    weights, acts = scheme.get("weights") or {}, scheme.get("input_activations") or {}
    if (weights.get("num_bits") != 8 or weights.get("type") != "float"
            or weights.get("strategy") != "block" or weights.get("block_structure") != [128, 128]
            or weights.get("dynamic", False) is not False or weights.get("symmetric", True) is not True
            or acts.get("num_bits") != 8 or acts.get("type") != "float"
            or acts.get("strategy") != "group" or acts.get("group_size") != 128
            or acts.get("dynamic") is not True or acts.get("symmetric", True) is not True
            or scheme.get("output_activations") is not None):
        raise ValueError("FP8_BLOCK does not match static 128x128 weights / dynamic groups of 128")


def read_positive_scales(directory, item, name):
    sizes = {"F32": 4, "F16": 2, "BF16": 2}
    dtype = item["dtype"]
    if dtype not in sizes:
        raise ValueError("Invalid scale dtype: " + name)
    first, last = item["data_offsets"]
    count = math.prod(item["shape"])
    if last - first != count * sizes[dtype]:
        raise ValueError("Invalid scale data size: " + name)
    with (directory / item["file"]).open("rb") as stream:
        stream.seek(item["data_start"] + first)
        data = stream.read(last - first)
    if dtype == "BF16":
        values = (struct.unpack("<f", struct.pack("<I", n << 16))[0]
                  for (n,) in struct.iter_unpack("<H", data))
    else:
        values = (v for (v,) in struct.iter_unpack("<f" if dtype == "F32" else "<e", data))
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Nonfinite or nonpositive scale: " + name)


def verify_export(output, expected, original_shapes):
    config = read_json(output / "config.json")
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method", "").replace("_", "-") != "compressed-tensors":
        raise ValueError("Expected actual compressed-tensors export; do not rename quant_method to fp8")
    if quant.get("quantization_status") != "compressed":
        raise ValueError("Export is not marked compressed")
    if "lm_head" not in quant.get("ignore", []):
        raise ValueError("Output lm_head must be excluded from quantization")
    if config.get("model_type") != "qwen3" or type(config.get("num_hidden_layers")) is not int or config["num_hidden_layers"] <= 0:
        raise ValueError("Missing Qwen3 model architecture metadata")
    architecture_weights = {f"model.layers.{i}.{projection}.weight"
        for i in range(config["num_hidden_layers"]) for projection in PROJECTIONS}
    if set(expected) != architecture_weights or len(expected) != len(architecture_weights):
        raise ValueError("Expected linear list does not match every Qwen3 projection")
    if not architecture_weights | {"model.embed_tokens.weight", "lm_head.weight"} <= set(original_shapes):
        raise ValueError("Source shape manifest is incomplete")
    groups = quant.get("config_groups") or {}
    if len(groups) != 1:
        raise ValueError("Expected exactly one FP8_BLOCK config group")
    scheme = next(iter(groups.values()))
    validate_scheme(scheme)
    if scheme.get("targets") != ["Linear"]:
        raise ValueError("Expected all Linear layers targeted")
    catalog = tensor_catalog(output)
    if set(catalog) != set(original_shapes) | {n.removesuffix(".weight") + ".weight_scale" for n in expected}:
        raise ValueError("Export tensor names differ from weights plus expected block scales")
    for name, shape in original_shapes.items():
        if catalog[name]["shape"] != shape:
            raise ValueError("Tensor shape changed: " + name)
    for name in expected:
        item = catalog[name]
        if item["dtype"] not in FP8_DTYPES:
            raise ValueError("Weight was not saved as E4M3 FP8: " + name)
        first, last = item["data_offsets"]
        if last - first != math.prod(item["shape"]):
            raise ValueError("FP8 weight byte size mismatch: " + name)
        scale_name = name.removesuffix(".weight") + ".weight_scale"
        scale = catalog[scale_name]
        scale_shape = [math.ceil(n / 128) for n in item["shape"]]
        if scale["shape"] != scale_shape:
            raise ValueError("Incorrect 128x128 scale shape: " + scale_name)
        read_positive_scales(output, scale, scale_name)
    for name, item in catalog.items():
        if name in original_shapes and name not in expected and item["dtype"] != "BF16":
            raise ValueError("Unquantized weights must remain BF16: " + name)
    for name in ("tokenizer_config.json", "tokenizer.json"):
        if not (output / name).is_file():
            raise ValueError("Missing saved tokenizer: " + name)
    return {"quantized_linear_count": len(expected), "weight_dtype": "FP8 E4M3",
            "weight_block_size": [128, 128], "activation_group_size": 128,
            "activation_scheme": "dynamic", "format": "compressed-tensors",
            "retained_dtype": "BF16", "config_and_tensors_verified": True,
            "target_gpu_inference_verified": False}


def fingerprints(directory, logger=None):
    files = sorted(f for f in directory.iterdir() if f.is_file()
                   and f.suffix in (".json", ".safetensors", ".jinja", ".txt"))
    result = {}
    for path in files:
        if path.name == "quantization_manifest.json":
            continue
        if logger:
            logger.info("SHA256 %s", path.name)
        result[path.name] = sha256(path)
    return result


def execute_quantization(args, logger, manifest):
    # Import only after input validation. Help/dry-run/verify-only need no GPU libraries.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme("FP8_BLOCK", ["Linear"])
    serialized = scheme.model_dump(mode="json")
    validate_scheme(serialized)
    manifest["recipe"] = {"targets": ["Linear"], "scheme": serialized, "ignore": ["lm_head"]}
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu or check the quantization environment")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        manifest["cuda"] = {"name": torch.cuda.get_device_name(device), "free_before_bytes": free,
                            "total_bytes": total, "capability": torch.cuda.get_device_capability(device)}
        logger.info("CUDA free %.2f GiB / total %.2f GiB", free / 2**30, total / 2**30)
    set_seed(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    if "enable_thinking" not in tokenizer.get_chat_template():
        raise ValueError("Expected Qwen3 chat template with thinking support")
    manifest["chat_template_sha256"] = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    logger.info("Loading BF16 source on %s", args.device)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16,
        device_map={"": args.device}, local_files_only=True, trust_remote_code=False,
        attn_implementation="eager")
    model.eval()
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise ValueError("Model has unloaded meta tensors")
    manifest["load_seconds"] = time.perf_counter() - started
    logger.info("Applying FP8_BLOCK RTN; no calibration dataset and no generation")
    started = time.perf_counter()
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_BLOCK", ignore=["lm_head"])
    oneshot(model=model, tokenizer=tokenizer, recipe=recipe, log_dir=manifest["compressor_log_dir"])
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    manifest["quantize_seconds"] = time.perf_counter() - started
    logger.info("Saving compressed tensors to %s", args.output)
    started = time.perf_counter()
    model.save_pretrained(str(args.output), save_compressed=True, safe_serialization=True,
                          max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(str(args.output))
    manifest["save_seconds"] = time.perf_counter() - started
    if args.device.startswith("cuda"):
        manifest["cuda"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved())
    # No vLLM instance here: loading a second copy would change memory requirements.
    del model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("/workspace/weights_pub/Qwen3-8B"))
    p.add_argument("--output", type=Path, default=Path("/workspace/weights_pub/Qwen3-8B-FP8-local"))
    p.add_argument("--device", default="cpu", help="cpu or cuda:N; CPU avoids conversion VRAM peaks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-shard-size", default="2GB")
    p.add_argument("--source-revision", default="unknown", help="optional known immutable source commit label")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="check local inputs and print plan; writes nothing")
    mode.add_argument("--verify-only", action="store_true", help="verify a completed export and file hashes")
    p.add_argument("--log-dir", type=Path, default=ROOT / "log")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args(argv)
    if a.device == "cuda":
        a.device = "cuda:0"
    if not re.fullmatch(r"cpu|cuda:[0-9]+", a.device):
        p.error("--device must be cpu or cuda:N")
    if not 0 <= a.seed < 2**32:
        p.error("seed must be in [0, 2**32)")
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?(?:KB|MB|GB)", a.max_shard_size):
        p.error("max-shard-size must use a positive KB/MB/GB size, e.g. 2GB")
    a.model = a.model.expanduser().resolve()
    a.output = a.output.expanduser().resolve()
    return a


def main(argv=None):
    args = parse_args(argv)
    logger = logging.getLogger("quantize_fp8")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    file_handler = None
    manifest = None
    created_output = False
    try:
        if args.verify_only:
            manifest = read_json(args.output / "quantization_manifest.json")
            if manifest.get("status") != "complete":
                raise ValueError("Output does not have a completed conversion manifest")
            expected = manifest["expected_linear_weights"]
            result = verify_export(args.output, expected, manifest["original_shapes"])
            actual_hashes = fingerprints(args.output)
            if actual_hashes != manifest["output_sha256"]:
                raise ValueError("Output SHA256 mismatch or incomplete file coverage")
            logger.info("Verified export (GPU inference still separate): %s", json.dumps(result))
            return 0
        separate_paths(args.model, args.output)
        if args.output.exists():
            raise ValueError("Output exists; choose a new directory. Partial outputs are never overwritten")
        config, catalog, expected = inspect_source(args.model)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        plan = {"source": str(args.model), "output": str(args.output), "device": args.device,
                "source_revision_label": args.source_revision, "versions": versions(),
                "reference_native_config": REFERENCE, "export_format": "compressed-tensors",
                "scheme": "FP8_BLOCK", "ignore": ["lm_head"], "calibration_dataset": None,
                "expected_linear_count": len(expected), "seed": args.seed,
                "stored_source_bytes": sum(x["data_offsets"][1] - x["data_offsets"][0] for x in catalog.values()),
                "note": "same block semantics; not a bit-identical reproduction of Qwen's private conversion script"}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        missing = [p for p, v in plan["versions"].items() if v is None]
        if missing:
            raise RuntimeError("Missing quantization dependencies: " + ", ".join(missing)
                               + ". Install requirements-quantization.txt in a separate environment")
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / ("quantize_fp8_" + run_id + ".log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(handler.formatter)
        logger.addHandler(file_handler)
        args.output.mkdir(parents=True, exist_ok=False)
        created_output = True
        manifest = dict(plan, schema_version=1, run_id=run_id, status="running", started_at_utc=utc(),
                        expected_linear_weights=expected, original_shapes={n:t["shape"] for n,t in catalog.items()},
                        log_path=str(log_path), compressor_log_dir=str(args.log_dir / ("compressor_" + run_id)),
                        script_sha256=sha256(Path(__file__)))
        write_json(args.output / "quantization_manifest.json", manifest)
        logger.info("Hashing source files before conversion")
        manifest["source_sha256"] = fingerprints(args.model, logger)
        write_json(args.output / "quantization_manifest.json", manifest)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        execute_quantization(args, logger, manifest)
        logger.info("Checking every projection dtype, block scale, shape and retained layer")
        manifest["verification"] = verify_export(args.output, expected, manifest["original_shapes"])
        for name in ("tokenizer.json", "tokenizer_config.json"):
            if sha256(args.model / name) != manifest["source_sha256"][name]:
                raise ValueError("Source tokenizer changed during conversion")
        # Compare the rendered template, since tokenizer serialization may normalize JSON.
        from transformers import AutoTokenizer
        saved_tokenizer = AutoTokenizer.from_pretrained(str(args.output), local_files_only=True, trust_remote_code=False)
        saved_hash = hashlib.sha256(saved_tokenizer.get_chat_template().encode()).hexdigest()
        if saved_hash != manifest["chat_template_sha256"]:
            raise ValueError("Saved chat template differs from source")
        manifest["output_sha256"] = fingerprints(args.output, logger)
        manifest.update(status="complete", finished_at_utc=utc())
        write_json(args.output / "quantization_manifest.json", manifest)
        logger.info("COMPLETE: %s; config uses compressed-tensors, not native fp8", args.output)
        return 0
    except KeyboardInterrupt:
        if created_output and manifest:
            manifest.update(status="interrupted", finished_at_utc=utc())
            write_json(args.output / "quantization_manifest.json", manifest)
        logger.error("Interrupted; partial output retained. Retry into a new directory")
        return 130
    except Exception as error:
        if created_output and manifest:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=utc())
            write_json(args.output / "quantization_manifest.json", manifest)
        logger.error("%s: %s", type(error).__name__, error)
        if "out of memory" in str(error).lower():
            logger.error("Retry into a new output directory with --device cpu; ensure enough host RAM")
        if args.debug:
            traceback.print_exc()
        return 1
    finally:
        for current in (file_handler, handler):
            if current:
                logger.removeHandler(current)
                current.close()


if __name__ == "__main__":
    raise SystemExit(main())
