#!/usr/bin/env python3
"""Freeze rendered prompts, model fingerprints and protocol on the target host."""
import argparse
from pathlib import Path
from experiment_utils import ROOT, digest, read, write, write_rows


def exact_prompt(tokenizer, target, index):
    # Re-render the entire conversation each attempt: padding before template rendering
    # can change BPE boundaries, so never infer final length from character count.
    prefix = f"Record {index:04d}. Read the following data and continue the sequence:"
    count = max(0, target - 40)
    for _ in range(40):
        messages = [{"role": "user", "content": prefix + " x" * count}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == target:
            if tokenizer.encode(text, add_special_tokens=False) != ids:
                raise ValueError("Non-deterministic tokenizer")
            return {"messages": messages, "rendered_prompt": text, "prompt_token_ids": ids}
        count += target - len(ids)
        if count < 0:
            break
    raise ValueError(f"Cannot construct exact {target}-token prompt; no truncated template saved")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("/workspace/weights_pub/Qwen3-8B"))
    p.add_argument("--revision", default="unknown", help="resolved source commit; unknown is recorded honestly")
    p.add_argument("--config", type=Path, default=ROOT / "configs/workloads.yaml")
    p.add_argument("--requests", type=positive_int,
                   help="formal requests to generate per case, excluding warmup; overrides config requests")
    p.add_argument("--output", type=Path, default=ROOT / "data/frozen")
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("Output exists; use a new directory to preserve the frozen experiment")
    protocol = read(a.config)
    if a.requests is not None:
        protocol["requests"] = a.requests
    if type(protocol.get("requests")) is not int or protocol["requests"] < 1:
        p.error("Config requests must be a positive integer, or pass --requests N")
    from transformers import AutoTokenizer
    model = a.model.resolve()
    config = read(model / "config.json")
    if config.get("quantization_config"):
        p.error("Prepare the primary snapshot from the unquantized base model")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    template = tokenizer.get_chat_template()
    if "enable_thinking" not in template:
        p.error("Template does not support the required thinking switch")
    case_count = len(protocol["cases"])
    total = case_count * (protocol["requests"] + protocol["warmup"])
    print(f"Preparing {case_count} cases: {protocol['requests']} formal + {protocol['warmup']} warmup "
          f"requests per case ({total} total).", flush=True)
    prepared = []
    for case in protocol["cases"]:
        for index in range(protocol["requests"] + protocol["warmup"]):
            item = exact_prompt(tokenizer, case["input_tokens"], index)
            prepared.append(dict(item, id=f"{case['id']}-{index:04d}", case=case["id"],
                                 phase="warmup" if index < protocol["warmup"] else "measure",
                                 input_tokens=case["input_tokens"], output_tokens=case["output_tokens"]))
    model_files = sorted(set(model.glob("*.safetensors")) | set(model.glob("*.json")) |
                         set(model.glob("*.jinja")) | set(model.glob("*.txt")))
    if not list(model.glob("*.safetensors")):
        p.error("No safetensors weights found")
    print("Hashing local weights and tokenizer files; this may take several minutes.", flush=True)
    fingerprints = {f.name: digest(f) for f in model_files}
    a.output.mkdir(parents=True)
    write_rows(a.output / "requests.jsonl", prepared)
    write(a.output / "workloads.yaml", protocol)
    (a.output / "chat_template.jinja").write_text(template, encoding="utf-8")
    for name in ("quality_holdout.jsonl", "quality_protocol.json"):
        (a.output / name).write_bytes((ROOT / "data" / name).read_bytes())
    layers = config["num_hidden_layers"]
    kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    kv_per_token = 2 * layers * kv_heads * head_dim * 2
    # Parameter count from safetensors headers, without allocating weight tensors.
    import struct, math
    parameters = 0
    for f in model.glob("*.safetensors"):
        with f.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            import json
            header = json.loads(stream.read(size))
        parameters += sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
    write(a.output / "model.json", {"path": str(model), "source_revision": a.revision,
          "identity": "sha256 of local files; unknown revision does not imply a source commit",
          "files": fingerprints, "config": config, "parameter_count_stored": parameters,
          "bf16_weight_payload_bytes": parameters * 2, "kv_bytes_per_token_bf16": kv_per_token,
          "kv_budget": {c["id"]: kv_per_token * (c["input_tokens"] + c["output_tokens"])
                        for c in protocol["cases"]},
          "budget_note": "concurrency=1 payload only; excludes allocator, paging, activations, workspace and runtime"})
    names = [f.name for f in a.output.iterdir() if f.is_file()]
    write(a.output / "manifest.json", {"schema_version": 1, "scorer_sha256": digest(ROOT / "scripts/evaluate_quality.py"), "files": {n: digest(a.output / n) for n in names}})
    print("Frozen:", a.output)

if __name__ == "__main__":
    main()
