#!/usr/bin/env python3
"""Managed BF16/AWQ/official-FP8/local-FP8 comparison on frozen token requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid
from experiment_utils import ROOT, digest, read, write, load_snapshot, failure_kind
from test_server import Client, summarize_requests, percentile, redact
from model_profiles import (PROFILE_NAMES, canonical_profile, resolve_model_alias,
                            inspect_model_config, resolve_runtime_options, quant_method,
                            profile_from_model_name, validate_model_profile, FP8_BACKENDS,
                            validate_fp8_backend, fp8_backend_environment)


def now():
    return datetime.now(timezone.utc).isoformat()


class GPURecorder:
    """Device-wide sampled memory, not process allocation or exact transient peak."""
    def __init__(self, path, interval):
        self.path, self.interval = path, interval
        self.phase = "before_load"
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        record = {"utc": now(), "phase": self.phase}
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
            record["gpus"] = []
            for line in out.stdout.splitlines():
                uuid_, name, total, used, free, util = [x.strip() for x in line.split(",")]
                record["gpus"].append(dict(uuid=uuid_, name=name, total_mib=float(total),
                    used_mib=float(used), free_mib=float(free), utilization_percent=util))
        except Exception as e:
            record["error"] = str(e)
        self.samples.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=7)


def client_args(config):
    return SimpleNamespace(base_url=f"http://127.0.0.1:{config['port']}",
        timeout=config["request_socket_timeout_seconds"], api_key_env="VLLM_API_KEY", no_proxy=True)


def stop_process(process):
    if process is None:
        return
    # Only signal the process group created by this run, never an existing service.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The API parent may exit before workers. Clean up remaining members too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def launch_command(config, case):
    command = [sys.executable, str(ROOT / "scripts/run_model.py"), "--serve", "--backend", "vllm",
        "--model", config["model"], "--dtype", config["dtype"], "--host", config["host"],
        "--port", str(config["port"]), "--served-model-name", config["served_model_name"],
        "--max-model-len", str(case["input_tokens"] + case["output_tokens"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
        "--seed", "42", "--enforce-eager" if config["enforce_eager"] else "--no-enforce-eager"]
    if config.get("tokenizer"):
        command.extend(["--tokenizer", config["tokenizer"]])
    if config.get("quantization") not in (None, "auto"):
        command.extend(["--quantization", config["quantization"]])
    if config.get("profile", "bf16").startswith("fp8-"):
        command.extend(["--fp8-backend", config.get("fp8_backend", "auto")])
    return command


def perform(args, config, item, round_id, phase):
    started = time.perf_counter()
    entry = {"id": item["id"], "case": item["case"], "model_profile": config.get("profile", "bf16"),
             "round": round_id, "phase": phase,
             "submitted_utc": now(), "started_monotonic": started, "ok": False}
    try:
        payload = {"model": config["served_model_name"], "prompt": item["prompt_token_ids"],
            "max_tokens": item["output_tokens"], "min_tokens": item["output_tokens"],
            "ignore_eos": True, "temperature": 0, "top_p": 1, "top_k": -1,
            "repetition_penalty": 1, "seed": 42}
        result = Client(args).stream_chat(payload, path="/v1/completions")
        entry["result"] = result
        usage = result.get("usage") or {}
        if usage.get("prompt_tokens") != item["input_tokens"]:
            raise ValueError("Input token mismatch with frozen rendered prompt")
        if usage.get("completion_tokens") != item["output_tokens"]:
            raise ValueError("Output token mismatch with fixed-length protocol")
        metrics = result.get("server_metrics") or {}
        if (not isinstance(metrics.get("inference_seconds"), (float, int))
                or metrics.get("output_tokens") != item["output_tokens"]
                or metrics.get("input_tokens") != item["input_tokens"]
                or metrics.get("output_tokens_per_second") is None):
            raise ValueError("Missing or inconsistent server profiling metrics")
        entry["ok"] = True
    except Exception as e:
        entry.update(error=str(e), failure_kind=failure_kind(str(e)))
    entry["finished_monotonic"] = time.perf_counter()
    entry["elapsed_seconds"] = entry["finished_monotonic"] - started
    return redact(entry, os.environ.get("VLLM_API_KEY", ""))


def phase(args, config, items, round_id, phase_name, concurrency, output):
    entries = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        jobs = [pool.submit(perform, args, config, item, round_id, phase_name) for item in items]
        for job in as_completed(jobs):
            entry = job.result()
            entries.append(entry)
            with output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    elapsed = max(x["finished_monotonic"] for x in entries) - min(x["started_monotonic"] for x in entries)
    summary = summarize_requests(entries, elapsed, concurrency)
    summary.update(round=round_id, phase=phase_name, timing_scope="first_submit_to_last_finish_including_failures")
    for key in ("time_to_first_text_seconds", "tpot_text_event_seconds"):
        values = [e["result"]["client_metrics"][key] for e in entries
                  if e["ok"] and e["result"]["client_metrics"][key] is not None]
        summary[key + "_p50"] = percentile(values, .5)
        summary[key + "_p95"] = percentile(values, .95)
    return summary


def validate(config, protocol, model=None):
    profile = config.get("profile", "bf16")
    if profile not in PROFILE_NAMES:
        raise ValueError("Unknown model profile: " + profile)
    if profile == "bf16" and (config["dtype"] != "bfloat16" or config.get("quantization") not in (None, "auto")):
        raise ValueError("BF16 baseline requires unquantized weights and explicit bfloat16")
    if type(config["enforce_eager"]) is not bool:
        raise ValueError("enforce_eager must be a JSON boolean, not a string")
    backend = config.get("fp8_backend", "auto")
    if backend not in FP8_BACKENDS or (backend != "auto" and not profile.startswith("fp8-")):
        raise ValueError("fp8_backend must be auto, or marlin for an FP8 model")
    if model is not None:
        validate_fp8_backend(model["config"], backend)
        validate_model_profile(model["config"], profile)
        resolve_runtime_options(model["config"], config["dtype"], config.get("quantization") or "auto")
    if config["prefix_caching"] or protocol["prefix_caching"] or protocol["thinking"]:
        raise ValueError("Protocol requires prefix caching and thinking disabled")
    if config["kv_cache_dtype"] != "auto":
        raise ValueError("This runner fixes KV dtype to auto (same as model activation dtype)")
    for key, minimum in (("requests", 1), ("warmup", 5), ("rounds", 1), ("concurrency", 1)):
        if type(protocol[key]) is not int or protocol[key] < minimum:
            raise ValueError("Require requests>0, warmup>=5, rounds>=1, concurrency>0")
    if protocol["seed"] != 42 or protocol["temperature"] != 0 or not protocol["ignore_eos"] or not protocol["min_tokens_equals_max_tokens"]:
        raise ValueError("Unsupported generation protocol")
    if config["max_num_seqs"] < protocol["concurrency"]:
        raise ValueError("Scheduler concurrency is smaller than client concurrency")
    if not 0 < config["gpu_memory_utilization"] <= 1 or config["gpu_sample_interval_seconds"] <= 0:
        raise ValueError("Invalid GPU memory fraction/sample interval")


def select_requests(items, snapshot_protocol, cases, requests=50, rounds=1):
    """Select a deterministic prefix without modifying the original frozen snapshot."""
    if type(requests) is not int or requests < 1 or type(rounds) is not int or rounds < 1:
        raise ValueError("requests and rounds must be positive integers")
    protocol = dict(snapshot_protocol, requests=requests, rounds=rounds,
        order="selected case order; round order 1..rounds; request ID ascending; same requests each model/round")
    selected = []
    for case in cases:
        for stage, count in (("warmup", protocol["warmup"]), ("measure", requests)):
            group = sorted((x for x in items if x["case"] == case["id"] and x["phase"] == stage), key=lambda x: x["id"])
            expected = snapshot_protocol["warmup"] if stage == "warmup" else snapshot_protocol["requests"]
            if len(group) != expected or len({x["id"] for x in group}) != expected:
                raise ValueError("Snapshot request count/ID mismatch: " + case["id"])
            if count > len(group):
                raise ValueError(f"Requested {count} {stage} requests but snapshot has {len(group)}; prepare a new larger snapshot")
            if any(len(x["prompt_token_ids"]) != case["input_tokens"] or x["input_tokens"] != case["input_tokens"]
                   or x["output_tokens"] != case["output_tokens"] for x in group):
                raise ValueError("Snapshot token length mismatch")
            selected.extend(group[:count])
    sha = hashlib.sha256(json.dumps(selected, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8")).hexdigest()
    return protocol, selected, sha


def load_configs(args, source_model):
    if args.config:
        config = read(args.config)
        named_profile = profile_from_model_name(config["model"])
        config["model"] = resolve_model_alias(config["model"])
        if "profile" not in config:
            method = config.get("quantization") or "auto"
            if named_profile is not None:
                config["profile"] = named_profile
            elif method in ("awq", "awq_marlin"):
                config["profile"] = "awq"
            elif method in ("fp8", "compressed-tensors"):
                config["profile"] = "fp8-local"
            else:
                config["profile"] = inspect_model_config(read(Path(config["model"]) / "config.json"))
        configs = [config]
    else:
        profiles = list(PROFILE_NAMES) if args.all_models else [canonical_profile(x) for x in (args.model or ["bf16"])]
        if len(set(profiles)) != len(profiles):
            raise ValueError("Duplicate model selection")
        configs = []
        for profile in profiles:
            filename = "fp8.yaml" if profile == "fp8-local" else profile + ".yaml"
            config = read(ROOT / "configs" / filename)
            config.update(profile=profile, model=resolve_model_alias(profile))
            configs.append(config)
    for config in configs:
        config["model"] = resolve_model_alias(config["model"])
        config.setdefault("profile", "bf16")
        config["profile"] = canonical_profile(config["profile"])
        # Direct token-ID requests must share the frozen base tokenizer across models.
        if not config.get("tokenizer"):
            config["tokenizer"] = config["model"] if config["profile"] == "bf16" else source_model["path"]
        config["tokenizer"] = resolve_model_alias(config["tokenizer"])
        if getattr(args, "enforce_eager", None) is not None:
            config["enforce_eager"] = args.enforce_eager
        if config["profile"].startswith("fp8-"):
            config.setdefault("fp8_backend", "auto")
            if getattr(args, "fp8_backend", None) is not None:
                config["fp8_backend"] = args.fp8_backend
        fp8_backend_environment(config.get("fp8_backend", "auto"))
    if (getattr(args, "fp8_backend", None) == "marlin"
            and not any(c["profile"].startswith("fp8-") for c in configs)):
        raise ValueError("--fp8-backend marlin requires selecting an FP8 model")
    return configs


def verify_model(config, source_model):
    """Hash the selected weights separately from the BF16 workload source."""
    directory = Path(config["model"])
    actual = read(directory / "config.json")
    profile = validate_model_profile(actual, config["profile"])
    validate_fp8_backend(actual, config.get("fp8_backend", "auto"))
    runtime = resolve_runtime_options(actual, config["dtype"], config.get("quantization") or "auto")
    config.update(dtype=runtime["dtype"], quantization=runtime["quantization"])
    if not any(directory.glob("*.safetensors")) and not any(directory.glob("*.bin")):
        raise ValueError("No local model weights found: " + str(directory))
    print("Hashing selected model:", profile, directory, flush=True)
    files = {f.name: digest(f) for f in sorted(directory.iterdir())
             if f.is_file() and f.suffix in (".safetensors", ".bin", ".json", ".jinja", ".txt")}
    if profile == "bf16":
        for name, sha in source_model["files"].items():
            if files.get(name) != sha:
                raise ValueError("Model/tokenizer changed since freeze: " + name)
    else:
        for key in ("model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings",
                    "rope_theta", "rope_scaling", "max_position_embeddings"):
            if key in source_model["config"] and actual.get(key) != source_model["config"][key]:
                raise ValueError("Quantized model architecture differs from frozen source: " + key)
    # Verify tokenizer/config files at the explicit base tokenizer location; do not
    # compare quantized shards with BF16 hashes or re-tokenize using another vocab.
    tokenizer_files = {name: sha for name, sha in source_model["files"].items()
                       if not name.endswith((".safetensors", ".bin")) and "safetensors.index" not in name}
    if not any(name in tokenizer_files for name in ("tokenizer.json", "vocab.json", "tokenizer.model")):
        raise ValueError("Frozen source has no tokenizer fingerprint")
    for name, sha in tokenizer_files.items():
        if digest(Path(config["tokenizer"]) / name) != sha:
            raise ValueError("Reference tokenizer changed since freeze: " + name)
    identity = {"path": str(directory.resolve()), "profile": profile, "config": actual, "files": files,
        "quant_method": quant_method(actual), "tokenizer_path": str(Path(config["tokenizer"]).resolve()),
        "tokenizer_sha256": tokenizer_files,
        "base_source_verification": "exact_frozen_files" if profile == "bf16" else "architecture_only; original base revision unproven"}
    manifest_path = directory / "quantization_manifest.json"
    if profile == "fp8-local" and manifest_path.exists():
        conversion = read(manifest_path)
        if conversion.get("status") != "complete":
            raise ValueError("Local FP8 conversion manifest is not complete")
        if any(conversion.get("source_sha256", {}).get(name) != sha for name, sha in source_model["files"].items()):
            raise ValueError("Local FP8 conversion source differs from frozen BF16 source")
        expected = conversion.get("output_sha256", {})
        current = {name: sha for name, sha in files.items() if name != "quantization_manifest.json"}
        if not expected or current != expected:
            raise ValueError("Local FP8 weights/config differ from conversion manifest")
        identity["base_source_verification"] = "conversion_manifest_matches_frozen_source"
        identity["conversion_manifest_sha256"] = files["quantization_manifest.json"]
    return identity


def verify_engine(startup, config, model):
    """Verify resolved engine settings, without claiming individual CUDA kernels."""
    lines = [line for line in startup.splitlines() if "LLM engine" in line and "config:" in line]
    if not lines:
        raise ValueError("Cannot find vLLM engine initialization config in startup log")
    line = lines[-1]
    dtype = re.search(r"\bdtype\s*=\s*(?:torch\.)?['\"]?(bfloat16|float16|float32)\b", line)
    quant = re.search(r"\bquantization\s*=\s*['\"]?([a-zA-Z0-9_-]+)", line)
    if not dtype or dtype.group(1) != config["dtype"]:
        raise ValueError("Cannot verify resolved model dtype from engine startup log")
    expected = config.get("quantization")
    if expected in (None, "auto"):
        expected = quant_method(model["config"]) or "None"
    if not quant or quant.group(1).replace("_", "-") != expected.replace("_", "-"):
        raise ValueError("Cannot verify resolved quantization from engine startup log")
    verified = {"dtype": dtype.group(1), "quantization": quant.group(1),
                "source": "vLLM engine initialization config", "cuda_kernel_verified": False}
    if config.get("fp8_backend") == "marlin":
        if "leveraging the Marlin kernel" not in startup:
            raise ValueError("Cannot verify requested FP8 Marlin initialization in startup log")
        verified.update(fp8_backend_requested="marlin", execution_precision="W8A16",
                        backend_initialization_verified=True,
                        backend_source="vLLM FP8 Marlin initialization message")
    return verified


def run_case(config, protocol, case, items, directory, model):
    case_dir = directory / config["profile"] / case["id"]
    case_dir.mkdir(parents=True)
    record = {"case": case["id"], "model_profile": config["profile"],
        "artifact_directory": case_dir.relative_to(directory).as_posix(),
        "status": "incomplete", "rounds": [], "command": launch_command(config, case),
        "enforce_eager": config["enforce_eager"],
        "fp8_backend_requested": config.get("fp8_backend", "auto"),
        "server_environment_overrides": {"VLLM_TEST_FORCE_FP8_MARLIN":
            "1" if config.get("fp8_backend") == "marlin" else "0"}}
    process = None
    gpu = GPURecorder(case_dir / "gpu.jsonl", config["gpu_sample_interval_seconds"])
    args = client_args(config)
    try:
        with gpu, (case_dir / "server.log").open("w", encoding="utf-8") as log:
            gpu.phase = "loading"
            process = subprocess.Popen(record["command"], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env=fp8_backend_environment(config.get("fp8_backend", "auto")))
            deadline = time.monotonic() + config["startup_timeout_seconds"]
            probe_args = SimpleNamespace(**vars(args)); probe_args.timeout = 2
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server startup failed: " + (case_dir / "server.log").read_text(encoding="utf-8")[-12000:])
                try:
                    Client(probe_args).health()
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup timeout")
                    time.sleep(1)
            startup = (case_dir / "server.log").read_text(encoding="utf-8")
            record["engine_verification"] = verify_engine(startup, config, model)
            record["bf16_verified"] = config["profile"] == "bf16"
            gpu.phase = "loaded"
            time.sleep(max(1, config["gpu_sample_interval_seconds"] * 2))
            for round_id in range(1, protocol["rounds"] + 1):
                for stage in ("warmup", "measure"):
                    gpu.phase = f"round_{round_id}_{stage}"
                    group = [x for x in items if x["case"] == case["id"] and x["phase"] == stage]
                    result = phase(args, config, group, round_id, stage, protocol["concurrency"], case_dir / "requests.jsonl")
                    record["rounds"].append(result)
                    write(case_dir / "summary.json", record)
                    print(config["profile"], case["id"], round_id, stage, result["successful_requests"], "/", len(group), flush=True)
                    if stage == "warmup" and result["failed_requests"]:
                        raise RuntimeError("Warmup failed; inspect requests.jsonl; formal round not started")
            record["status"] = "passed" if all(not x["failed_requests"] for x in record["rounds"]) else "failed"
    except Exception as error:
        record.update(status="failed", error=redact(str(error), os.environ.get("VLLM_API_KEY", "")), failure_kind=failure_kind(str(error)))
    finally:
        stop_process(process)
        peaks = {}
        for sample in gpu.samples:
            for device in sample.get("gpus", []):
                key = sample["phase"] + ":" + device["uuid"]
                peaks[key] = max(peaks.get(key, 0), device["used_mib"])
        record["device_sampled_peak_mib_by_phase_uuid"] = peaks
        record["memory_note"] = "whole-device sampled use; includes other processes; not exact transient/process peak"
        if not peaks:
            record.update(status="failed", memory_error="No usable nvidia-smi samples")
        write(case_dir / "summary.json", record)
    return record


def check_runtime_versions():
    """Accept complete pinned stacks, never a mixture of old/new versions."""
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")}
    key = (versions["torch"], versions["transformers"], versions["vllm"].split("+")[0])
    if key == ("2.8.0+cu129", "4.57.6", "0.11.0"):
        return {"name": "legacy-cutedsl", "versions": versions}
    if key == ("2.9.1+cu128", "4.57.3", "0.14.1"):
        versions["compressed-tensors"] = importlib.metadata.version("compressed-tensors")
        if versions["compressed-tensors"] != "0.13.0":
            raise ValueError("Unified stack requires compressed-tensors==0.13.0")
        return {"name": "unified-quantization-inference", "versions": versions}
    raise ValueError("Unsupported runtime version combination: " + json.dumps(versions)
                     + "; use requirements-vllm-0112.txt for the unified environment or the complete legacy stack")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def boolean_value(value):
    normalized = value.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("use true or false (also accepts 1 or 0)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=ROOT / "data/frozen")
    choices = p.add_mutually_exclusive_group()
    choices.add_argument("--config", type=Path, help="custom single-model config")
    choices.add_argument("--model", action="append", help="bf16, awq, fp8-official/fp8-origin, fp8-local or qwen3-8b aliases; repeat to compare")
    choices.add_argument("--all-models", action="store_true", help="sequentially compare BF16, AWQ, official FP8 and local FP8")
    p.add_argument("--rounds", type=positive_int, default=1, help="rounds per model/case (default: 1)")
    p.add_argument("--requests", type=positive_int, default=50, help="formal requests per round (default: 50)")
    p.add_argument("--case", action="append", help="select case IDs; omitted runs all five")
    eager = p.add_mutually_exclusive_group()
    eager.add_argument("--enforce-eager", nargs="?", const=True, type=boolean_value, default=None,
                       metavar="{true,false}", help="override config eager mode; bare flag means true")
    eager.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false", default=None,
                       help="override config with enforce_eager=false (allow CUDA Graph)")
    p.add_argument("--fp8-backend", choices=FP8_BACKENDS, default=None,
                   help="FP8 models only: auto follows vLLM dispatch; marlin requests native-FP8 W8A16 (v0.14.1)")
    p.add_argument("--plan-only", action="store_true", help="validate snapshot hashes and print plan without loading/hash-checking model weights")
    a = p.parse_args(argv)
    manifest, items, frozen_protocol = load_snapshot(a.snapshot)
    source_model = read(a.snapshot / "model.json")
    configs = load_configs(a, source_model)
    cases = [c for c in frozen_protocol["cases"] if not a.case or c["id"] in a.case]
    if not cases or (a.case and set(a.case) - {c["id"] for c in cases}):
        p.error("Unknown case ID")
    protocol, items, subset_sha = select_requests(items, frozen_protocol, cases, a.requests, a.rounds)
    for config in configs:
        validate(config, protocol)
    print(json.dumps({"models": [c["profile"] for c in configs], "rounds": protocol["rounds"],
        "formal_requests_per_round": protocol["requests"], "warmup_per_round": protocol["warmup"],
        "cases": [c["id"] for c in cases], "selected_requests_sha256": subset_sha,
        "selected_ids": [x["id"] for x in items],
        "commands": [{"model_profile": config["profile"], "case": case["id"], "command": launch_command(config, case),
                      "enforce_eager": config["enforce_eager"],
                      "fp8_backend_requested": config.get("fp8_backend", "auto"),
                      "environment_overrides": {"VLLM_TEST_FORCE_FP8_MARLIN":
                          "1" if config.get("fp8_backend") == "marlin" else "0"}}
                     for config in configs for case in cases],
        "model_verification": "pending: actual weights and engine settings checked during execution"}, ensure_ascii=False, indent=2), flush=True)
    if a.plan_only:
        return 0
    if sys.platform != "linux":
        p.error("Run managed GPU experiments on the Linux target host; use --plan-only locally")
    runtime = check_runtime_versions()
    if (any(c.get("fp8_backend") == "marlin" for c in configs)
            and runtime["versions"]["vllm"].split("+")[0] != "0.14.1"):
        p.error("--fp8-backend marlin is audited for vLLM 0.14.1; use the unified runtime")
    for config in configs:
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", config["port"]))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = ROOT / "log" / ("benchmark_" + run_id)
    directory.mkdir(parents=True)
    run = {"schema_version": 2, "run_id": run_id, "started": now(),
        "runtime": runtime,
        "models": [{"profile": c["profile"], "config": c, "model": None} for c in configs],
        "protocol": protocol, "snapshot_protocol": frozen_protocol,
        "manifest_sha256": digest(a.snapshot / "manifest.json"),
        "selected_requests_sha256": subset_sha, "selected_ids": [x["id"] for x in items],
        "snapshot": str(a.snapshot.resolve()), "workload_source_model": source_model,
        "cases": [c["id"] for c in cases]}
    write(directory / "run.json", run)
    # Save the actual selected request bodies as well as their digest and IDs.
    from experiment_utils import write_rows
    write_rows(directory / "selected_requests.jsonl", items)
    with (directory / "environment.log").open("w", encoding="utf-8") as out:
        subprocess.run([sys.executable, str(ROOT / "scripts/collect_environment.py"),
            "--output", str(directory / "environment.json")], stdout=out, stderr=subprocess.STDOUT, timeout=180, check=True)
    environment = read(directory / "environment.json")
    devices = environment.get("checks", {}).get("torch", {}).get("runtime", {}).get("devices", [])
    if not any("4090d" in d.get("name", "").lower().replace(" ", "") and d.get("bf16_supported") for d in devices):
        write(directory / "summary.json", {"run_id": run_id, "cases": [],
            "error": "No visible BF16-capable RTX 4090D; inspect environment.json"})
        print("Target GPU precheck failed; artifacts:", directory)
        return 1
    code = {f.name: digest(f) for f in (ROOT / "scripts").glob("*.py")}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=10)
    write(directory / "code.json", {"scripts_sha256": code, "commit": git.stdout.strip(), "worktree": status.stdout})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60, check=True)
    (directory / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    results = []
    for descriptor in run["models"]:
        config = descriptor["config"]
        try:
            descriptor["model"] = verify_model(config, source_model)
            validate(config, protocol, descriptor["model"])
        except Exception as error:
            descriptor["validation_error"] = str(error)
        write(directory / "run.json", run)
        for case in cases:
            if "validation_error" in descriptor:
                record = {"case": case["id"], "model_profile": config["profile"], "rounds": [],
                    "artifact_directory": config["profile"] + "/" + case["id"], "status": "failed",
                    "error": descriptor["validation_error"], "failure_kind": "model_precheck"}
                write(directory / record["artifact_directory"] / "summary.json", record)
            else:
                record = run_case(config, protocol, case, items, directory, descriptor["model"])
            results.append(record)
            write(directory / "summary.json", {"run_id": run_id, "cases": results})
    print("Artifacts:", directory)
    return 0 if results and all(x["status"] == "passed" for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
