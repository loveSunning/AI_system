#!/usr/bin/env python3
"""Managed BF16 baseline: run on the GPU host, never reuse an unknown server."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
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
    return [sys.executable, str(ROOT / "scripts/run_model.py"), "--serve", "--backend", "vllm",
        "--model", config["model"], "--dtype", "bfloat16", "--host", config["host"],
        "--port", str(config["port"]), "--served-model-name", config["served_model_name"],
        "--max-model-len", str(case["input_tokens"] + case["output_tokens"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
        "--seed", "42", "--enforce-eager" if config["enforce_eager"] else "--no-enforce-eager"]


def perform(args, config, item, round_id, phase):
    started = time.perf_counter()
    entry = {"id": item["id"], "case": item["case"], "round": round_id, "phase": phase,
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


def validate(config, protocol, model):
    if config["dtype"] != "bfloat16" or config["quantization"] is not None or model["config"].get("quantization_config"):
        raise ValueError("BF16 baseline requires unquantized weights and explicit bfloat16")
    if config["prefix_caching"] or protocol["prefix_caching"] or protocol["thinking"]:
        raise ValueError("Week-1 protocol requires prefix caching and thinking disabled")
    if config["kv_cache_dtype"] != "auto":
        raise ValueError("This runner fixes KV dtype to auto (same as model dtype)")
    if any(protocol[k] < v for k, v in (("requests", 1), ("warmup", 5), ("rounds", 3), ("concurrency", 1))):
        raise ValueError("Require requests>0, warmup>=5, rounds>=3, concurrency>0")
    if protocol["seed"] != 42 or protocol["temperature"] != 0 or not protocol["ignore_eos"] or not protocol["min_tokens_equals_max_tokens"]:
        raise ValueError("Unsupported generation protocol")
    if config["max_num_seqs"] < protocol["concurrency"]:
        raise ValueError("Scheduler concurrency is smaller than client concurrency")
    if not 0 < config["gpu_memory_utilization"] <= 1 or config["gpu_sample_interval_seconds"] <= 0:
        raise ValueError("Invalid GPU memory fraction/sample interval")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=ROOT / "data/frozen")
    p.add_argument("--config", type=Path, default=ROOT / "configs/bf16.yaml")
    p.add_argument("--case", action="append", help="select case IDs; omitted runs all five")
    p.add_argument("--plan-only", action="store_true", help="validate hashes and print commands without GPU execution")
    a = p.parse_args(argv)
    manifest, items, protocol = load_snapshot(a.snapshot)
    config = read(a.config)
    model = read(a.snapshot / "model.json")
    validate(config, protocol, model)
    cases = [c for c in protocol["cases"] if not a.case or c["id"] in a.case]
    if not cases or (a.case and set(a.case) - {c["id"] for c in cases}):
        p.error("Unknown case ID")
    for case in cases:
        for phase_name, count in (("warmup", protocol["warmup"]), ("measure", protocol["requests"])):
            group = [x for x in items if x["case"] == case["id"] and x["phase"] == phase_name]
            if len(group) != count or len({x["id"] for x in group}) != count:
                raise ValueError("Snapshot request count/ID mismatch")
            if any(len(x["prompt_token_ids"]) != case["input_tokens"] or x["input_tokens"] != case["input_tokens"]
                   or x["output_tokens"] != case["output_tokens"] for x in group):
                raise ValueError("Snapshot token length mismatch")
        print(json.dumps(launch_command(config, case)))
    if a.plan_only:
        return 0
    if sys.platform != "linux":
        p.error("Run managed GPU experiments on the Linux target host; use --plan-only locally")
    # Content identity, not mutable path names, connects prepared data to this model.
    for name, sha in model["files"].items():
        if digest(Path(config["model"]) / name) != sha:
            raise ValueError("Model/tokenizer changed since freeze: " + name)
    for package, version in {"torch": "2.8.0+cu129", "transformers": "4.57.6"}.items():
        if importlib.metadata.version(package) != version:
            raise ValueError("Unexpected package version: " + package)
    if importlib.metadata.version("vllm").split("+")[0] != "0.11.0":
        raise ValueError("Protocol validated for vLLM 0.11.0")
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", config["port"]))  # occupied port is an error, never kill/reuse it
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = ROOT / "log" / ("benchmark_" + run_id)
    directory.mkdir()
    write(directory / "run.json", {"run_id": run_id, "started": now(), "config": config,
        "protocol": protocol, "manifest_sha256": digest(a.snapshot / "manifest.json"),
        "snapshot": str(a.snapshot.resolve()), "model": model, "cases": [c["id"] for c in cases]})
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
    args = client_args(config)
    for case in cases:
        case_dir = directory / case["id"]
        case_dir.mkdir()
        record = {"case": case["id"], "status": "incomplete", "rounds": [], "command": launch_command(config, case)}
        process = None
        gpu = GPURecorder(case_dir / "gpu.jsonl", config["gpu_sample_interval_seconds"])
        try:
            with gpu, (case_dir / "server.log").open("w", encoding="utf-8") as log:
                gpu.phase = "loading"
                process = subprocess.Popen(record["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
                if not re.search(r"dtype\s*=\s*torch\.bfloat16", startup):
                    raise ValueError("Cannot verify resolved torch.bfloat16 from engine startup log")
                record["bf16_verified"] = True
                gpu.phase = "loaded"
                time.sleep(max(1, config["gpu_sample_interval_seconds"] * 2))
                for round_id in range(1, protocol["rounds"] + 1):
                    for stage in ("warmup", "measure"):
                        gpu.phase = f"round_{round_id}_{stage}"
                        group = [x for x in items if x["case"] == case["id"] and x["phase"] == stage]
                        summary = phase(args, config, group, round_id, stage, protocol["concurrency"], case_dir / "requests.jsonl")
                        record["rounds"].append(summary)
                        write(case_dir / "summary.json", record)
                        print(case["id"], round_id, stage, summary["successful_requests"], "/", len(group), flush=True)
                        if stage == "warmup" and summary["failed_requests"]:
                            raise RuntimeError("Warmup failed; inspect requests.jsonl; formal round not started")
                record["status"] = "passed" if all(not x["failed_requests"] for x in record["rounds"]) else "failed"
        except Exception as e:
            record.update(status="failed", error=redact(str(e), os.environ.get("VLLM_API_KEY", "")), failure_kind=failure_kind(str(e)))
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
            results.append(record)
            write(directory / "summary.json", {"run_id": run_id, "cases": results})
    print("Artifacts:", directory)
    return 0 if all(x["status"] == "passed" for x in results) else 1

if __name__ == "__main__":
    raise SystemExit(main())
