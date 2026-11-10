#!/usr/bin/env python3
"""Launch the standalone W31 service with the official vLLM API server.

Dry runs use only the standard library and do not require downloaded weights.
Actual serving targets this project's locked Linux/CUDA runtime. All model
resolution and validation are implemented inside runtime_training.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys

from model_config import QUANTIZATIONS, prepare_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_VLLM_VERSION = "0.14.1"
DEFAULT_CONFIG = {
    "model": "qwen3-8b",
    "served_model_name": "qwen3-8b",
    "dtype": "bfloat16",
    "host": "127.0.0.1",
    "port": 7081,
    "max_model_len": 2048,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.85,
    "tensor_parallel_size": 1,
    "enforce_eager": True,
    "seed": 42,
    "quantization": "auto",
    "trust_remote_code": False,
}
OPTIONAL_KEYS = {"tokenizer"}
FIXED_SETTINGS = {
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "generation_config": "vllm",
    "force_include_usage": True,
}


def validate_config(config):
    """Return a validated, complete configuration; reject misspelled fields."""
    if not isinstance(config, dict):
        raise ValueError("Server config must be a JSON object")
    unknown = set(config) - set(DEFAULT_CONFIG) - OPTIONAL_KEYS
    if unknown:
        raise ValueError("Unknown server config keys: " + ", ".join(sorted(unknown)))
    result = {**DEFAULT_CONFIG, **config}
    for key in ("model", "served_model_name", "host", "quantization"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if "tokenizer" in result and (
        not isinstance(result["tokenizer"], str) or not result["tokenizer"].strip()
    ):
        raise ValueError("tokenizer must be a nonempty string when supplied")
    if result["quantization"] not in QUANTIZATIONS:
        raise ValueError("quantization must be auto, awq, or awq_marlin; W31 does not support FP8")
    if result["dtype"] not in ("auto", "bfloat16", "float16", "float32"):
        raise ValueError("dtype must be auto, bfloat16, float16, or float32")
    for key in ("enforce_eager", "trust_remote_code"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "tensor_parallel_size"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(result["port"]) is not int or not 1 <= result["port"] <= 65535:
        raise ValueError("port must be an integer in [1, 65535]")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    memory = result["gpu_memory_utilization"]
    if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    if result["max_num_batched_tokens"] < result["max_num_seqs"]:
        raise ValueError("max_num_batched_tokens must be at least max_num_seqs")
    return result


def resolve_path(path):
    return Path(path).expanduser().resolve()


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def build_command(config, *, python_executable=None):
    """Build a direct upstream argv without a shell or external project code."""
    config, _ = prepare_model(validate_config(config), require_files=False)
    command = [str(python_executable or sys.executable), "-m", "vllm.entrypoints.openai.api_server"]
    for key in (
        "model", "tokenizer", "served_model_name", "dtype", "host", "port", "max_model_len",
        "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
        "tensor_parallel_size", "seed",
    ):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    if config["quantization"] != "auto":
        command.extend(["--quantization", config["quantization"]])
    command.extend(["--generation-config", "vllm", "--no-enable-prefix-caching",
                    "--enable-chunked-prefill", "--enable-force-include-usage"])
    command.append("--enforce-eager" if config["enforce_eager"] else "--no-enforce-eager")
    if config["trust_remote_code"]:
        command.append("--trust-remote-code")
    return command


def server_environment():
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return environment


def package_versions():
    versions = {}
    for package in ("vllm", "torch", "transformers", "tokenizers", "httpx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def require_vllm_version(version):
    # Local build suffixes do not change the pinned upstream release.
    if not version or version.split("+", 1)[0] != REQUIRED_VLLM_VERSION:
        detected = version or "not installed"
        raise ValueError(
            f"W31 requires vLLM {REQUIRED_VLLM_VERSION}; detected {detected}. "
            "Install this project's requirements-server.txt or use --dry-run to inspect the command without execution."
        )


def gpu_inventory():
    """Read driver inventory without importing torch or initializing CUDA."""
    query = ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(query, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    if result.returncode:
        return {"available": False, "reason": result.stderr.strip(), "returncode": result.returncode}
    return {"available": True, "query": query, "csv_rows": result.stdout.strip().splitlines()}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def command_display(command):
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stop_process_group(process):
    """Give the wrapper and API process a graceful interrupt before escalation."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=15)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def run_logged(command, log, on_started=None):
    environment = server_environment()
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", bufsize=1, env=environment, **options)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    termination_signal = None

    def interrupted(signum, frame):
        nonlocal termination_signal
        termination_signal = signum
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, interrupted)
        if on_started is not None:
            on_started(process)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait(), False
    except KeyboardInterrupt:
        log.write("\n[W31 launcher] Interrupted; stopping the server process group.\n")
        log.flush()
        stop_process_group(process)
        return (128 + termination_signal if termination_signal is not None else 130), True
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
        if process.poll() is None:
            stop_process_group(process)
        if process.stdout is not None:
            process.stdout.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "server_w31.json")
    parser.add_argument("--model", help="Override the model alias/local path; API model name remains served_model_name")
    parser.add_argument("--output-dir", type=Path, help="New output directory for manifest.json and startup.log")
    parser.add_argument("--dry-run", action="store_true", help="Validate and record the launch plan without starting vLLM")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        config_path = resolve_path(args.config)
        config = load_config(config_path)
        if args.model is not None:
            config = validate_config({**config, "model": args.model})
        runtime_config, model_info = prepare_model(config, require_files=False)
        command = build_command(runtime_config)
        output_dir = resolve_path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / (
            "server-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        if (output_dir / "manifest.json").exists() or (output_dir / "startup.log").exists():
            raise ValueError(f"Output directory already contains a server run; choose a new --output-dir: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        versions = package_versions()
        manifest = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "status": "dry_run" if args.dry_run else "planned",
            "execution_started": False,
            "server_ready": None,
            "config_path": str(config_path),
            "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "effective_config_sha256": hashlib.sha256(json.dumps(runtime_config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "config": config,
            "runtime_config": runtime_config,
            "model_metadata": model_info,
            "fixed_server_settings": FIXED_SETTINGS,
            "required_vllm_version": REQUIRED_VLLM_VERSION,
            "runtime_target": {"python": "3.10-3.12", "platform": "Linux/CUDA", "vllm": REQUIRED_VLLM_VERSION},
            "runtime_check": "not_executed_dry_run" if args.dry_run else "pending",
            "versions": versions,
            "python": sys.executable,
            "platform": platform.platform(),
            "gpu_inventory": gpu_inventory(),
            "environment": {key: server_environment().get(key) for key in (
                "MODEL_ROOT", "CUDA_VISIBLE_DEVICES", "VLLM_WORKER_MULTIPROC_METHOD",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONUNBUFFERED",
            )},
            "command_argv": command,
            "command_display": command_display(command),
            "startup_log": str(output_dir / "startup.log"),
            "readiness_note": "Process launch is recorded; server readiness must be verified by client health/model checks.",
        }
        manifest_path = output_dir / "manifest.json"
        write_manifest(manifest_path, manifest)
        with (output_dir / "startup.log").open("x", encoding="utf-8") as log:
            header = f"Command: {command_display(command)}\nManifest: {manifest_path}\n"
            print(header, end="", flush=True)
            log.write(header)
            log.flush()
            if args.dry_run:
                message = "DRY RUN: no server was started; model paths, CUDA readiness, and attention backend are unverified.\n"
                print(message, end="", flush=True)
                log.write(message)
                return 0
            try:
                require_vllm_version(versions["vllm"])
                if not (3, 10) <= sys.version_info[:2] < (3, 13):
                    raise ValueError("W31 actual serving requires the project's Python 3.10-3.12 runtime; use --dry-run on this host.")
                if sys.platform != "linux":
                    raise ValueError("The locked W31 service requires Linux/CUDA (including WSL2); native Windows supports --dry-run only.")
                runtime_config, model_info = prepare_model(config, require_files=True)
                command = build_command(runtime_config)
                manifest.update(runtime_config=runtime_config, model_metadata=model_info,
                                status="launching", spawn_attempted=True, runtime_check="passed")
                write_manifest(manifest_path, manifest)

                def on_started(process):
                    manifest.update(status="running", execution_started=True, started_at_utc=utc_now(), launcher_child_pid=process.pid)
                    write_manifest(manifest_path, manifest)

                returncode, interrupted = run_logged(command, log, on_started=on_started)
                manifest.update(status="interrupted" if interrupted else ("exited" if returncode == 0 else "failed"),
                                returncode=returncode, finished_at_utc=utc_now())
                write_manifest(manifest_path, manifest)
                return returncode
            except (ValueError, OSError) as error:
                if manifest["runtime_check"] == "pending":
                    manifest["runtime_check"] = "rejected"
                manifest.update(status="rejected" if not manifest["execution_started"] else "failed",
                                error=str(error), finished_at_utc=utc_now(), returncode=1)
                write_manifest(manifest_path, manifest)
                log.write(f"ERROR: {error}\n")
                print(f"ERROR: {error}", file=sys.stderr)
                return 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
