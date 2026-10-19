#!/usr/bin/env python3
"""Inspect the active Python environment and GPU without loading model weights.

Uses only the Python standard library. Independent probes run concurrently;
PyTorch is imported in a child process so a broken import or CUDA initialization
cannot hang the collector indefinitely. Package metadata does not prove that a
vLLM server can start.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024 ** 3
PACKAGES = (
    "torch", "vllm", "transformers", "tokenizers", "triton",
    "sglang", "modelscope", "huggingface-hub", "safetensors", "accelerate",
)
TORCH_MARKER = "__AI_SYSTEM_TORCH_JSON__"
TORCH_PROBE = r"""
import json
import torch

result = {
    "version": str(torch.__version__),
    "cuda_build_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "devices": [],
}
if result["cuda_available"]:
    result["compiled_arches"] = torch.cuda.get_arch_list()
    for index in range(torch.cuda.device_count()):
        device = {"logical_index": index}
        try:
            with torch.cuda.device(index):
                properties = torch.cuda.get_device_properties(index)
                device.update({
                    "name": properties.name,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_bytes": properties.total_memory,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                })
                try:
                    free, total = torch.cuda.mem_get_info(index)
                    device.update({"free_bytes": free, "total_bytes": total})
                except Exception as error:
                    device["memory_error"] = str(error)
        except Exception as error:
            device["error"] = str(error)
        result["devices"].append(device)
print("__AI_SYSTEM_TORCH_JSON__" + json.dumps(result))
"""


def run_command(command: list[str], timeout: float) -> dict:
    """Run an argument vector without a shell, retaining diagnostic output."""
    started = time.perf_counter()
    result = {"command": command}
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
        result.update(
            status="ok" if completed.returncode == 0 else "error",
            returncode=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )
    except FileNotFoundError:
        result.update(status="missing", error="Executable not found")
    except subprocess.TimeoutExpired:
        result.update(status="timeout", error=f"Probe exceeded {timeout:g} seconds")
    except OSError as error:
        result.update(status="error", error=str(error))
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def parse_gpu_csv(text: str) -> list[dict]:
    """Parse nvidia-smi memory values (MiB); unavailable fields remain null."""
    devices = []
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 7:
            raise ValueError(f"Expected 7 nvidia-smi fields, received {len(row)}")
        index, uuid, name, driver, total, used, free = (value.strip() for value in row)

        def memory_bytes(value: str):
            try:
                amount = float(value)
            except ValueError:
                return None
            return int(amount * 1024 ** 2) if math.isfinite(amount) and amount >= 0 else None

        devices.append({
            "physical_index": index, "uuid": uuid, "name": name,
            "driver_version": driver, "total_bytes": memory_bytes(total),
            "used_bytes": memory_bytes(used), "free_bytes": memory_bytes(free),
        })
    return devices


def probe_gpu(timeout: float) -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"status": "missing", "error": "nvidia-smi not found on PATH", "devices": []}
    query = run_command([
        executable,
        "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], timeout)
    if query["status"] != "ok":
        return {**query, "devices": []}
    try:
        devices = parse_gpu_csv(query["stdout"])
    except ValueError as error:
        return {**query, "status": "error", "error": str(error), "devices": []}
    header = run_command([executable], timeout)
    match = re.search(r"CUDA Version:\s*([0-9.]+)", header.get("stdout", ""))
    return {
        "status": "ok" if devices else "error", "devices": devices,
        "driver_cuda_max_version": match.group(1) if match else None,
        "query": query, "header": header,
        **({"error": "nvidia-smi returned no GPUs"} if not devices else {}),
    }


def probe_tool(name: str, timeout: float) -> dict:
    executable = shutil.which(name)
    if not executable:
        return {"status": "missing", "error": f"{name} not found on PATH"}
    result = run_command([executable, "--version"], timeout)
    if name == "nvcc" and result["status"] == "ok":
        match = re.search(r"release\s+([0-9.]+)", result.get("stdout", ""))
        result["version"] = match.group(1) if match else None
    return result


def probe_packages() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = {"status": "ok", "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"status": "missing", "version": None}
        except Exception as error:
            packages[name] = {"status": "error", "version": None, "error": str(error)}
    return {"status": "ok", "packages": packages}


def probe_torch(timeout: float) -> dict:
    result = run_command([sys.executable, "-c", TORCH_PROBE], timeout)
    if result["status"] != "ok":
        return result
    try:
        payload = result["stdout"].rsplit(TORCH_MARKER, 1)[1]
        result["runtime"] = json.loads(payload)
    except (IndexError, ValueError) as error:
        result.update(status="error", error=f"Cannot parse PyTorch probe: {error}")
    return result


def collect_checks(timeout: float, workers: int) -> dict:
    probes = {
        "packages": probe_packages,
        "gpu": lambda: probe_gpu(timeout),
        "torch": lambda: probe_torch(timeout),
        "nvcc": lambda: probe_tool("nvcc", timeout),
        "nsys": lambda: probe_tool("nsys", timeout),
        "ncu": lambda: probe_tool("ncu", timeout),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(probe): name for name, probe in probes.items()}
        for future in as_completed(pending):
            name = pending[future]
            try:
                results[name] = future.result()
            except Exception as error:
                results[name] = {"status": "error", "error": str(error)}
    return {name: results[name] for name in probes}


def normalize_gpu_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def assess(checks: dict, expected_gpu: str, min_free_gib: float, system_name: str) -> dict:
    issues = []
    packages = checks.get("packages", {}).get("packages", {})
    for required in ("torch", "vllm"):
        if packages.get(required, {}).get("status") != "ok":
            issues.append(f"{required} is unavailable in the active Python environment")
    torch_check = checks.get("torch", {})
    runtime = torch_check.get("runtime", {})
    if torch_check.get("status") != "ok":
        issues.append("PyTorch import/runtime probe failed; inspect its diagnostics")
    elif not runtime.get("cuda_available"):
        issues.append("torch.cuda.is_available() is False")
    # Only PyTorch-visible devices can satisfy the target. nvidia-smi can list
    # physical GPUs hidden from this process by CUDA_VISIBLE_DEVICES.
    visible = runtime.get("devices", [])
    target = normalize_gpu_name(expected_gpu)
    matches = [
        device for device in visible
        if not device.get("error") and
        (not target or target in normalize_gpu_name(device.get("name", "")))
    ]
    if not matches:
        issues.append(f"No PyTorch-visible GPU matches target {expected_gpu or '(any GPU)'}")
    elif not any(
        isinstance(device.get("free_bytes"), (int, float)) and
        device["free_bytes"] >= min_free_gib * GIB
        for device in matches
    ):
        issues.append(f"No matching visible GPU has at least {min_free_gib:g} GiB free")
    if system_name != "Linux":
        issues.append("Target vLLM deployment uses Linux/WSL2; this host is not Linux")
    return {
        "expected_gpu": expected_gpu, "min_free_gib": min_free_gib,
        "prechecks_passed": not issues, "issues": issues,
        "note": "Inventory and CUDA visibility checks only; no vLLM server/model smoke test.",
    }


def gib(value) -> str:
    return f"{value / GIB:.2f} GiB" if isinstance(value, (int, float)) else "unknown"


def diagnostic(check: dict) -> str:
    value = check.get("error") or check.get("stderr") or check.get("stdout") or "No details"
    # Keep terminal output concise; the JSON snapshot retains full diagnostics.
    return str(value)[-1200:]


def print_report(snapshot: dict) -> None:
    system = snapshot["system"]
    checks = snapshot["checks"]
    print("=== Qwen3-8B environment check (parallel probes) ===")
    print(f"Collected: {snapshot['collected_at_utc']}")
    print(f"OS: {system['platform']}")
    print(f"CPU: {system['machine']} / {system['logical_cpus']} logical CPUs")
    print(f"Python: {system['python_version']}")
    print(f"Interpreter: {system['python_executable']}")
    for name, value in system["environment"].items():
        print(f"{name}: {value if value is not None else '<unset>'}")
    print("\n--- Package versions (active interpreter metadata) ---")
    packages = checks["packages"].get("packages", {})
    for name in PACKAGES:
        package = packages.get(name, {})
        print(f"{name:18} {package.get('version') or package.get('status', 'unknown')}")

    print("\n--- Physical GPUs (nvidia-smi snapshot, may include hidden GPUs) ---")
    gpu = checks["gpu"]
    if gpu["status"] == "ok":
        for device in gpu["devices"]:
            print(f"GPU {device['physical_index']}: {device['name']} [{device['uuid']}]")
            print(
                f"  Driver: {device['driver_version']} | "
                f"total {gib(device['total_bytes'])} / used {gib(device['used_bytes'])} / "
                f"free {gib(device['free_bytes'])}"
            )
    else:
        print(f"nvidia-smi: {gpu['status']} - {diagnostic(gpu)}")

    print("\n--- Three different CUDA version sources ---")
    print(f"Driver's supported CUDA maximum (nvidia-smi): {gpu.get('driver_cuda_max_version') or 'unknown'}")
    nvcc = checks["nvcc"]
    print(f"Local CUDA Toolkit (nvcc): {nvcc.get('version') or nvcc['status']}")
    torch_check = checks["torch"]
    runtime = torch_check.get("runtime", {})
    cuda_build = runtime.get("cuda_build_version")
    cuda_label = cuda_build or ("CPU-only build" if runtime else "not collected (PyTorch probe failed)")
    print(f"PyTorch CUDA build: {cuda_label}")
    if nvcc["status"] not in ("ok", "missing"):
        print(f"nvcc diagnostics: {diagnostic(nvcc)}")
    print("These values describe different layers; exact equality is not required.")

    print("\n--- PyTorch runtime and visible GPUs ---")
    if torch_check["status"] == "ok":
        print(f"Imported torch: {runtime.get('version')}")
        print(f"CUDA available: {runtime.get('cuda_available')} | cuDNN: {runtime.get('cudnn_version')}")
        for device in runtime.get("devices", []):
            print(f"cuda:{device['logical_index']}: {device.get('name', 'unknown')}")
            if device.get("error"):
                print(f"  ERROR: {device['error']}")
                continue
            capability = ".".join(str(part) for part in device.get("compute_capability", []))
            print(
                f"  Compute capability: {capability} | BF16 support: {device.get('bf16_supported')} | "
                f"free {gib(device.get('free_bytes'))} / total {gib(device.get('total_bytes'))}"
            )
            if device.get("memory_error"):
                print(f"  Memory query error: {device['memory_error']}")
        if torch_check.get("stderr"):
            print(f"PyTorch diagnostics: {diagnostic({'stderr': torch_check['stderr']})}")
    else:
        print(f"PyTorch probe: {torch_check['status']} - {diagnostic(torch_check)}")

    print("\n--- Optional profiling tools ---")
    for name in ("nsys", "ncu"):
        check = checks[name]
        detail = check.get("stdout") or check.get("stderr") or check.get("error") or ""
        summary = " | ".join(detail.splitlines()[:4]) if detail else "unknown"
        print(f"{name}: {check['status']} - {summary}")

    assessment = snapshot["assessment"]
    print("\n--- Target prechecks ---")
    print(f"Target: {assessment['expected_gpu'] or '(any GPU)'}, >= {assessment['min_free_gib']:g} GiB free")
    print("Result: " + ("PASS" if assessment["prechecks_passed"] else "CHECK REQUIRED"))
    for issue in assessment["issues"]:
        print(f"  - {issue}")
    print(assessment["note"])
    print("Free memory is a point-in-time sample; PyTorch initialization and other processes can change it.")
    print("Package metadata confirms installation, not vLLM import/kernel compatibility.")
    print(f"Collection wall time: {snapshot['elapsed_seconds']:.2f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the full UTF-8 JSON snapshot")
    parser.add_argument("--timeout", type=float, default=30, help="Timeout per external command in seconds (default: 30)")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent probes (default: 6)")
    parser.add_argument("--expected-gpu", default="4090D", help="Expected GPU name fragment; empty accepts any GPU")
    parser.add_argument("--min-free-gib", type=float, default=20, help="Target free VRAM in GiB (default: 20)")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if target prechecks fail (inventory mode exits 0)")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.timeout) or not math.isfinite(args.min_free_gib)
            or args.timeout <= 0 or args.workers < 1 or args.min_free_gib < 0):
        parser.error("timeout/workers must be positive; min-free-gib must be nonnegative")
    started = time.perf_counter()
    checks = collect_checks(args.timeout, args.workers)
    snapshot = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "environment": {name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_HOME",
                "CUDA_PATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
            )},
        },
        "checks": checks,
        "assessment": assess(checks, args.expected_gpu, args.min_free_gib, platform.system()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print_report(snapshot)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            print(f"ERROR: cannot write snapshot to {args.output}: {error}", file=sys.stderr)
            return 2
        print(f"JSON snapshot: {args.output.resolve()}")
    return 1 if args.strict and not snapshot["assessment"]["prechecks_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
