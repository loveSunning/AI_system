"""Shared deterministic artifact and protocol helpers (standard library only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")

def load_snapshot(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    for name, sha in manifest["files"].items():
        if digest(directory / name) != sha:
            raise ValueError("Snapshot hash mismatch: " + name)
    return manifest, rows(directory / "requests.jsonl"), read(directory / "workloads.yaml")

def failure_kind(message):
    s = message.lower()
    if "out of memory" in s or "cuda oom" in s:
        return "oom"
    if "kv cache" in s and ("memory" in s or "capacity" in s):
        return "kv_capacity"
    if "timeout" in s or "timed out" in s:
        return "timeout"
    if "http 4" in s:
        return "rejected"
    if "profiling" in s:
        return "missing_metrics"
    return "error"
