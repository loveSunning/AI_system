#!/usr/bin/env python3
"""Evaluate the frozen 100-task holdout; infrastructure failures never become scores."""
import argparse
import json
from pathlib import Path
from experiment_utils import ROOT, digest, read, rows, write
from test_server import Client, parse_args as client_args, redact
import os


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def same_types(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_types(actual[k], v) for k, v in expected.items())
    return actual == expected


def score(task, text, finish_reason):
    if finish_reason != "stop":
        return False
    if task["scorer"] == "exact":
        return text.strip() == task["expected"]
    if task["scorer"] != "json_exact":
        raise ValueError("Unknown scorer")
    try:
        return same_types(json.loads(text, object_pairs_hook=strict_object), task["expected"])
    except (ValueError, TypeError):
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=ROOT / "data/frozen")
    p.add_argument("--base-url", default="http://127.0.0.1:7081")
    p.add_argument("--model", default="local-model")
    p.add_argument("--label", required=True, help="configuration label; does not prove server precision")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    from experiment_utils import load_snapshot
    manifest, _, _ = load_snapshot(a.snapshot)
    if manifest["scorer_sha256"] != digest(Path(__file__)):
        p.error("Scorer changed after freeze; create a new snapshot")
    protocol = read(a.snapshot / "quality_protocol.json")
    generation = dict(protocol["generation"])
    thinking = generation.pop("thinking")
    tasks = rows(a.snapshot / "quality_holdout.jsonl")
    if len(tasks) != 100 or len({t["id"] for t in tasks}) != 100:
        p.error("Require exactly 100 unique task IDs")
    if a.output.exists():
        p.error("Use a new output directory; previous results are immutable")
    a.output.mkdir(parents=True)
    args = client_args(["--base-url", a.base_url, "--no-proxy"])
    client = Client(args)
    results = []
    for task in tasks:
        entry = {"id": task["id"], "category": task["category"], "expected": task["expected"], "ok": False}
        try:
            result = client.chat({"model": a.model, "messages": [{"role": "user", "content": task["prompt"]}],
                **generation, "chat_template_kwargs": {"enable_thinking": thinking}})
            entry.update(ok=True, result=result, correct=score(task, result["text"], result["finish_reason"]))
        except Exception as e:
            entry["error"] = redact(str(e), os.environ.get("VLLM_API_KEY", ""))
        results.append(entry)
        with (a.output / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    complete = all(e["ok"] for e in results)
    write(a.output / "summary.json", {"label": a.label, "precision_verified": False,
        "precision_note": "associate with saved server startup log; label alone is not evidence",
        "manifest_sha256": digest(a.snapshot / "manifest.json"), "complete": complete,
        "infrastructure_failures": sum(not e["ok"] for e in results),
        "accuracy": sum(e.get("correct", False) for e in results) / 100 if complete else None,
        "categories": {category: {"correct": sum(e.get("correct", False) for e in results if e["category"] == category),
            "total": 25} for category in sorted({t["category"] for t in tasks})}})
    return 0 if complete else 1

if __name__ == "__main__":
    raise SystemExit(main())
