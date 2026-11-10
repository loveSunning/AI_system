#!/usr/bin/env python3
"""W31 closed-loop HTTP benchmark: fixed workloads, warmup and separate repeats."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from common import (ROOT, APIError, RunWriter, add_connection_arguments,
                    add_generation_arguments, frozen_arguments, measure_request,
                    nonnegative_int, redact, positive_int, preflight, read_jsonl,
                    utc_now, validate_arguments)
from summarize import summarize_directory


def load_workloads(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Workloads file is empty")
    seen = set()
    for row in rows:
        if set(row) - {"id", "prompt", "max_tokens", "system"}:
            raise ValueError(f"Unknown workload fields: {set(row) - {'id', 'prompt', 'max_tokens', 'system'}}")
        if not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Workload IDs must be nonempty unique strings")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{row['id']}: prompt must be nonempty text")
        if type(row.get("max_tokens")) is not int or row["max_tokens"] < 1:
            raise ValueError(f"{row['id']}: max_tokens must be a positive integer")
        if "system" in row and (not isinstance(row["system"], str) or not row["system"].strip()):
            raise ValueError(f"{row['id']}: system must be nonempty text")
        seen.add(row["id"])
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_arguments(parser)
    add_generation_arguments(parser)
    parser.add_argument("--workloads", type=Path, default=ROOT / "configs/workloads_w31.jsonl")
    parser.add_argument("--concurrency", type=positive_int, nargs="+", default=[1, 4])
    parser.add_argument("--requests-per-workload", type=positive_int, default=20)
    parser.add_argument("--warmup", type=nonnegative_int, default=2,
                        help="sequential warmup requests before every measured case")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/w31")
    parser.add_argument("--server-manifest", type=Path,
                        help="optional saved server manifest; copied as provenance, not proof of server identity")
    parser.add_argument("--dry-run", action="store_true", help="save the plan without contacting a server")
    args = validate_arguments(parser, parser.parse_args(argv))
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("--concurrency values must be unique")
    if any(value > args.requests_per_workload for value in args.concurrency):
        parser.error("requests-per-workload must be >= every concurrency value")
    return args


def run_case(args, workload, model, case_id, concurrency, repeat_index, writer):
    for index in range(args.warmup):
        record = measure_request(args, workload, model, case_id, index + 1)
        writer.append("warmup.jsonl", record)
        if not record["ok"]:
            raise APIError(f"Warmup failed for {case_id}: {record['error']}")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(measure_request, args, workload, model, case_id, index + 1)
            for index in range(args.requests_per_workload)
        ]
        failures = 0
        for future in as_completed(futures):
            record = future.result()
            failures += not record["ok"]
            writer.append("metrics.jsonl", record)
    case = {
        "case_id": case_id, "workload_id": workload["id"],
        "concurrency": concurrency, "repeat_index": repeat_index, "mode": "stream",
        "total_requests": args.requests_per_workload,
        "wall_seconds": time.perf_counter() - started, "warmup_requests": args.warmup,
    }
    writer.append("cases.jsonl", case)
    print(f"{case_id}: {args.requests_per_workload - failures}/{args.requests_per_workload} succeeded; "
          f"wall={case['wall_seconds']:.3f}s", flush=True)
    return failures


def main(argv=None):
    args = parse_args(argv)
    secret = os.environ.get(args.api_key_env, "")
    writer = None
    manifest = {
        "schema_version": 1, "kind": "benchmark", "status": "starting",
        "started_at_utc": utc_now(), "arguments": frozen_arguments(args),
        "client_python": sys.version, "client_platform": platform.platform(),
        "load_model": "closed_loop_fixed_concurrency; per-request latency starts in worker",
        "measurement": "client text-event TTFT/TPOT approximations; warmup excluded",
        "cache_state": "not reset by client; record server configuration; W31 expects prefix cache disabled",
    }
    try:
        workloads = load_workloads(args.workloads)
        manifest.update(
            workloads=workloads,
            workloads_sha256=hashlib.sha256(args.workloads.read_bytes()).hexdigest(),
            planned_cases=len(workloads) * len(args.concurrency) * args.repeats)
        if args.server_manifest is not None:
            manifest["server_manifest_source"] = json.loads(
                args.server_manifest.read_text(encoding="utf-8-sig"))
        writer = RunWriter(args.output_dir, secret)
        writer.save("manifest.json", manifest)
        print(f"Run directory: {writer.directory}", flush=True)
        if args.dry_run:
            manifest.update(status="dry_run", execution_started=False)
            print(f"Plan: {manifest['planned_cases']} cases; no server requests sent.")
            return 0
        model, server = preflight(args)
        manifest.update(model=model, server=server, execution_started=True, status="running")
        writer.save("manifest.json", manifest)
        failures = 0
        for workload_index, workload in enumerate(workloads, 1):
            for concurrency in args.concurrency:
                for repeat_index in range(1, args.repeats + 1):
                    case_id = f"w{workload_index}_c{concurrency}_r{repeat_index}"
                    failures += run_case(args, workload, model, case_id, concurrency, repeat_index, writer)
        manifest.update(status="completed" if failures == 0 else "failed",
                        failed_requests=failures)
        writer.save("manifest.json", manifest)
        summarize_directory(writer.directory)
        print(f"Saved summary.json and summary.csv in {writer.directory}", flush=True)
        return 0 if failures == 0 else 1
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error="Interrupted")
        return 130
    except Exception as error:
        manifest.update(status="failed", error=redact(str(error), secret))
        print(manifest["error"], file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            manifest["finished_at_utc"] = utc_now()
            writer.save("manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())

