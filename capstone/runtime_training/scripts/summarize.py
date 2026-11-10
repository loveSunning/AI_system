#!/usr/bin/env python3
"""Recompute case-level W31 summaries from saved request records and wall times."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from common import read_jsonl, summarize_case, write_json


def summarize_directory(directory):
    directory = Path(directory).expanduser().resolve()
    records = read_jsonl(directory / "metrics.jsonl")
    cases = read_jsonl(directory / "cases.jsonl")
    if not cases:
        raise ValueError("No completed measurement cases; dry runs/preflight failures have no performance summary")
    case_ids = {case["case_id"] for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("Duplicate case IDs")
    if any(record.get("case_id") not in case_ids for record in records):
        raise ValueError("Requests belong to an unfinished/unknown case; run is incomplete")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
    if manifest.get("kind") == "benchmark" and manifest.get("planned_cases") != len(cases):
        raise ValueError("Incomplete run: measured case count differs from frozen plan")
    summaries = [summarize_case(records, case) for case in cases]
    result = {"schema_version": 1, "run_id": directory.name,
              "run_status": manifest.get("status", "unknown"), "cases": summaries,
              "note": "Cases/repeats remain separate; throughput is never summed across repetitions"}
    write_json(directory / "summary.json", result)
    rows = []
    for summary in summaries:
        row = {key: summary[key] for key in (
            "case_id", "workload_id", "concurrency", "repeat_index", "total_requests",
            "successful_requests", "failed_requests", "error_rate", "wall_seconds",
            "output_tokens", "output_tokens_per_second", "requests_per_second")}
        for metric in ("latency_seconds", "ttft_text_seconds", "tpot_text_seconds"):
            row.update({f"{metric}_{stat}": value for stat, value in summary[metric].items()})
        row["finish_reasons"] = json.dumps(summary["finish_reasons"], ensure_ascii=False)
        rows.append(row)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize_directory(args.run_dir)
        print(f"Summarized {len(result['cases'])} cases in {args.run_dir.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot summarize: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

