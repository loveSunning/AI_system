#!/usr/bin/env python3
"""Check key data coverage in an exported nsys SQLite; no GPU is required."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3


KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
GRAPH_TABLE = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
API_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")


def inspect_capture(path, require_worker_ranges=False, expect_graph=False,
                    measure_range="controller/measure", require_load_model=True,
                    fail_on_incomplete_cuda=False):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def count(table, predicate="1", parameters=()):
            if table not in tables:
                return 0
            return db.execute(f'SELECT count(*) FROM "{table}" WHERE {predicate}', parameters).fetchone()[0]

        def columns(table):
            return {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}

        names, measure = {}, []
        if {"NVTX_EVENTS", "StringIds"} <= tables:
            rows = db.execute("""SELECT coalesce(n.text, s.value), n.start, n.end
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId""")
            for name, start, end in rows:
                if name:
                    names[name] = names.get(name, 0) + 1
                    if name == measure_range and end is not None and end > start:
                        measure.append((start, end))

        def in_measure(table, predicate="1"):
            # Coverage, not a duration calculation. Adjacent benchmark rounds
            # normally have disjoint windows; overlapping records are not clipped.
            return sum(count(table, f"({predicate}) AND start<? AND end>?", (end, start))
                       for start, end in measure)

        graph_node_predicate = "graphNodeId IS NOT NULL AND graphNodeId!=0"
        has_graph_nodes = KERNEL_TABLE in tables and "graphNodeId" in columns(KERNEL_TABLE)
        launch_predicate = "nameId IN (SELECT id FROM StringIds WHERE value LIKE '%GraphLaunch%')"
        launches = measure_launches = 0
        if "StringIds" in tables:
            for table in API_TABLES:
                launches += count(table, launch_predicate)
                measure_launches += in_measure(table, launch_predicate)

        result = {
            "measure_range": measure_range,
            "kernel_records": count(KERNEL_TABLE),
            "gpu_graph_records": count(GRAPH_TABLE),
            "graph_resource_records": count("CUDA_GRAPH_EVENTS"),
            "api_records": sum(count(table) for table in API_TABLES),
            "worker_ranges": {name: total for name, total in names.items() if name.startswith("worker/")},
            "measure_gpu_records": sum(in_measure(table) for table in (KERNEL_TABLE, GRAPH_TABLE)),
            "graph_node_kernel_records": count(KERNEL_TABLE, graph_node_predicate) if has_graph_nodes else 0,
            "graph_launch_api_records": launches,
            "measure_graph_records": in_measure(GRAPH_TABLE),
            "measure_graph_node_kernel_records": in_measure(KERNEL_TABLE, graph_node_predicate) if has_graph_nodes else 0,
            "measure_graph_launch_api_records": measure_launches,
            "incomplete_cuda_warnings": count("DIAGNOSTIC_EVENT", "text LIKE 'Not all CUDA events%'"),
        }
        errors = []
        if not result["kernel_records"] and not result["gpu_graph_records"]:
            errors.append("No GPU execution data; Graph Creation markers are not execution timing.")
        if require_worker_ranges:
            required = ["worker/prefill", "worker/decode"]
            if require_load_model:
                required.append("worker/load_model")
            for name in required:
                if not names.get(name):
                    errors.append("Missing NVTX range: " + name)
            if not measure:
                errors.append(f"Missing completed {measure_range} range.")
            elif not result["measure_gpu_records"]:
                errors.append(f"No GPU execution data inside {measure_range}.")
        if expect_graph:
            # With new annotations, require evidence in the measured phase;
            # graph replay during internal warmup alone is insufficient.
            scoped = "measure_" if measure else ""
            graph_launches = result[scoped + "graph_launch_api_records"]
            gpu_graphs = result["measure_graph_records" if measure else "gpu_graph_records"]
            graph_nodes = result[scoped + "graph_node_kernel_records"]
            if not graph_launches or not (gpu_graphs or graph_nodes):
                errors.append("CUDA Graph replay was not demonstrated; check capture completeness and actual runtime configuration.")
        if fail_on_incomplete_cuda and result["incomplete_cuda_warnings"]:
            errors.append("CUDA event-loss diagnostics present; capture is incomplete.")
        result["errors"] = errors
        result["key_data_present"] = not errors
        result["note"] = "Coverage check only; inspect worker diagnostics for event loss. CPU NVTX scopes are not GPU durations."
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--require-worker-ranges", action="store_true")
    parser.add_argument("--expect-graph", action="store_true")
    parser.add_argument("--measure-range", default="controller/measure")
    parser.add_argument("--skip-load-model-range", action="store_true",
                        help="steady capture intentionally excludes model loading")
    parser.add_argument("--fail-on-incomplete-cuda", action="store_true")
    args = parser.parse_args(argv)
    result = inspect_capture(args.sqlite, args.require_worker_ranges, args.expect_graph,
                             measure_range=args.measure_range,
                             require_load_model=not args.skip_load_model_range,
                             fail_on_incomplete_cuda=args.fail_on_incomplete_cuda)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["key_data_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
