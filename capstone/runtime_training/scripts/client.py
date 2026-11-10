#!/usr/bin/env python3
"""Send one W31 chat request, optionally printing SSE deltas immediately."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from common import (ROOT, RunWriter, add_connection_arguments, add_generation_arguments,
                    frozen_arguments, measure_request, redact, preflight, positive_int,
                    summarize_case, utc_now, validate_arguments)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_arguments(parser)
    add_generation_arguments(parser)
    parser.add_argument("--prompt", default="请用三句话解释大模型推理中的 prefill 和 decode。")
    parser.add_argument("--system")
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--stream", action="store_true", help="print generated SSE text as it arrives")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/client")
    args = validate_arguments(parser, parser.parse_args(argv))
    if not args.prompt.strip() or (args.system is not None and not args.system.strip()):
        parser.error("prompt/system must not be empty")
    return args


def main(argv=None):
    args = parse_args(argv)
    secret = os.environ.get(args.api_key_env, "")
    writer = None
    manifest = {"schema_version": 1, "kind": "client", "status": "starting",
                "started_at_utc": utc_now(), "arguments": frozen_arguments(args)}
    try:
        writer = RunWriter(args.output_dir, secret)
        writer.save("manifest.json", manifest)
        print(f"Run directory: {writer.directory}", flush=True)
        model, server = preflight(args)
        manifest.update(model=model, server=server)
        workload = {"id": "single", "prompt": args.prompt, "max_tokens": args.max_tokens}
        if args.system is not None:
            workload["system"] = args.system

        def display_delta(content, reasoning):
            text = reasoning + content
            print(redact(text, secret), end="", flush=True)

        started = time.perf_counter()
        record = measure_request(
            args, workload, model, "single", 1, streaming=args.stream,
            on_delta=display_delta if args.stream else None)
        wall = time.perf_counter() - started
        if args.stream:
            print(flush=True)
        elif record["ok"]:
            print(redact(record.get("reasoning_content", "") + record["text"], secret))
        case = {"case_id": "single", "workload_id": "single", "concurrency": 1,
                "repeat_index": 1, "mode": "stream" if args.stream else "non_stream",
                "total_requests": 1, "wall_seconds": wall, "warmup_requests": 0}
        writer.append("metrics.jsonl", record)
        writer.append("cases.jsonl", case)
        summary = summarize_case([record], case)
        writer.save("summary.json", {"schema_version": 1, "cases": [summary]})
        manifest.update(status="completed" if record["ok"] else "failed")
        print(f"finish_reason={record.get('finish_reason')}; "
              f"input_tokens={record.get('input_tokens')}; output_tokens={record.get('output_tokens')}")
        print(f"TTFT(text)={record['ttft_text_seconds']}; "
              f"TPOT(text approximation)={record['tpot_text_seconds']}")
        if not record["ok"]:
            print(redact(record["error"], secret), file=sys.stderr)
        return 0 if record["ok"] else 1
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

