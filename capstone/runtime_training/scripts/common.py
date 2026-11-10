"""W31 stream handling, request records and per-case metrics.

HTTP/SSE primitives are implemented in this project's http_client module.
Text event times are client observations, never GPU ITL.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
from http_client import (
    APIError, HTTPClient, checked_object, client_metrics, finite_float,
    iter_sse_data, message_text, normalize_base_url, percentile, redact,
    request_payload, usage_tokens,
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def add_connection_arguments(parser):
    parser.add_argument("--base-url", default="http://127.0.0.1:7081")
    parser.add_argument("--model", help="served alias; defaults to first /v1/models entry")
    parser.add_argument("--timeout", type=positive_float, default=180,
                        help="socket connect/read timeout, not a total request deadline")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--no-proxy", action="store_true")


def add_generation_arguments(parser):
    parser.add_argument("--thinking", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--temperature", type=finite_float, default=0.0)
    parser.add_argument("--top-p", type=finite_float, default=1.0)
    parser.add_argument("--top-k", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=nonnegative_int, default=42)


def validate_arguments(parser, args):
    try:
        args.base_url = normalize_base_url(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.seed >= 2 ** 32:
        parser.error("require temperature >= 0, 0 < top-p <= 1 and seed < 2**32")
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


class RunWriter:
    """Coordinator-only writer, using a new directory for every execution."""
    def __init__(self, output_dir, secret=""):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        self.directory = Path(output_dir).expanduser().resolve() / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.secret = secret
        self.requests_path = self.directory / "metrics.jsonl"
        self.cases_path = self.directory / "cases.jsonl"
        self.requests_path.touch(exist_ok=False)
        self.cases_path.touch(exist_ok=False)

    def save(self, name, value):
        write_json(self.directory / name, redact(value, self.secret))

    def append(self, name, value):
        with (self.directory / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redact(value, self.secret), ensure_ascii=False,
                                    allow_nan=False) + "\n")


class RuntimeClient(HTTPClient):
    """Read chat output and emit text callbacks before the response finishes."""
    def stream_chat(self, payload, on_delta=None):
        streaming = dict(payload, stream=True, stream_options={"include_usage": True})
        started = time.perf_counter()
        first_text = last_text = None
        usage = finish_reason = server_metrics = None
        parts, reasoning_parts = [], []
        event_count = 0
        done = seen_choice = False
        with self.open("/v1/chat/completions", streaming, stream=True) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                raise APIError("Expected Content-Type: text/event-stream")
            for event in iter_sse_data(response):
                if event.strip() == "[DONE]":
                    done = True
                    break
                try:
                    data = checked_object(json.loads(event))
                except (ValueError, UnicodeError) as error:
                    raise APIError("SSE event is not valid JSON") from error
                if data.get("usage") is not None:
                    usage = data["usage"]
                if data.get("profiling") is not None:
                    server_metrics = data["profiling"]
                choices = data.get("choices", [])
                if not isinstance(choices, list):
                    raise APIError("Invalid SSE choices")
                for choice in choices:
                    if not isinstance(choice, dict):
                        raise APIError("Invalid SSE choice")
                    if choice.get("index", 0) != 0:
                        continue
                    seen_choice = True
                    content, reasoning = message_text(choice.get("delta", {}))
                    if content or reasoning:
                        last_text = time.perf_counter()
                        if first_text is None:
                            first_text = last_text
                        event_count += 1
                        parts.append(content)
                        reasoning_parts.append(reasoning)
                        if on_delta is not None:
                            on_delta(content, reasoning)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        elapsed = time.perf_counter() - started
        if not done or not seen_choice or not isinstance(finish_reason, str) or not finish_reason:
            raise APIError("Incomplete SSE stream: missing [DONE], choices or finish_reason")
        if first_text is None:
            raise APIError("SSE response contains no generated text")
        return {
            "text": "".join(parts), "reasoning_content": "".join(reasoning_parts),
            "finish_reason": finish_reason, "usage": usage, "text_chunks": event_count,
            "server_metrics": server_metrics,
            "client_metrics": client_metrics(
                usage, elapsed, first_text - started, last_text - started),
        }


def preflight(args):
    client = RuntimeClient(args)
    client.health()
    model_ids = client.models()["model_ids"]
    model = args.model or model_ids[0]
    if model not in model_ids:
        raise APIError(f"Model {model!r} is not served; available: {model_ids}")
    version = None
    try:
        version = client.json_call("/version")
    except APIError:
        pass  # Optional endpoint; never fabricate a version.
    return model, {"model_ids": model_ids, "server_version": version}


def payload_for(args, workload, model):
    params = SimpleNamespace(
        prompt=workload["prompt"], system=workload.get("system"),
        max_tokens=workload["max_tokens"], thinking=args.thinking,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        seed=args.seed,
    )
    return request_payload(params, model)


def measure_request(args, workload, model, case_id, request_index, *, streaming=True, on_delta=None):
    started = time.perf_counter()
    record = {
        "schema_version": 1, "case_id": case_id, "workload_id": workload["id"],
        "request_index": request_index, "mode": "stream" if streaming else "non_stream",
        "started_at_utc": utc_now(), "ok": False, "model": model,
        "configured_max_tokens": workload["max_tokens"],
        "elapsed_seconds": None, "ttft_text_seconds": None, "tpot_text_seconds": None,
        "input_tokens": None, "output_tokens": None, "finish_reason": None,
    }
    try:
        client = RuntimeClient(args)
        payload = payload_for(args, workload, model)
        result = (client.stream_chat(payload, on_delta=on_delta) if streaming
                  else client.chat(payload))
        if not isinstance(result.get("finish_reason"), str) or not result["finish_reason"]:
            raise APIError("Response has no finish_reason")
        metrics = result["client_metrics"]
        record.update(
            ok=True, elapsed_seconds=metrics["end_to_end_seconds"],
            ttft_text_seconds=metrics["time_to_first_text_seconds"],
            tpot_text_seconds=metrics["tpot_text_event_seconds"],
            input_tokens=usage_tokens(result.get("usage"), "prompt_tokens"),
            output_tokens=usage_tokens(result.get("usage"), "completion_tokens"),
            finish_reason=result["finish_reason"], text=result["text"],
            reasoning_content=result.get("reasoning_content", ""),
            text_events=result.get("text_chunks"), server_metrics=result.get("server_metrics"),
        )
    except Exception as error:
        secret = os.environ.get(args.api_key_env, "")
        record["error"] = redact(str(error), secret)
        record["elapsed_seconds"] = time.perf_counter() - started
    return record


def distribution(values):
    clean = [value for value in values if isinstance(value, (int, float))
             and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
    return {
        "samples": len(clean),
        "p50": percentile(clean, 0.5),
        "p95": percentile(clean, 0.95),
    }


def summarize_case(records, case):
    selected = [record for record in records if record["case_id"] == case["case_id"]]
    expected = case["total_requests"]
    if len(selected) != expected:
        raise ValueError(f"{case['case_id']}: expected {expected} records, found {len(selected)}")
    if len({record["request_index"] for record in selected}) != len(selected):
        raise ValueError(f"{case['case_id']}: duplicate request indices")
    success = [record for record in selected if record["ok"]]
    complete_usage = bool(success) and all(
        isinstance(record.get("output_tokens"), int) and not isinstance(record["output_tokens"], bool)
        and record["output_tokens"] >= 0 for record in success)
    tokens = sum(record["output_tokens"] for record in success) if complete_usage else None
    wall = case["wall_seconds"]
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise ValueError(f"{case['case_id']}: invalid measurement wall time")
    return {
        **case,
        "successful_requests": len(success), "failed_requests": len(selected) - len(success),
        "error_rate": (len(selected) - len(success)) / len(selected) if selected else None,
        "usage_complete": complete_usage, "output_tokens": tokens,
        "output_tokens_per_second": tokens / wall if tokens is not None else None,
        "requests_per_second": len(success) / wall,
        "latency_seconds": distribution([record["elapsed_seconds"] for record in success]),
        "ttft_text_seconds": distribution([record.get("ttft_text_seconds") for record in success]),
        "tpot_text_seconds": distribution([record.get("tpot_text_seconds") for record in success]),
        "finish_reasons": dict(Counter(record.get("finish_reason") for record in success)),
        "timing_scope": "client_http_wall_time; case wall includes failed requests; excludes warmup",
        "ttft_tpot_scope": "text-event approximation; SSE event can contain multiple tokens",
        "percentile_method": "nearest_rank_successful_requests_only",
    }


def frozen_arguments(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}

