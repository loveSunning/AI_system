#!/usr/bin/env python3
"""Check and concurrently exercise a vLLM HTTP API; persist logs and measured results."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "log"


class APIError(RuntimeError):
    pass


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def normalize_base_url(value):
    parts = urlsplit(value.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("--base-url must be an http(s) server URL")
    if parts.hostname in ("0.0.0.0", "::"):
        raise ValueError("0.0.0.0/:: is a bind address; use the server IP or hostname as --base-url")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("--base-url must not contain credentials, a query or a fragment")
    port = parts.port
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Invalid server port")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check /health, /v1/models and chat generation on a vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:7081",
                        help="server root URL; an optional trailing /v1 is accepted")
    parser.add_argument("--model", help="served model alias; defaults to the first /v1/models entry")
    parser.add_argument("--prompt", default="请用三句话解释 KV cache。")
    parser.add_argument("--system", help="optional system instruction")
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=finite_float, default=0.7)
    parser.add_argument("--top-p", type=finite_float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20, help="0 disables top-k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=finite_float, default=180,
                        help="socket connect/read timeout in seconds, not a total generation deadline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="only check health and model listing")
    mode.add_argument("--stream", action="store_true", help="also run a second generation phase using SSE streaming")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY",
                        help="read optional Bearer token from this environment variable")
    parser.add_argument("--no-proxy", action="store_true", help="ignore environment HTTP proxy settings")
    mode.add_argument("--stream-only", action="store_true", help="send only streaming generation requests")
    parser.add_argument("--requests", type=int, help="requests per generation phase; defaults to concurrency")
    parser.add_argument("--concurrency", type=int, default=1, help="maximum simultaneous HTTP requests")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="always save .log, per-request .jsonl and final .json here")
    parser.add_argument("--require-server-metrics", action="store_true",
                        help="fail generation checks if the run_model profiling extension is absent")
    parser.add_argument("--output", type=Path, help="also export the final JSON to this path")
    args = parser.parse_args(argv)
    try:
        args.base_url = normalize_base_url(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    if args.max_tokens <= 0 or args.timeout <= 0 or args.temperature < 0:
        parser.error("require max-tokens > 0, timeout > 0 and temperature >= 0")
    if not 0 < args.top_p <= 1 or args.top_k < 0 or not 0 <= args.seed < 2 ** 32:
        parser.error("require 0 < top-p <= 1, top-k >= 0 and 0 <= seed < 2**32")
    if args.concurrency <= 0 or (args.requests is not None and args.requests <= 0):
        parser.error("--concurrency and --requests must be positive")
    if args.requests is None:
        args.requests = args.concurrency
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    return args


def iter_sse_data(lines):
    """Join SSE data fields; ignore comments, event names and empty heartbeats."""
    fields = []
    for raw in lines:
        line = raw.decode("utf-8-sig").rstrip("\r\n")
        if line == "":
            if fields:
                yield "\n".join(fields)
                fields = []
        elif not line.startswith(":"):
            name, separator, value = line.partition(":")
            if separator and name == "data":
                fields.append(value[1:] if value.startswith(" ") else value)
    if fields:
        yield "\n".join(fields)


def checked_object(data):
    if not isinstance(data, dict):
        raise APIError("Expected a JSON object")
    if data.get("error"):
        raise APIError("Server returned an error: " + json.dumps(data["error"], ensure_ascii=False))
    return data


def message_text(message):
    if not isinstance(message, dict):
        raise APIError("Missing message/delta object")
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise APIError("Expected text content/reasoning_content")
    return content, reasoning


class Client:
    def __init__(self, args):
        self.base_url = args.base_url
        self.timeout = args.timeout
        self.api_key = os.environ.get(args.api_key_env, "")
        self.on_first_text = getattr(args, "on_first_text", None)
        self.opener = build_opener(ProxyHandler({})) if args.no_proxy else build_opener()

    def open(self, path, payload=None, stream=False):
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers,
                          method="POST" if payload is not None else "GET")
        try:
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise APIError(f"{path}: HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, socket.timeout) as error:
            raise APIError(f"{path}: connection/read failed: {error}") from error

    def health(self):
        with self.open("/health") as response:
            response.read()
            if response.status != 200:
                raise APIError(f"/health: unexpected HTTP {response.status}")
            return {"http_status": response.status}

    def json_call(self, path, payload=None):
        with self.open(path, payload) as response:
            try:
                data = json.load(response)
            except (ValueError, UnicodeError) as error:
                raise APIError(f"{path}: response is not valid JSON") from error
            return checked_object(data)

    def models(self):
        data = self.json_call("/v1/models")
        models = data.get("data")
        if not isinstance(models, list) or not models:
            raise APIError("/v1/models returned no models")
        ids = [item.get("id") for item in models if isinstance(item, dict)]
        if not ids or any(not isinstance(model, str) or not model for model in ids):
            raise APIError("/v1/models returned invalid model IDs")
        return {"model_ids": ids}

    def chat(self, payload):
        started = time.perf_counter()
        data = self.json_call("/v1/chat/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise APIError("Chat response has no valid choices")
        content, reasoning = message_text(choices[0].get("message"))
        if not content and not reasoning:
            raise APIError("Chat response contains no text")
        elapsed = time.perf_counter() - started
        return {"text": content, "reasoning_content": reasoning,
                "finish_reason": choices[0].get("finish_reason"), "usage": data.get("usage"),
                "server_metrics": data.get("profiling"),
                "client_metrics": client_metrics(data.get("usage"), elapsed)}

    def stream_chat(self, payload, path="/v1/chat/completions"):
        streaming = dict(payload, stream=True, stream_options={"include_usage": True})
        started = time.perf_counter()
        first_text = None
        last_text = None
        server_metrics = None
        content_parts, reasoning_parts = [], []
        usage, finish_reason = None, None
        done, seen_choice = False, False
        text_chunks = 0
        with self.open(path, streaming, stream=True) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                raise APIError("Expected Content-Type: text/event-stream")
            for event in iter_sse_data(response):
                if event.strip() == "[DONE]":
                    done = True
                    break
                try:
                    data = checked_object(json.loads(event))
                except ValueError as error:
                    raise APIError("SSE event is not valid JSON") from error
                if data.get("profiling") is not None:
                    server_metrics = data["profiling"]
                if data.get("usage") is not None:
                    usage = data["usage"]
                choices = data.get("choices", [])
                if not isinstance(choices, list):
                    raise APIError("SSE choices must be a list")
                for choice in choices:
                    if not isinstance(choice, dict):
                        raise APIError("Invalid SSE choice")
                    if choice.get("index", 0) != 0:
                        continue
                    seen_choice = True
                    content, reasoning = (message_text(choice.get("delta", {}))
                        if path.endswith("chat/completions") else (choice.get("text") or "", ""))
                    if content or reasoning:
                        text_chunks += 1
                        content_parts.append(content)
                        reasoning_parts.append(reasoning)
                        last_text = time.perf_counter()
                        if first_text is None:
                            first_text = last_text
                            if self.on_first_text is not None:
                                self.on_first_text(first_text - started)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        elapsed = time.perf_counter() - started
        if not done or not seen_choice or finish_reason is None:
            raise APIError("Incomplete SSE stream: missing [DONE], choices or finish_reason")
        if not content_parts and not reasoning_parts:
            raise APIError("SSE response contains no text")
        return {"text": "".join(content_parts), "reasoning_content": "".join(reasoning_parts),
                "finish_reason": finish_reason, "usage": usage, "text_chunks": text_chunks,
                "server_metrics": server_metrics,
                "client_metrics": client_metrics(usage, elapsed,
                    first_text - started if first_text is not None else None,
                    last_text - started if last_text is not None else None)}


def request_payload(args, model):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    greedy = args.temperature == 0
    payload = {
        "model": model, "messages": messages, "stream": False,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "top_p": 1.0 if greedy else args.top_p,
        "top_k": -1 if greedy or args.top_k == 0 else args.top_k,
        "repetition_penalty": 1.0, "seed": args.seed,
    }
    if args.thinking != "auto":
        payload["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    return payload


def run_check(report, name, operation, secret, log=None):
    entry = {"name": name, "ok": False}
    report["checks"].append(entry)
    started = time.perf_counter()
    try:
        result = operation()
        entry.update(ok=True, result=result)
    except Exception as error:
        detail = str(error).replace(secret, "<redacted>") if secret else str(error)
        entry["error"] = detail
        raise APIError(detail) from error
    finally:
        entry["elapsed_seconds"] = time.perf_counter() - started
        if log:
            log.record("check", entry)
    return result


def usage_tokens(usage, name):
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def client_metrics(usage, elapsed, first_text=None, last_text=None):
    tokens = usage_tokens(usage, "completion_tokens")
    return {
        "end_to_end_seconds": elapsed,
        "output_tokens_per_second": tokens / elapsed if tokens is not None and elapsed > 0 else None,
        "time_to_first_text_seconds": first_text,
        "time_to_last_text_seconds": last_text,
        "tpot_text_event_seconds": ((last_text - first_text) / (tokens - 1)
            if tokens is not None and tokens > 1 and first_text is not None and last_text is not None else None),
        "tpot_note": "text-event approximation; SSE chunks can contain multiple or buffered tokens",
        "timing_scope": "client_http_wall_time_including_network_and_server",
    }


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "<redacted>") if secret else value
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


class TestLog:
    """Only the coordinator writes files; worker completions are recorded as they arrive."""
    def __init__(self, directory, secret):
        directory = directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        prefix = directory / ("test_server_" + self.run_id)
        self.paths = {kind: str(prefix.with_suffix(suffix)) for kind, suffix in (
            ("console", ".log"), ("requests", ".jsonl"), ("report", ".json")
        )}
        self.secret = secret
        self.logger = logging.Logger("test_server." + self.run_id, level=logging.INFO)
        self.events = None
        try:
            for handler in (logging.FileHandler(self.paths["console"], encoding="utf-8"),
                            logging.StreamHandler(sys.stdout)):
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                self.logger.addHandler(handler)
            self.events = open(self.paths["requests"], "x", encoding="utf-8")
        except BaseException:
            self.close()
            raise

    def info(self, message):
        self.logger.info(redact(message, self.secret))

    def record(self, event, value):
        record = redact({"run_id": self.run_id,
                         "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                         "event": event, "data": value}, self.secret)
        self.events.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.events.flush()
        if event in ("request", "check"):
            status = "PASS" if value.get("ok") else "FAIL"
            self.info(f"[{status}] {value['name']} ({value['elapsed_seconds']:.3f}s)"
                      + (f" error={value['error']}" if value.get("error") else ""))
            result = value.get("result", {})
            if "client_metrics" in result:
                self.info("Metrics: " + json.dumps({
                    "usage": result.get("usage"), "server": result.get("server_metrics"),
                    "client": result["client_metrics"]}, ensure_ascii=False))
                self.info("Response: " + result.get("text", ""))
                if result.get("reasoning_content"):
                    self.info("Reasoning: " + result["reasoning_content"])

    def save_report(self, report, extra_output=None):
        serialized = json.dumps(redact(report, self.secret), ensure_ascii=False, indent=2) + "\n"
        Path(self.paths["report"]).write_text(serialized, encoding="utf-8")
        if extra_output:
            extra_output.parent.mkdir(parents=True, exist_ok=True)
            extra_output.write_text(serialized, encoding="utf-8")

    def close(self):
        if self.events is not None:
            self.events.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def percentile(values, fraction):
    """Nearest-rank percentile; retain raw per-request latencies for recomputation."""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def summarize_requests(entries, elapsed, concurrency):
    successful = [entry for entry in entries if entry["ok"]]
    counts = [usage_tokens(entry["result"].get("usage"), "completion_tokens") for entry in successful]
    usage_complete = bool(successful) and all(count is not None for count in counts)
    total_tokens = sum(counts) if usage_complete else None
    latencies = [entry["result"]["client_metrics"]["end_to_end_seconds"] for entry in successful]
    return {
        "total_requests": len(entries), "successful_requests": len(successful),
        "failed_requests": len(entries) - len(successful),
        "error_rate": (len(entries) - len(successful)) / len(entries) if entries else None,
        "configured_concurrency": concurrency, "worker_count": min(concurrency, len(entries)),
        "wall_seconds": elapsed, "output_tokens": total_tokens, "usage_complete": usage_complete,
        "output_tokens_per_second": total_tokens / elapsed if total_tokens is not None and elapsed > 0 else None,
        "requests_per_second": len(successful) / elapsed if elapsed > 0 else None,
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "percentile_method": "nearest_rank_successful_requests_only",
        "timing_scope": "client_phase_wall_time_all_requests_including_failures",
    }


def execute_request(args, payload, streaming, index):
    name = "POST /v1/chat/completions" + (" (SSE)" if streaming else "")
    entry = {"name": name, "request_index": index, "mode": "stream" if streaming else "non_stream",
             "ok": False}
    started = time.perf_counter()
    secret = os.environ.get(args.api_key_env, "")
    try:
        # Each worker owns its urllib opener; no mutable HTTP client is shared.
        client = Client(args)
        result = client.stream_chat(payload) if streaming else client.chat(payload)
        entry["result"] = result
        metrics = result.get("server_metrics")
        if args.require_server_metrics and (not isinstance(metrics, dict)
                or not isinstance(metrics.get("inference_seconds"), (int, float))
                or metrics.get("output_tokens_per_second") is None):
            raise APIError("Missing server profiling metrics; start the server with updated run_model.py")
        entry["ok"] = True
    except Exception as error:
        entry["error"] = redact(str(error), secret)
    finally:
        entry["elapsed_seconds"] = time.perf_counter() - started
    return entry


def run_phase(args, payload, streaming, report, log):
    entries = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, args.requests)) as executor:
        futures = [executor.submit(execute_request, args, payload, streaming, index)
                   for index in range(1, args.requests + 1)]
        for future in as_completed(futures):
            entry = future.result()
            entries.append(entry)
            report["checks"].append(entry)
            log.record("request", entry)
    elapsed = time.perf_counter() - started
    summary = summarize_requests(entries, elapsed, args.concurrency)
    summary["mode"] = "stream" if streaming else "non_stream"
    report["summaries"].append(summary)
    log.record("summary", summary)
    log.info("Phase summary: " + json.dumps(summary, ensure_ascii=False))
    return entries


def main(argv=None):
    args = parse_args(argv)
    report = {"schema_version": 2, "base_url": args.base_url, "ok": False,
              "checks": [], "summaries": [], "concurrency": args.concurrency,
              "requests_per_phase": args.requests, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    secret = os.environ.get(args.api_key_env, "")
    try:
        log = TestLog(args.log_dir, secret)
    except OSError as error:
        print("[FAIL] Cannot create test logs:", redact(str(error), secret), file=sys.stderr)
        return 1
    report.update(run_id=log.run_id, log_files=log.paths)
    return_code = 1
    try:
        log.info("Server: " + args.base_url)
        log.info("Logs: " + json.dumps(log.paths, ensure_ascii=False))
        log.record("run_started", {key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()})
        client = Client(args)
        run_check(report, "GET /health", client.health, secret, log)
        models = run_check(report, "GET /v1/models", client.models, secret, log)["model_ids"]
        model = args.model or models[0]
        if model not in models:
            raise APIError(f"Model {model!r} is not served. Available: {models}")
        report["model"] = model
        log.info("Selected model: " + model)
        if not args.check_only:
            payload = request_payload(args, model)
            report["request"] = payload
            entries = []
            if not args.stream_only:
                entries.extend(run_phase(args, payload, False, report, log))
            if args.stream or args.stream_only:
                entries.extend(run_phase(args, payload, True, report, log))
            failures = [entry for entry in entries if not entry["ok"]]
            if failures:
                raise APIError(f"{len(failures)} request(s) failed; first error: {failures[0]['error']}")
            if any(not entry["result"].get("server_metrics") for entry in entries):
                log.info("Server profiling is absent for some requests; client timings remain available.")
        report["ok"] = True
        return_code = 0
        log.info("All requested API checks passed. These are exploratory measurements; warmup is not automatic.")
    except KeyboardInterrupt:
        report["error"] = "Interrupted"
        return_code = 130
        log.info("[FAIL] Interrupted")
    except Exception as error:
        report["error"] = redact(str(error), secret)
        log.info("[FAIL] " + report["error"])
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            log.record("run_finished", {"ok": report["ok"], "error": report.get("error")})
            log.save_report(report, args.output)
            log.info("Saved report: " + log.paths["report"])
        except OSError as error:
            print("[FAIL] Cannot save report:", redact(str(error), secret), file=sys.stderr)
            return_code = 1
        finally:
            log.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
