#!/usr/bin/env python3
"""Smoke-test a local/remote vLLM HTTP server using only the standard library."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


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
    mode.add_argument("--stream", action="store_true", help="also send a second request to test SSE streaming")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY",
                        help="read optional Bearer token from this environment variable")
    parser.add_argument("--no-proxy", action="store_true", help="ignore environment HTTP proxy settings")
    parser.add_argument("--output", type=Path, help="write JSON results, including partial failures")
    args = parser.parse_args(argv)
    try:
        args.base_url = normalize_base_url(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    if args.max_tokens <= 0 or args.timeout <= 0 or args.temperature < 0:
        parser.error("require max-tokens > 0, timeout > 0 and temperature >= 0")
    if not 0 < args.top_p <= 1 or args.top_k < 0 or not 0 <= args.seed < 2 ** 32:
        parser.error("require 0 < top-p <= 1, top-k >= 0 and 0 <= seed < 2**32")
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
        data = self.json_call("/v1/chat/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise APIError("Chat response has no valid choices")
        content, reasoning = message_text(choices[0].get("message"))
        if not content and not reasoning:
            raise APIError("Chat response contains no text")
        print("\nResponse:", content)
        if reasoning:
            print("Reasoning:", reasoning)
        return {"text": content, "reasoning_content": reasoning,
                "finish_reason": choices[0].get("finish_reason"), "usage": data.get("usage")}

    def stream_chat(self, payload):
        streaming = dict(payload, stream=True, stream_options={"include_usage": True})
        content_parts, reasoning_parts = [], []
        usage, finish_reason = None, None
        done, seen_choice = False, False
        text_chunks = 0
        print("\nStreaming response:", flush=True)
        with self.open("/v1/chat/completions", streaming, stream=True) as response:
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
                    content, reasoning = message_text(choice.get("delta", {}))
                    if content or reasoning:
                        text_chunks += 1
                        content_parts.append(content)
                        reasoning_parts.append(reasoning)
                        print(content or reasoning, end="", flush=True)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        print()
        if not done or not seen_choice or finish_reason is None:
            raise APIError("Incomplete SSE stream: missing [DONE], choices or finish_reason")
        if not content_parts and not reasoning_parts:
            raise APIError("SSE response contains no text")
        return {"text": "".join(content_parts), "reasoning_content": "".join(reasoning_parts),
                "finish_reason": finish_reason, "usage": usage, "text_chunks": text_chunks}


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


def run_check(report, name, operation, secret):
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
    print(f"[PASS] {name} ({entry['elapsed_seconds']:.3f}s)", flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    report = {"base_url": args.base_url, "ok": False, "checks": []}
    client = Client(args)
    try:
        print("Server:", args.base_url, flush=True)
        run_check(report, "GET /health", client.health, client.api_key)
        models = run_check(report, "GET /v1/models", client.models, client.api_key)["model_ids"]
        model = args.model or models[0]
        if model not in models:
            raise APIError(f"Model {model!r} is not served. Available: {models}")
        report["model"] = model
        print("Selected model:", model, flush=True)
        if not args.check_only:
            payload = request_payload(args, model)
            report["request"] = payload
            run_check(report, "POST /v1/chat/completions", lambda: client.chat(payload), client.api_key)
            if args.stream:
                run_check(report, "POST /v1/chat/completions (SSE)",
                          lambda: client.stream_chat(payload), client.api_key)
        report["ok"] = True
    except KeyboardInterrupt:
        report["error"] = "Interrupted"
        return_code = 130
    except Exception as error:
        detail = str(error).replace(client.api_key, "<redacted>") if client.api_key else str(error)
        report["error"] = detail
        print("[FAIL]", detail, file=sys.stderr)
        return_code = 1
    else:
        print("All requested API checks passed. Elapsed times are smoke-test observations.")
        return_code = 0
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Saved:", args.output)
        except OSError as error:
            print("[FAIL] Cannot save report:", error, file=sys.stderr)
            return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
