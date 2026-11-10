"""HTTP/SSE protocol and timing primitives for this project.

Only the Python standard library is needed. This module can be studied and run
without any other capstone project or an installed model runtime.
"""
from __future__ import annotations

import argparse
import codecs
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


class APIError(RuntimeError):
    """Connection, HTTP status, or chat protocol failure."""


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a finite number") from error
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be a finite number")
    return result


def normalize_base_url(value):
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--base-url must be an http(s) URL with a hostname")
    if parsed.hostname in {"0.0.0.0", "::"}:
        raise ValueError("Use a reachable hostname for --base-url, not a bind address")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("--base-url must not contain credentials, query parameters or a fragment")
    port = parsed.port
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Invalid port")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def redact(value, secret):
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "<redacted>")
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def iter_sse_data(chunks):
    """Decode UTF-8 SSE across arbitrary byte boundaries and LF/CRLF/CR lines.

    Only data fields are returned; comments and event/id/retry fields do not
    count as generated text. Multiple data lines form one event.
    """
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    pending = ""
    data_fields = []

    def split_lines(text, final=False):
        # Keep a trailing CR until the next chunk distinguishes CRLF from CR.
        start = index = 0
        while index < len(text):
            char = text[index]
            if char not in "\r\n":
                index += 1
                continue
            if char == "\r" and index == len(text) - 1 and not final:
                break
            yield text[start:index]
            index += 2 if char == "\r" and text[index:index + 2] == "\r\n" else 1
            start = index
        yield text[start:]

    def field(line):
        name, separator, value = line.partition(":")
        if not separator:
            value = ""
        if value.startswith(" "):
            value = value[1:]
        return name, value

    for chunk in chunks:
        pending += decoder.decode(chunk)
        lines = list(split_lines(pending))
        pending = lines.pop()
        for line in lines:
            if not line:
                if data_fields:
                    yield "\n".join(data_fields)
                    data_fields = []
            elif not line.startswith(":"):
                name, value = field(line)
                if name == "data":
                    data_fields.append(value)
    pending += decoder.decode(b"", final=True)
    lines = list(split_lines(pending, final=True))
    for line in lines:
        if not line:
            if data_fields:
                yield "\n".join(data_fields)
                data_fields = []
        elif not line.startswith(":"):
            name, value = field(line)
            if name == "data":
                data_fields.append(value)
    if data_fields:
        yield "\n".join(data_fields)


def checked_object(value):
    if not isinstance(value, dict):
        raise APIError("Expected a JSON object")
    if value.get("error") is not None:
        raise APIError("Server returned an error: " + json.dumps(value["error"], ensure_ascii=False))
    return value


def message_text(message):
    if not isinstance(message, dict):
        raise APIError("Expected a message/delta object")
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise APIError("Expected text content/reasoning_content")
    return content, reasoning


def usage_tokens(usage, name):
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else None


def client_metrics(usage, elapsed, first_text=None, last_text=None):
    tokens = usage_tokens(usage, "completion_tokens")
    tpot = None
    if (tokens is not None and tokens > 1 and first_text is not None
            and last_text is not None and last_text >= first_text):
        tpot = (last_text - first_text) / (tokens - 1)
    return {
        "end_to_end_seconds": elapsed,
        "time_to_first_text_seconds": first_text,
        "time_to_last_text_seconds": last_text,
        "tpot_text_event_seconds": tpot,
        "output_tokens_per_second": tokens / elapsed if tokens is not None and elapsed > 0 else None,
        "tpot_note": "text-event approximation; SSE events are not individual GPU tokens",
        "timing_scope": "client_http_wall_time_including_network_and_server",
    }


def percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def request_payload(args, model):
    messages = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    greedy = args.temperature == 0
    payload = {
        "model": model, "messages": messages, "stream": False,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "top_p": 1.0 if greedy else args.top_p,
        "top_k": -1 if greedy or args.top_k == 0 else args.top_k,
        "seed": args.seed, "repetition_penalty": 1.0,
    }
    if args.thinking != "auto":
        payload["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    return payload


class HTTPClient:
    """One urllib opener per worker; token usage is always server-reported."""
    def __init__(self, args):
        self.base_url = normalize_base_url(args.base_url)
        self.timeout = args.timeout
        self.api_key = os.environ.get(args.api_key_env, "")
        self.opener = build_opener(ProxyHandler({})) if args.no_proxy else build_opener()

    def open(self, path, payload=None, stream=False):
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        encoded = None
        if payload is not None:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=encoded, headers=headers,
                          method="GET" if payload is None else "POST")
        try:
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as error:
            with error:
                detail = error.read(4096).decode("utf-8", errors="replace")
            raise APIError(redact(f"{path}: HTTP {error.code}: {detail}", self.api_key)) from error
        except (URLError, OSError) as error:
            raise APIError(redact(f"{path}: connection/read failed: {error}", self.api_key)) from error

    def health(self):
        with self.open("/health") as response:
            response.read()
            if response.status != 200:
                raise APIError(f"Health check returned HTTP {response.status}")
            return {"http_status": response.status}

    def json_call(self, path, payload=None):
        with self.open(path, payload) as response:
            try:
                value = json.load(response)
            except (ValueError, UnicodeError) as error:
                raise APIError(f"{path}: invalid JSON response") from error
        return checked_object(value)

    def models(self):
        rows = self.json_call("/v1/models").get("data")
        if not isinstance(rows, list) or not rows:
            raise APIError("/v1/models returned no models")
        if any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
               or not row["id"].strip() for row in rows):
            raise APIError("/v1/models returned invalid model IDs")
        return {"model_ids": [row["id"] for row in rows]}

    def chat(self, payload):
        started = time.perf_counter()
        response = self.json_call("/v1/chat/completions", dict(payload, stream=False))
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise APIError("Chat response has no valid choices")
        content, reasoning = message_text(choices[0].get("message"))
        reason = choices[0].get("finish_reason")
        if not isinstance(reason, str) or not reason:
            raise APIError("Chat response has no finish_reason")
        if not content and not reasoning:
            raise APIError("Chat response contains no generated text")
        return {
            "text": content, "reasoning_content": reasoning, "finish_reason": reason,
            "usage": response.get("usage"), "server_metrics": response.get("profiling"),
            "client_metrics": client_metrics(response.get("usage"), time.perf_counter() - started),
        }

