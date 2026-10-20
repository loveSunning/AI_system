"""Response metrics for vLLM 0.11.0's supported --middleware extension.

Only successful OpenAI chat/completion responses are extended. No engine internals
are patched. Times cover server request handling, including queueing/tokenization/
generation (and streaming backpressure), not GPU kernel time or model loading.
All state is request-local so overlapping ASGI calls cannot mix their metrics.
"""
from __future__ import annotations

import json
import time
import uuid


def token_count(usage, name):
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def response_metrics(started, usage, request_id, first_text=None):
    elapsed = time.perf_counter() - started
    output_tokens = token_count(usage, "completion_tokens")
    return {
        "schema_version": 1,
        "request_id": request_id,
        "inference_seconds": elapsed,
        "input_tokens": token_count(usage, "prompt_tokens"),
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            output_tokens / elapsed if output_tokens is not None and elapsed > 0 else None
        ),
        "time_to_first_text_seconds": first_text - started if first_text is not None else None,
        "timing_scope": "server_request_wall_time_including_queue_and_generation",
        "token_count_source": "response.usage",
    }


class ProfilingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") not in (
                    "/v1/chat/completions", "/v1/completions")):
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        request_id = "profiling-" + uuid.uuid4().hex
        response_start = None
        mode = None
        body = bytearray()
        usage = None
        envelope = {}
        first_text = None
        saw_error = False
        injected = False

        async def profiled_send(message):
            nonlocal response_start, mode, usage, first_text, saw_error, injected, request_id
            if message["type"] == "http.response.start":
                response_start = dict(message)
                headers = list(message.get("headers", []))
                header_map = {key.lower(): value for key, value in headers}
                content_type = header_map.get(b"content-type", b"").lower()
                if message["status"] == 200 and b"content-encoding" not in header_map:
                    if b"text/event-stream" in content_type:
                        mode = "sse"
                    elif b"application/json" in content_type:
                        mode = "json"
                if mode == "sse":
                    response_start["headers"] = [
                        (key, value) for key, value in headers if key.lower() != b"content-length"
                    ]
                if mode != "json":
                    await send(response_start)
                return
            if message["type"] != "http.response.body" or mode is None:
                await send(message)
                return

            body.extend(message.get("body", b""))
            more = message.get("more_body", False)
            if mode == "json":
                if more:
                    return
                raw = bytes(body)
                try:
                    data = json.loads(raw)
                except (ValueError, UnicodeError):
                    data = None
                if isinstance(data, dict) and not data.get("error"):
                    data["profiling"] = response_metrics(
                        started, data.get("usage"), data.get("id") or request_id
                    )
                    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
                response_start["headers"] = [
                    (key, value) for key, value in response_start.get("headers", [])
                    if key.lower() != b"content-length"
                ] + [(b"content-length", str(len(raw)).encode("ascii"))]
                await send(response_start)
                await send(dict(message, body=raw, more_body=False))
                return

            # Buffer at most an unfinished SSE event, never the whole generation.
            # Normalize CRLF only after appending so split delimiters are handled.
            normalized = bytes(body).replace(b"\r\n", b"\n")
            frames = normalized.split(b"\n\n")
            body.clear()
            body.extend(frames.pop())
            outgoing = bytearray()
            for frame in frames:
                fields = [line[5:].lstrip(b" ") for line in frame.split(b"\n")
                          if line.startswith(b"data:")]
                data_bytes = b"\n".join(fields)
                if data_bytes.strip() == b"[DONE]" and not injected and not saw_error:
                    # Retain the standard chunk envelope for OpenAI SDK consumers.
                    metrics = dict(envelope, choices=[], profiling=response_metrics(
                        started, usage, request_id, first_text
                    ))
                    outgoing.extend(b"data: " + json.dumps(metrics).encode("utf-8") + b"\n\n")
                    injected = True
                elif fields:
                    try:
                        data = json.loads(data_bytes)
                    except (ValueError, UnicodeError):
                        data = None
                    if isinstance(data, dict):
                        saw_error = saw_error or bool(data.get("error"))
                        for key in ("id", "object", "created", "model", "system_fingerprint"):
                            if key in data:
                                envelope[key] = data[key]
                        if data.get("id"):
                            request_id = data["id"]
                        if isinstance(data.get("usage"), dict):
                            usage = data["usage"]
                        for choice in data.get("choices", []) or []:
                            if not isinstance(choice, dict):
                                continue
                            delta = choice.get("delta") or {}
                            if (choice.get("text") or delta.get("content")
                                    or delta.get("reasoning_content") or delta.get("reasoning")):
                                if first_text is None:
                                    first_text = time.perf_counter()
                outgoing.extend(frame + b"\n\n")
            if not more:
                outgoing.extend(body)
                body.clear()
            if outgoing or not more:
                await send(dict(message, body=bytes(outgoing), more_body=more))

        await self.app(scope, receive, profiled_send)
