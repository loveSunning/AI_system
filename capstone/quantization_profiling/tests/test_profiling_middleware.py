"""Exercise the actual ASGI middleware without vLLM, FastAPI, GPU or model weights."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profiling_middleware.py"
SPEC = importlib.util.spec_from_file_location("quantization_profiling_middleware", SCRIPT)
profiling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiling)


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, app, path="/v1/chat/completions", scope_extra=None):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "method": "POST", "path": path, **(scope_extra or {})}
        await profiling.ProfilingMiddleware(app)(scope, receive, send)
        return messages

    async def test_json_counts_headers_and_utf8_are_preserved(self):
        async def app(scope, receive, send):
            raw = json.dumps({"id": "request-A", "choices": [{"message": {"content": "hello"}}],
                              "usage": {"prompt_tokens": 12, "completion_tokens": 3}}).encode()
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(raw)).encode())]})
            await asyncio.sleep(0.005)
            await send({"type": "http.response.body", "body": raw[:13], "more_body": True})
            await send({"type": "http.response.body", "body": raw[13:], "more_body": False})

        messages = await self.invoke(app)
        self.assertEqual(len(messages), 2)
        raw = messages[1]["body"]
        data = json.loads(raw)
        metrics = data["profiling"]
        self.assertEqual(dict(messages[0]["headers"])[b"content-length"], str(len(raw)).encode())
        self.assertEqual(metrics["request_id"], "request-A")
        self.assertEqual((metrics["input_tokens"], metrics["output_tokens"]), (12, 3))
        self.assertGreater(metrics["inference_seconds"], 0)
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 3 / metrics["inference_seconds"])
        self.assertIsNone(metrics["time_to_first_text_seconds"])

    async def test_split_sse_frames_include_usage_and_metrics_before_done(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
            events = [
                {"id": "request-S", "object": "chat.completion.chunk", "created": 123,
                 "model": "local-model", "choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            ]
            raw = b"".join(b"data: " + json.dumps(event).encode() + b"\r\n\r\n" for event in events)
            raw += b"data: [DONE]\r\n\r\n"
            for offset in range(0, len(raw), 7):
                await send({"type": "http.response.body", "body": raw[offset:offset + 7], "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        messages = await self.invoke(app)
        raw = b"".join(message.get("body", b"") for message in messages)
        frames = [frame[6:] for frame in raw.split(b"\n\n") if frame.startswith(b"data: ")]
        self.assertEqual(frames[-1], b"[DONE]")
        extension = json.loads(frames[-2])
        self.assertEqual(extension["id"], "request-S")
        self.assertEqual(extension["object"], "chat.completion.chunk")
        self.assertEqual(extension["model"], "local-model")
        metrics = extension["profiling"]
        self.assertEqual(metrics["output_tokens"], 2)
        self.assertEqual(metrics["request_id"], "request-S")
        self.assertIsNotNone(metrics["time_to_first_text_seconds"])
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 2 / metrics["inference_seconds"])
        self.assertEqual(sum(b'"profiling"' in frame for frame in frames), 1)

    async def test_sse_is_forwarded_before_generation_finishes(self):
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            await send({"type": "http.response.body", "body": b'data: {"choices":[]}\n\n',
                        "more_body": True})
            self.assertTrue(any(b"choices" in message.get("body", b"") for message in sent))
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            sent.append(message)

        await profiling.ProfilingMiddleware(app)(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions"}, receive, send)

    async def test_concurrent_requests_do_not_share_usage_or_identity(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json")]})
            await asyncio.sleep(0.01 if scope["number"] == 1 else 0.002)
            await send({"type": "http.response.body", "body": json.dumps({
                "id": str(scope["number"]), "usage": {"completion_tokens": scope["number"]}
            }).encode()})

        responses = await asyncio.gather(*(self.invoke(app, scope_extra={"number": i}) for i in (1, 9)))
        for number, messages in zip((1, 9), responses):
            metrics = json.loads(messages[-1]["body"])["profiling"]
            self.assertEqual(metrics["request_id"], str(number))
            self.assertEqual(metrics["output_tokens"], number)

    async def test_error_and_unrelated_routes_pass_through_unchanged(self):
        raw = b'{"error":"failure"}'
        for path, status in (("/health", 200), ("/v1/chat/completions", 400)):
            async def app(scope, receive, send):
                await send({"type": "http.response.start", "status": status,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": raw})
            messages = await self.invoke(app, path)
            self.assertEqual(messages[-1]["body"], raw)

    async def test_stream_errors_never_receive_success_metrics(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            await send({"type": "http.response.body", "body":
                        b'data: {"error":"engine failed"}\n\ndata: [DONE]\n\n'})
        messages = await self.invoke(app)
        self.assertNotIn(b'"profiling"', b"".join(message.get("body", b"") for message in messages))

    def test_missing_or_invalid_token_usage_is_explicitly_unknown(self):
        for usage in (None, {}, {"completion_tokens": True}, {"completion_tokens": -1}):
            metrics = profiling.response_metrics(profiling.time.perf_counter(), usage, "r")
            self.assertIsNone(metrics["output_tokens"])
            self.assertIsNone(metrics["output_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
