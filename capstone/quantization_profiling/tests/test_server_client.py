"""Exercise the remote client over loopback HTTP; no vLLM, GPU or model is needed."""
from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_server.py"
SPEC = importlib.util.spec_from_file_location("quantization_server_client", SCRIPT)
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, body, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_body(json.dumps(value, ensure_ascii=False).encode("utf-8"), status)

    def do_GET(self):
        state = self.server.state
        state["calls"].append({"path": self.path, "auth": self.headers.get("Authorization")})
        if self.path == "/health":
            if state["mode"] == "unhealthy":
                self.send_json({"error": "engine is not ready"}, 503)
            else:
                self.send_body(b"", content_type="text/plain")
        elif self.path == "/v1/models":
            if state["mode"] == "empty_models":
                self.send_json({"data": []})
            else:
                self.send_json({"data": [{"id": "local-model"}]})
        else:
            self.send_json({"error": "unknown path"}, 404)

    def do_POST(self):
        state = self.server.state
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with state["lock"]:
            state["post_count"] += 1
            index = state["post_count"]
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        if state.get("delay"):
            time.sleep(state["delay"])
        with state["lock"]:
            state["active"] -= 1
        if state["mode"] == "mixed" and index % 2 == 0:
            self.send_json({"error": "intentional failure"}, 503)
            return
        state["calls"].append({"path": self.path, "payload": payload,
                               "auth": self.headers.get("Authorization")})
        if state["mode"] == "http_error":
            self.send_json({"error": {"message": "bad request"}}, 400)
            return
        if state["mode"] == "bad_json":
            self.send_body(b"not JSON")
            return
        if state["mode"] == "empty_response":
            self.send_json({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
            return
        if not payload["stream"]:
            self.send_json({
                "choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
                **({"profiling": {"inference_seconds": 0.2, "output_tokens_per_second": 10}}
                   if state["mode"] == "profiled" else {}),
            })
            return
        if state["mode"] == "wrong_stream_type":
            self.send_json({"message": "not SSE"})
            return
        events = [
            ": heartbeat",
            'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}',
        ]
        if state["mode"] == "profiled":
            events.append('data: {"choices":[],"profiling":{"inference_seconds":0.2,"output_tokens_per_second":10}}')
        if state["mode"] == "stream_error":
            events.append('data: {"error":{"message":"worker failed"}}')
        elif state["mode"] == "bad_stream_json":
            events.append("data: not-json")
        elif state["mode"] != "truncated":
            events.append("data: [DONE]")
        self.send_body(("\n\n".join(events) + "\n\n").encode("utf-8"),
                       content_type="text/event-stream; charset=utf-8")


class ServerClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(
            target=lambda: cls.server.serve_forever(poll_interval=0.01), daemon=True
        )
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        self.server.state = {"mode": "normal", "calls": [], "lock": threading.Lock(),
                             "post_count": 0, "active": 0, "peak": 0}
        temporary = tempfile.TemporaryDirectory(prefix="vllm-client-test-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "checks.json"
        environment = patch.dict(os.environ, {"VLLM_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = client.main([
                "--base-url", self.base_url, "--no-proxy", "--output", str(self.output),
                "--log-dir", str(self.output.parent / "logs"), *args,
            ])
        report = json.loads(self.output.read_text(encoding="utf-8"))
        return code, report, stdout.getvalue(), stderr.getvalue()

    def test_plain_http_generation_discovers_alias_and_sends_requested_protocol(self):
        code, report, stdout, stderr = self.invoke("--prompt", "中文问题")
        self.assertEqual(code, 0, stderr)
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["checks"]), 3)
        self.assertEqual(report["model"], "local-model")
        self.assertEqual(report["checks"][-1]["result"]["text"], "你好")
        call = self.server.state["calls"][-1]
        self.assertEqual(call["payload"]["messages"][-1]["content"], "中文问题")
        self.assertFalse(call["payload"]["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(call["payload"]["stream"])

    def test_stream_also_checks_terminal_event_usage_and_utf8_content(self):
        code, report, stdout, stderr = self.invoke("--stream")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(report["checks"]), 4)
        stream = report["checks"][-1]["result"]
        self.assertEqual(stream["text"], "你好")
        self.assertEqual(stream["finish_reason"], "stop")
        self.assertEqual(stream["usage"]["completion_tokens"], 2)
        self.assertEqual(stream["text_chunks"], 2)
        self.assertTrue(self.server.state["calls"][-1]["payload"]["stream_options"]["include_usage"])

    def test_truncated_stream_is_failure_even_after_receiving_text(self):
        self.server.state["mode"] = "truncated"
        code, report, stdout, stderr = self.invoke("--stream")
        self.assertEqual(code, 1)
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][-2]["ok"])
        self.assertFalse(report["checks"][-1]["ok"])
        self.assertIn("Incomplete SSE", report["error"])

    def test_stream_protocol_errors_are_failures(self):
        for mode, expected in (
            ("stream_error", "worker failed"),
            ("bad_stream_json", "not valid JSON"),
            ("wrong_stream_type", "text/event-stream"),
        ):
            with self.subTest(mode=mode):
                self.server.state["mode"] = mode
                code, report, stdout, stderr = self.invoke("--stream")
                self.assertEqual(code, 1)
                self.assertIn(expected, report["error"])

    def test_health_failure_stops_and_saves_partial_report(self):
        self.server.state["mode"] = "unhealthy"
        code, report, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(len(report["checks"]), 1)
        self.assertIn("HTTP 503", report["error"])
        self.assertEqual(len(self.server.state["calls"]), 1)

    def test_json_and_http_errors_are_not_reported_as_success(self):
        for mode, expected in (
            ("bad_json", "not valid JSON"), ("http_error", "HTTP 400"),
            ("empty_response", "no text"), ("empty_models", "no models"),
        ):
            with self.subTest(mode=mode):
                self.server.state["mode"] = mode
                code, report, stdout, stderr = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn(expected, report["error"])

    def test_check_only_never_sends_a_generation_request(self):
        code, report, stdout, stderr = self.invoke("--check-only")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(report["checks"]), 2)
        self.assertNotIn("request", report)
        self.assertTrue(all("payload" not in call for call in self.server.state["calls"]))

    def test_wrong_model_is_rejected_before_generation(self):
        code, report, stdout, stderr = self.invoke("--model", "not-served")
        self.assertEqual(code, 1)
        self.assertIn("not served", report["error"])
        self.assertTrue(all("payload" not in call for call in self.server.state["calls"]))

    def test_optional_v1_suffix_is_normalized_without_duplicate_paths(self):
        code, report, stdout, stderr = self.invoke("--base-url", self.base_url + "/v1/", "--check-only")
        self.assertEqual(code, 0, stderr)
        self.assertEqual([call["path"] for call in self.server.state["calls"]],
                         ["/health", "/v1/models"])

    def test_api_key_is_sent_as_header_and_not_written_to_report(self):
        with patch.dict(os.environ, {"VLLM_API_KEY": "fake-unit-test-secret"}):
            code, report, stdout, stderr = self.invoke()
        self.assertEqual(code, 0, stderr)
        self.assertTrue(all(call["auth"] == "Bearer fake-unit-test-secret"
                            for call in self.server.state["calls"]))
        self.assertNotIn("fake-unit-test-secret", self.output.read_text(encoding="utf-8"))
        self.assertNotIn("fake-unit-test-secret", stdout + stderr)
        for path in report["log_files"].values():
            self.assertNotIn("fake-unit-test-secret", Path(path).read_text(encoding="utf-8"))

    def test_auto_thinking_omits_extension_and_greedy_normalizes_sampling(self):
        args = client.parse_args(["--thinking", "auto", "--temperature", "0", "--top-k", "0"])
        payload = client.request_payload(args, "local-model")
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual((payload["top_p"], payload["top_k"]), (1, -1))

    def test_sse_multiline_data_comments_and_unterminated_final_line(self):
        lines = [
            b": comment\r\n", b"event: message\r\n",
            b'data: {"choices":\r\n', b'data: []}\r\n', b"\r\n",
            b"data: [DONE]",
        ]
        self.assertEqual(list(client.iter_sse_data(lines)), ['{"choices":\n[]}', "[DONE]"])

    def test_real_http_requests_overlap_and_all_are_logged(self):
        self.server.state["delay"] = 0.05
        code, report, stdout, stderr = self.invoke("--concurrency", "3", "--requests", "8")
        self.assertEqual(code, 0, stderr)
        self.assertGreater(self.server.state["peak"], 1)
        self.assertLessEqual(self.server.state["peak"], 3)
        summary = report["summaries"][0]
        self.assertEqual(summary["successful_requests"], 8)
        self.assertEqual(summary["output_tokens"], 16)
        self.assertAlmostEqual(summary["output_tokens_per_second"], 16 / summary["wall_seconds"])
        events = [json.loads(line) for line in Path(report["log_files"]["requests"]).read_text(
            encoding="utf-8").splitlines()]
        requests = [event["data"] for event in events if event["event"] == "request"]
        self.assertEqual(sorted(entry["request_index"] for entry in requests), list(range(1, 9)))
        self.assertTrue(all(Path(path).is_file() for path in report["log_files"].values()))
        self.assertIn("Metrics:", Path(report["log_files"]["console"]).read_text(encoding="utf-8"))

    def test_partial_failures_do_not_stop_remaining_requests(self):
        self.server.state["mode"] = "mixed"
        code, report, stdout, stderr = self.invoke("--concurrency", "2", "--requests", "6")
        self.assertEqual(code, 1)
        summary = report["summaries"][0]
        self.assertEqual((summary["successful_requests"], summary["failed_requests"]), (3, 3))
        self.assertEqual(summary["error_rate"], 0.5)
        self.assertEqual(summary["output_tokens"], 6)
        self.assertEqual(len(report["checks"]), 8)

    def test_server_metrics_are_received_for_json_and_sse(self):
        self.server.state["mode"] = "profiled"
        code, report, stdout, stderr = self.invoke("--stream", "--require-server-metrics")
        self.assertEqual(code, 0, stderr)
        for entry in report["checks"][2:]:
            self.assertEqual(entry["result"]["server_metrics"]["output_tokens_per_second"], 10)
            self.assertGreater(entry["result"]["client_metrics"]["end_to_end_seconds"], 0)
        self.assertEqual(len(report["summaries"]), 2)

    def test_stream_only_load_sends_exactly_requested_streams(self):
        self.server.state["delay"] = 0.02
        code, report, stdout, stderr = self.invoke("--stream-only", "--concurrency", "2", "--requests", "4")
        self.assertEqual(code, 0, stderr)
        posts = [call for call in self.server.state["calls"] if "payload" in call]
        self.assertEqual(len(posts), 4)
        self.assertTrue(all(call["payload"]["stream"] for call in posts))
        self.assertEqual(report["summaries"][0]["mode"], "stream")

    def test_default_run_always_saves_logs_without_output(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = client.main(["--base-url", self.base_url, "--no-proxy", "--check-only",
                                "--log-dir", str(self.output.parent / "automatic")])
        self.assertEqual(code, 0)
        files = list((self.output.parent / "automatic").iterdir())
        self.assertEqual({path.suffix for path in files}, {".json", ".jsonl", ".log"})
        report = json.loads(next(path for path in files if path.suffix == ".json").read_text(encoding="utf-8"))
        self.assertTrue(report["ok"])

    def test_missing_server_metrics_can_be_required(self):
        code, report, stdout, stderr = self.invoke("--require-server-metrics")
        self.assertEqual(code, 1)
        self.assertIn("Missing server profiling", report["error"])
        self.assertIn("result", report["checks"][-1])

    def test_missing_usage_never_fabricates_zero_throughput(self):
        entry = {"ok": True, "result": {"usage": None,
                 "client_metrics": {"end_to_end_seconds": 2.0}}}
        summary = client.summarize_requests([entry], 2.0, 1)
        self.assertIsNone(summary["output_tokens"])
        self.assertIsNone(summary["output_tokens_per_second"])
        self.assertFalse(summary["usage_complete"])
        self.assertEqual(client.percentile([1, 2, 3, 4], 0.95), 4)

    def test_invalid_target_and_numbers_fail_before_connecting(self):
        for arguments in (
            ["--base-url", "http://0.0.0.0:7081"],
            ["--base-url", "http://[::]:7081"],
            ["--base-url", "ftp://server:7081"],
            ["--base-url", "http://user:password@server:7081"],
            ["--base-url", "http://server:invalid"],
            ["--timeout", "nan"], ["--timeout", "0"], ["--max-tokens", "0"],
            ["--check-only", "--stream"], ["--concurrency", "0"], ["--requests", "-1"],
            ["--stream", "--stream-only"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    client.parse_args(arguments)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
