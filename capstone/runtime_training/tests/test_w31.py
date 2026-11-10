"""W31 protocol and measurement regressions; no model, GPU or vLLM required."""
from __future__ import annotations

from collections import Counter
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name, dependencies=None):
    # Distinct import names keep this suite isolated during repository discovery.
    spec = importlib.util.spec_from_file_location("_p3_w31_test_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies or {}):
        spec.loader.exec_module(module)
    return module


http_client = load_script("http_client")
common = load_script("common", {"http_client": http_client})
summarize = load_script("summarize", {"common": common})
client = load_script("client", {"common": common})
benchmark = load_script("benchmark", {"common": common, "summarize": summarize})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def remember(self, payload=None):
        state = self.server.state
        with state["lock"]:
            state["calls"].append({"path": self.path, "payload": payload,
                                   "auth": self.headers.get("Authorization")})
        if state["required_key"] and self.headers.get("Authorization") != "Bearer " + state["required_key"]:
            self.send_json({"error": "authentication failed"}, 401)
            return False
        return True

    def do_GET(self):
        if not self.remember():
            return
        if self.path == "/health":
            self.send_json({"healthy": True})
        elif self.path == "/v1/models":
            self.send_json({"data": [{"id": "local-model"}]})
        elif self.path == "/version":
            self.send_json({"version": "test-server"})
        else:
            self.send_json({"error": "unknown route"}, 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.remember(payload):
            return
        prompt = payload["messages"][-1]["content"]
        if prompt == "failure":
            self.send_json({"error": "failed request; echoed key=" + self.server.state["required_key"]}, 500)
            return
        finish = "length" if prompt == "length" else "stop"
        token_count = 1 if prompt == "single" else 5
        usage = {"prompt_tokens": 9, "completion_tokens": token_count, "total_tokens": 9 + token_count}
        if not payload.get("stream"):
            response = {"choices": [{"message": {"content": "完整回答"}, "finish_reason": finish}]}
            if prompt != "no_usage":
                response["usage"] = usage
            self.send_json(response)
            return

        def choice(delta, reason=None):
            return {"choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}

        events = [choice({"role": "assistant", "content": ""}), choice({})]
        if prompt == "single":
            events.append(choice({"content": "一"}))
        else:
            events.extend([choice({"reasoning_content": "思考"}),
                           choice({"content": "多个 token 同属一个事件。"}),
                           choice({"content": "结束"})])
        events.append(choice({}, finish))
        if prompt != "no_usage":
            events.append({"choices": [], "usage": usage})
        encoded = [b": heartbeat\r\n\r\n"]
        encoded.extend(("data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n").encode("utf-8")
                       for event in events)
        if prompt != "missing_done":
            encoded.append(b"data: [DONE]\r\n\r\n")
        body = b"".join(encoded)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # Force network writes through arbitrary UTF-8 and SSE line boundaries.
        for event_index, event in enumerate(encoded):
            for offset in range(0, len(event), 11):
                self.wfile.write(event[offset:offset + 11])
                self.wfile.flush()
            if prompt == "live" and event_index == 3:
                self.server.state["callback_before_end"] = self.server.state["delta_seen"].wait(2)


class W31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=lambda: cls.server.serve_forever(poll_interval=0.01), daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        self.server.state = {"calls": [], "lock": threading.Lock(), "required_key": "",
                             "delta_seen": threading.Event(), "callback_before_end": False}
        temporary = tempfile.TemporaryDirectory(prefix="p3-w31-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        environment = patch.dict(os.environ, {"VLLM_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def args(self):
        return client.parse_args(["--base-url", self.base_url, "--model", "local-model", "--no-proxy"])

    def workload(self, prompt="normal"):
        return {"id": prompt, "prompt": prompt, "max_tokens": 32}

    def capture(self, module, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = module.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def invoke_client(self, *extra):
        output = self.directory / "client-runs"
        code, stdout, stderr = self.capture(client, ["--base-url", self.base_url, "--no-proxy",
                                                    "--output-dir", str(output), *extra])
        runs = list(output.iterdir())
        self.assertEqual(len(runs), 1)
        return code, runs[0], stdout, stderr

    def invoke_benchmark(self, *extra, workloads=None):
        workload_path = self.directory / "workloads.jsonl"
        rows = workloads or [self.workload("normal"), self.workload("length")]
        workload_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        output = self.directory / "benchmark-runs"
        before = set(output.iterdir()) if output.exists() else set()
        code, stdout, stderr = self.capture(benchmark, [
            "--base-url", self.base_url, "--no-proxy", "--workloads", str(workload_path),
            "--output-dir", str(output), "--requests-per-workload", "4", "--concurrency", "1", "4",
            "--repeats", "2", "--warmup", "1", *extra,
        ])
        created = set(output.iterdir()) - before
        self.assertEqual(len(created), 1)
        return code, created.pop(), stdout, stderr

    def test_sse_parser_handles_each_byte_boundary_bom_all_newlines_and_rejects_invalid_utf8(self):
        source = ("\ufeff: 中文心跳\r\nid: 7\r\nevent: message\r\nrole: assistant\r\nretry: 1000\r\n"
                  "data: 第一行\r\ndata: 第二行\r\n\r\n"
                  ": ignored\rdata: 第三\r\rdata: 第四\n\ndata: [DONE]")
        encoded = source.encode("utf-8")
        events = list(http_client.iter_sse_data(bytes([byte]) for byte in encoded))
        self.assertEqual(events, ["第一行\n第二行", "第三", "第四", "[DONE]"])
        for malformed in (b"data: \xff\n\n", b"data: \xe4\xb8"):
            with self.subTest(malformed=malformed):
                with self.assertRaises(UnicodeDecodeError):
                    list(http_client.iter_sse_data(bytes([byte]) for byte in malformed))

    def test_fragmented_stream_preserves_text_ignores_empty_events_and_uses_usage(self):
        deltas = []
        payload = common.payload_for(self.args(), self.workload(), "local-model")
        result = common.RuntimeClient(self.args()).stream_chat(payload, on_delta=lambda c, r: deltas.append((c, r)))
        self.assertEqual(result["text"], "多个 token 同属一个事件。结束")
        self.assertEqual(result["reasoning_content"], "思考")
        self.assertEqual(result["text_chunks"], 3)
        self.assertEqual(len(deltas), 3)
        self.assertEqual(result["usage"]["completion_tokens"], 5)
        self.assertEqual(result["finish_reason"], "stop")
        metrics = result["client_metrics"]
        self.assertGreaterEqual(metrics["time_to_first_text_seconds"], 0)
        self.assertGreaterEqual(metrics["tpot_text_event_seconds"], 0)
        self.assertIn("approximation", metrics["tpot_note"])
        post = self.server.state["calls"][-1]["payload"]
        self.assertTrue(post["stream_options"]["include_usage"])

    def test_stream_callback_happens_before_completion_of_http_response(self):
        def delta(content, reasoning):
            self.server.state["delta_seen"].set()
        payload = common.payload_for(self.args(), self.workload("live"), "local-model")
        common.RuntimeClient(self.args()).stream_chat(payload, on_delta=delta)
        self.assertTrue(self.server.state["callback_before_end"])

    def test_truncated_stream_is_failure_even_when_text_and_finish_reason_arrived(self):
        record = common.measure_request(self.args(), self.workload("missing_done"), "local-model", "case", 1)
        self.assertFalse(record["ok"])
        self.assertIn("missing [DONE]", record["error"])
        self.assertIsNone(record["output_tokens"])

    def test_single_token_and_non_streaming_do_not_claim_per_token_timing(self):
        single = common.measure_request(self.args(), self.workload("single"), "local-model", "case", 1)
        self.assertTrue(single["ok"])
        self.assertEqual(single["output_tokens"], 1)
        self.assertIsNone(single["tpot_text_seconds"])
        plain = common.measure_request(self.args(), self.workload(), "local-model", "case", 2, streaming=False)
        self.assertTrue(plain["ok"])
        self.assertEqual(plain["mode"], "non_stream")
        self.assertIsNone(plain["ttft_text_seconds"])
        self.assertIsNone(plain["tpot_text_seconds"])

    def test_length_is_a_valid_completed_request(self):
        record = common.measure_request(self.args(), self.workload("length"), "local-model", "case", 1)
        self.assertTrue(record["ok"])
        self.assertEqual(record["finish_reason"], "length")

    def test_missing_usage_retains_latency_without_fabricating_throughput(self):
        record = common.measure_request(self.args(), self.workload("no_usage"), "local-model", "case", 1)
        self.assertTrue(record["ok"])
        self.assertIsNone(record["output_tokens"])
        case = {"case_id": "case", "total_requests": 1, "wall_seconds": 2.0}
        result = common.summarize_case([record], case)
        self.assertFalse(result["usage_complete"])
        self.assertIsNone(result["output_tokens_per_second"])
        self.assertEqual(result["latency_seconds"]["samples"], 1)

    def test_model_validation_prevents_generation_and_records_failure(self):
        code, run, stdout, stderr = self.invoke_client("--model", "unknown-model")
        self.assertEqual(code, 1)
        self.assertIn("not served", stderr)
        self.assertFalse(any(call["payload"] is not None for call in self.server.state["calls"]))
        self.assertEqual(common.read_jsonl(run / "metrics.jsonl"), [])
        self.assertEqual(json.loads((run / "manifest.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_client_sends_auth_but_redacts_an_echoed_key_from_artifacts_and_console(self):
        secret = "synthetic-w31-auth-secret"
        self.server.state["required_key"] = secret
        with patch.dict(os.environ, {"VLLM_API_KEY": secret}):
            code, run, stdout, stderr = self.invoke_client("--prompt", "failure", "--stream")
        self.assertEqual(code, 1)
        self.assertTrue(all(call["auth"] == "Bearer " + secret for call in self.server.state["calls"]))
        self.assertNotIn(secret, stdout + stderr)
        self.assertIn("<redacted>", stderr)
        for path in run.iterdir():
            self.assertNotIn(secret, path.read_text(encoding="utf-8"), str(path))

    def test_client_defaults_to_non_stream_and_persists_its_measurement(self):
        code, run, stdout, stderr = self.invoke_client("--model", "local-model", "--prompt", "normal", "--thinking", "off")
        self.assertEqual(code, 0, stderr)
        record, = common.read_jsonl(run / "metrics.jsonl")
        self.assertTrue(record["ok"])
        self.assertEqual(record["mode"], "non_stream")
        self.assertIn("完整回答", stdout)
        payload = self.server.state["calls"][-1]["payload"]
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertIsNone(record["ttft_text_seconds"])

    def test_failure_rate_and_throughput_use_full_measurement_wall_time(self):
        records = [
            {"case_id": "c", "request_index": 1, "ok": True, "output_tokens": 10,
             "elapsed_seconds": 1.0, "ttft_text_seconds": .1, "tpot_text_seconds": .05, "finish_reason": "stop"},
            {"case_id": "c", "request_index": 2, "ok": False, "elapsed_seconds": 4.0},
        ]
        result = common.summarize_case(records, {"case_id": "c", "total_requests": 2, "wall_seconds": 5.0})
        self.assertEqual(result["error_rate"], .5)
        self.assertEqual(result["output_tokens_per_second"], 2.0)
        self.assertEqual(result["requests_per_second"], .2)
        self.assertEqual(result["latency_seconds"], {"samples": 1, "p50": 1.0, "p95": 1.0})
        self.assertIn("includes failed", result["timing_scope"])

    def test_benchmark_keeps_workloads_concurrency_repeats_and_warmup_separate(self):
        code, run, stdout, stderr = self.invoke_benchmark()
        self.assertEqual(code, 0, stderr)
        metrics = common.read_jsonl(run / "metrics.jsonl")
        warmup = common.read_jsonl(run / "warmup.jsonl")
        cases = common.read_jsonl(run / "cases.jsonl")
        result = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(result["cases"]), 8)
        self.assertEqual(len(metrics), 32)
        self.assertEqual(len(warmup), 8)
        self.assertEqual(Counter(row["case_id"] for row in metrics), {case["case_id"]: 4 for case in cases})
        self.assertEqual({(case["workload_id"], case["concurrency"], case["repeat_index"]) for case in cases},
                         {(w, c, r) for w in ("normal", "length") for c in (1, 4) for r in (1, 2)})
        for case in result["cases"]:
            self.assertEqual(case["successful_requests"], 4)
            self.assertEqual(case["output_tokens"], 20)
            self.assertAlmostEqual(case["output_tokens_per_second"], 20 / case["wall_seconds"])
            self.assertEqual(case["ttft_text_seconds"]["samples"], 4)
        self.assertEqual(sum(call["payload"] is not None for call in self.server.state["calls"]), 40)
        self.assertTrue((run / "summary.csv").is_file())
        before = result["cases"]
        code, stdout, stderr = self.capture(summarize, [str(run)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads((run / "summary.json").read_text(encoding="utf-8"))["cases"], before)

    def test_dry_run_never_contacts_server_and_does_not_claim_performance(self):
        code, run, stdout, stderr = self.invoke_benchmark("--dry-run")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.server.state["calls"], [])
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "dry_run")
        self.assertEqual(manifest["planned_cases"], 8)
        self.assertEqual(common.read_jsonl(run / "metrics.jsonl"), [])
        self.assertFalse((run / "summary.json").exists())
        code, stdout, stderr = self.capture(summarize, [str(run)])
        self.assertEqual(code, 1)
        self.assertIn("No completed", stderr)

    def test_failed_warmup_stops_before_measured_cases(self):
        code, run, stdout, stderr = self.invoke_benchmark(workloads=[self.workload("failure")])
        self.assertEqual(code, 1)
        self.assertIn("Warmup failed", stderr)
        self.assertEqual(common.read_jsonl(run / "metrics.jsonl"), [])
        self.assertEqual(common.read_jsonl(run / "cases.jsonl"), [])
        self.assertFalse((run / "summary.json").exists())

    def test_repeated_runs_get_distinct_directories(self):
        first = self.invoke_benchmark("--dry-run")[1]
        second = self.invoke_benchmark("--dry-run")[1]
        self.assertNotEqual(first, second)

    def test_summarize_rejects_lost_duplicate_or_unknown_measurement_records(self):
        code, run, stdout, stderr = self.invoke_benchmark("--concurrency", "1", "--repeats", "1", "--warmup", "0",
                                                        workloads=[self.workload()])
        self.assertEqual(code, 0, stderr)
        metrics_path = run / "metrics.jsonl"
        rows = common.read_jsonl(metrics_path)
        variants = [rows[:-1], [rows[0], rows[0], *rows[2:]],
                    [{**rows[0], "case_id": "unfinished"}, *rows[1:]]]
        for changed in variants:
            with self.subTest(rows=changed):
                metrics_path.write_text("".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8")
                code, stdout, stderr = self.capture(summarize, [str(run)])
                self.assertEqual(code, 1)
                self.assertIn("Cannot summarize", stderr)

    def test_workload_validation_rejects_ambiguous_ids_and_boolean_token_budget(self):
        path = self.directory / "invalid.jsonl"
        for rows in ([self.workload(), self.workload()], [{**self.workload(), "max_tokens": True}]):
            with self.subTest(rows=rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    benchmark.load_workloads(path)

    def test_cli_entry_points_work_from_an_unrelated_directory(self):
        for name in ("client", "benchmark", "summarize"):
            with self.subTest(name=name):
                process = subprocess.run([sys.executable, str(SCRIPTS / (name + ".py")), "--help"],
                                         cwd=self.directory, capture_output=True, text=True, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("usage:", process.stdout)


if __name__ == "__main__":
    unittest.main()



