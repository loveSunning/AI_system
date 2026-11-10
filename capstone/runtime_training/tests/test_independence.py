"""Run a copied W31 project with isolated import paths and a loopback API."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest


PROJECT = Path(__file__).resolve().parents[1]
ISOLATED_LAUNCHER = """
import runpy
import sys
class NoGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm', 'torch'}:
            raise RuntimeError('GPU imports are disabled in standalone CLI tests')
sys.meta_path.insert(0, NoGpuImports())
sys.path.insert(0, sys.argv[1])
script, *arguments = sys.argv[2:]
sys.argv = [script, *arguments]
runpy.run_path(script, run_name='__main__')
"""


class StandaloneHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, data, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def send_json(self, value):
        self.send_body(json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({"data": [{"id": "standalone-model"}]})
        elif self.path == "/version":
            self.send_json({"version": "fake-loopback"})
        else:
            self.send_json({"healthy": True})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.payloads.append(payload)
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "独立"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "运行"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        ]
        body = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        self.send_body(body.encode("utf-8"), "text/event-stream; charset=utf-8")


class IndependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StandaloneHandler)
        cls.thread = threading.Thread(target=lambda: cls.server.serve_forever(poll_interval=.01), daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="w31-independent-project-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "runtime_training"
        shutil.copytree(PROJECT, self.project, ignore=shutil.ignore_patterns("runs", "__pycache__", "*.pyc"))
        self.unrelated_cwd = self.root / "unrelated-working-directory"
        self.unrelated_cwd.mkdir()
        self.server.lock = threading.Lock()
        self.server.payloads = []

    def invoke(self, script, *arguments):
        scripts = self.project / "scripts"
        environment = os.environ.copy()
        environment["VLLM_API_KEY"] = ""
        environment.pop("PYTHONPATH", None)
        command = [sys.executable, "-I", "-B", "-X", "utf8", "-c", ISOLATED_LAUNCHER,
                   str(scripts), str(scripts / script), *map(str, arguments)]
        return subprocess.run(command, cwd=self.unrelated_cwd, env=environment,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def only_run(self, output):
        runs = list(output.iterdir())
        self.assertEqual(len(runs), 1)
        return runs[0]

    def read_jsonl(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    def test_copied_project_plans_and_help_work_without_gpu_imports(self):
        # Only the copied project and an unrelated cwd exist under this empty parent.
        self.assertEqual({path.name for path in self.root.iterdir()},
                         {"runtime_training", "unrelated-working-directory"})
        server_output = self.root / "server-dry-run"
        process = self.invoke("serve.py", "--dry-run", "--output-dir", server_output)
        self.assertEqual(process.returncode, 0, process.stderr)
        manifest = json.loads((server_output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "dry_run")
        self.assertFalse(manifest["execution_started"])
        self.assertTrue((server_output / "startup.log").is_file())

        benchmark_output = self.root / "benchmark-dry-run"
        process = self.invoke("benchmark.py", "--dry-run", "--output-dir", benchmark_output)
        self.assertEqual(process.returncode, 0, process.stderr)
        run = self.only_run(benchmark_output)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "dry_run")
        self.assertGreater(manifest["planned_cases"], 0)
        self.assertFalse((run / "summary.json").exists())
        self.assertEqual(self.server.payloads, [])
        for script in ("client.py", "summarize.py"):
            with self.subTest(script=script):
                process = self.invoke(script, "--help")
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("usage:", process.stdout)

    def test_copied_project_streams_benchmarks_and_recomputes_saved_results(self):
        client_output = self.root / "client-results"
        process = self.invoke("client.py", "--base-url", self.base_url, "--no-proxy", "--stream",
                              "--model", "standalone-model", "--prompt", "standalone request",
                              "--max-tokens", "8", "--output-dir", client_output)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("独立运行", process.stdout)
        client_run = self.only_run(client_output)
        record, = self.read_jsonl(client_run / "metrics.jsonl")
        self.assertTrue(record["ok"])
        self.assertEqual(record["text"], "独立运行")
        self.assertEqual(record["output_tokens"], 2)
        self.assertIsNotNone(record["ttft_text_seconds"])

        workloads = self.root / "one-workload.jsonl"
        workloads.write_text(json.dumps({"id": "one", "prompt": "benchmark request", "max_tokens": 8}) + "\n",
                             encoding="utf-8")
        benchmark_output = self.root / "benchmark-results"
        process = self.invoke("benchmark.py", "--base-url", self.base_url, "--no-proxy",
                              "--workloads", workloads, "--concurrency", "1", "--requests-per-workload", "1",
                              "--repeats", "1", "--warmup", "0", "--output-dir", benchmark_output)
        self.assertEqual(process.returncode, 0, process.stderr)
        run = self.only_run(benchmark_output)
        self.assertEqual(len(self.read_jsonl(run / "metrics.jsonl")), 1)
        self.assertEqual(len(self.read_jsonl(run / "cases.jsonl")), 1)
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        case, = summary["cases"]
        self.assertEqual(case["successful_requests"], 1)
        self.assertEqual(case["error_rate"], 0)
        self.assertEqual(case["output_tokens"], 2)
        self.assertGreater(case["output_tokens_per_second"], 0)
        self.assertTrue((run / "summary.csv").is_file())
        self.assertEqual(len(self.server.payloads), 2)
        self.assertTrue(all(payload["stream"] for payload in self.server.payloads))
        self.assertTrue(all(payload["model"] == "standalone-model" for payload in self.server.payloads))
        process = self.invoke("summarize.py", run)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads((run / "summary.json").read_text(encoding="utf-8"))["cases"], summary["cases"])


if __name__ == "__main__":
    unittest.main()
