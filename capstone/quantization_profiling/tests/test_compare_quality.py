"""Synthetic HTTP and artifacts test quality gates; these are not GPU quality results."""
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_quality as quality
from experiment_utils import ROOT, digest, read, rows, write, write_rows


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        for name in ("quality_holdout.jsonl", "quality_protocol.json"):
            shutil.copyfile(ROOT / "data" / name, self.inputs / name)
        write(self.inputs / "model.json", {"path": "/synthetic/base"})
        write(self.inputs / "workloads.yaml", {})
        write_rows(self.inputs / "requests.jsonl", [])
        (self.inputs / "chat_template.jinja").write_text("template", encoding="utf-8")
        write(self.inputs / "manifest.json", {
            "scorer_sha256": digest(Path(quality.evaluator.__file__)),
            "files": {p.name: digest(p) for p in self.inputs.iterdir()}})
        self.tasks, _ = quality.check_holdout(self.inputs)
        write_rows(self.root / "quality_requests.jsonl",
                   [dict(id=t["id"], input_tokens=7) for t in self.tasks])
        self.manifest = digest(self.inputs / "manifest.json")
        self.profiles = ["bf16", "awq", "fp8-official", "fp8-local"]
        write(self.root / "run.json", dict(run_id="synthetic-not-a-gpu-run", profiles=self.profiles,
            manifest_sha256=self.manifest, quality_requests_sha256=digest(self.root / "quality_requests.jsonl")))

    def artifact(self, profile, wrong=(), infrastructure=(), truncated=(), missing=False,
                 bad_token=False, wrong_saved_score=False, precision=True):
        entries = []
        for index, task in enumerate(self.tasks):
            text = (json.dumps(task["expected"]) if task["scorer"] == "json_exact" else task["expected"])
            if index in wrong:
                text = "incorrect answer"
            result = dict(text=text, finish_reason="length" if index in truncated else "stop",
                          usage={"prompt_tokens": 8 if bad_token and index == 0 else 7})
            entry = dict(id=task["id"], category=task["category"], expected=task["expected"], ok=True,
                         result=result, correct=quality.evaluator.score(task, text, result["finish_reason"]))
            if index in infrastructure:
                entry = dict(id=task["id"], category=task["category"], expected=task["expected"],
                             ok=False, error="synthetic HTTP timeout")
            entries.append(entry)
        if missing:
            entries.pop()
        if wrong_saved_score:
            entries[0]["correct"] = not entries[0]["correct"]
        directory = self.root / profile
        write_rows(directory / "evaluation/results.jsonl", entries)
        write(directory / "evaluation/summary.json", dict(label=profile, manifest_sha256=self.manifest))
        write(directory / "deployment.json", dict(profile=profile, status="complete",
            precision_verified=precision, model={"base_source_verification": "synthetic"},
            artifacts_sha256={name: digest(directory / name)
                for name in ("evaluation/results.jsonl", "evaluation/summary.json")}))
        return directory

    def test_paired_errors_and_inclusive_two_percentage_point_gate(self):
        self.artifact("bf16", wrong=(0, 1))  # 98/100
        self.artifact("awq", wrong=(1, 2, 3, 4))  # 96/100: 3 new errors, 1 improvement
        self.artifact("fp8-official", wrong=(0, 1))
        self.artifact("fp8-local", wrong=(0, 1))
        self.assertEqual(quality.summarize(self.root), 0)
        summary = read(self.root / "summary.json")
        awq = summary["comparisons"]["awq"]
        self.assertEqual((awq["delta_percentage_points"], awq["regressions"], awq["improvements"]), (-2, 3, 1))
        self.assertTrue(awq["gate_passed"])
        self.assertEqual(len(rows(self.root / "paired_results.jsonl")), 300)
        self.assertEqual({x["id"] for x in rows(self.root / "regressions.jsonl")},
                         {self.tasks[i]["id"] for i in (2, 3, 4)})
        self.assertIn("AWQ/W4A16", (self.root / "Quality_Report.md").read_text(encoding="utf-8"))
        self.assertTrue((self.root / "quality_categories.csv").exists())

    def test_quality_drop_is_failed_not_infrastructure_incomplete(self):
        for profile in self.profiles:
            self.artifact(profile, wrong=range(3) if profile == "fp8-local" else ())
        self.assertEqual(quality.summarize(self.root), 2)
        summary = read(self.root / "summary.json")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["comparisons"]["fp8-local"]["gate_passed"])

    def test_http_failure_missing_ids_and_wrong_precision_never_pass(self):
        for kwargs in (dict(infrastructure=(0,)), dict(missing=True), dict(precision=False),
                       dict(bad_token=True), dict(wrong_saved_score=True)):
            with self.subTest(kwargs=kwargs):
                for profile in self.profiles:
                    self.artifact(profile, **(kwargs if profile == "bf16" else {}))
                self.assertEqual(quality.summarize(self.root), 1)
                summary = read(self.root / "summary.json")
                self.assertIsNone(summary["models"]["bf16"]["accuracy"])
                self.assertIsNone(summary["all_quality_gates_passed"])
                self.assertIsNone(summary["comparisons"]["awq"]["delta_percentage_points"])
                self.assertEqual(rows(self.root / "paired_results.jsonl"), [])

    def test_truncation_is_a_quality_error_even_when_text_matches(self):
        for profile in self.profiles:
            self.artifact(profile, truncated=(0,) if profile == "awq" else ())
        self.assertEqual(quality.summarize(self.root), 0)
        summary = read(self.root / "summary.json")
        self.assertEqual(summary["models"]["awq"]["truncated"], 1)
        self.assertEqual(summary["models"]["awq"]["correct"], 99)
        self.assertEqual(summary["comparisons"]["awq"]["regressions"], 1)

    def test_artifact_tampering_and_duplicate_ids_are_detected(self):
        for profile in self.profiles:
            self.artifact(profile)
        path = self.root / "awq/evaluation/results.jsonl"
        data = rows(path)
        data[-1] = copy.deepcopy(data[0])
        write_rows(path, data)
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            quality.summarize(self.root)
        deployment = read(self.root / "awq/deployment.json")
        deployment["artifacts_sha256"]["evaluation/results.jsonl"] = digest(path)
        write(self.root / "awq/deployment.json", deployment)
        self.assertEqual(quality.summarize(self.root), 1)
        self.assertFalse(read(self.root / "summary.json")["models"]["awq"]["complete"])

    def test_plan_defaults_four_models_and_requires_baseline(self):
        with patch.object(quality, "load_snapshot", return_value=({}, [], read(ROOT / "configs/workloads.yaml"))), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(quality.main(["--snapshot", str(self.inputs), "--plan-only"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["total_requests"], 400)
        self.assertFalse(plan["generation"]["ignore_eos"])
        for command in plan["commands"]:
            self.assertIn("--no-enforce-eager", command)
            self.assertEqual(command[command.index("--max-model-len") + 1], "2048")
        with patch.object(quality, "load_snapshot", return_value=({}, [], read(ROOT / "configs/workloads.yaml"))), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            quality.main(["--snapshot", str(self.inputs), "--model", "awq", "--plan-only"])

    def test_template_context_and_frozen_generation_cannot_silently_change(self):
        tokenizer = SimpleNamespace(get_chat_template=lambda: "template",
            apply_chat_template=lambda *a, **k: "rendered prompt", encode=lambda *a, **k: [1] * 10)
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
        with patch.dict(sys.modules, {"transformers": module}):
            self.assertEqual(len(quality.prepare_inputs(self.inputs, {"path": "synthetic"}, self.tasks, 2048)), 100)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                quality.prepare_inputs(self.inputs, {"path": "synthetic"}, self.tasks, 130)
            tokenizer.get_chat_template = lambda: "changed"
            with self.assertRaisesRegex(ValueError, "template"):
                quality.prepare_inputs(self.inputs, {"path": "synthetic"}, self.tasks, 2048)
        protocol = read(self.inputs / "quality_protocol.json")
        protocol["generation"]["ignore_eos"] = True
        write(self.inputs / "quality_protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            quality.check_holdout(self.inputs)

    def test_managed_server_uses_real_http_and_existing_frozen_scorer(self):
        tasks = {t["prompt"]: t for t in self.tasks}
        payloads = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                data = {"data": [{"id": "local-model"}]} if self.path == "/v1/models" else {}
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payloads.append(payload)
                task = tasks[payload["messages"][0]["content"]]
                text = json.dumps(task["expected"]) if task["scorer"] == "json_exact" else task["expected"]
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                             "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        config = read(ROOT / "configs/bf16.yaml")
        config.update(tokenizer="/synthetic/base", port=server.server_port, enforce_eager=False)
        def launch(command, stdout, **kwargs):
            self.assertEqual(kwargs["env"]["VLLM_TEST_FORCE_FP8_MARLIN"], "0")
            self.assertTrue(kwargs["start_new_session"])
            stdout.write("Initializing a V1 LLM engine with config: dtype=torch.bfloat16, quantization=None\n")
            stdout.flush()
            return SimpleNamespace(poll=lambda: None)
        with patch.object(quality, "wait_port"), patch.object(quality.subprocess, "Popen", side_effect=launch), \
             patch.object(quality.bench, "stop_process") as stop, contextlib.redirect_stdout(io.StringIO()):
            record = quality.run_model(config, {"config": {}}, self.inputs, self.root / "bf16", 2048)
        self.assertEqual(record["status"], "complete", record)
        stop.assert_called_once()
        self.assertEqual(len(payloads), 100)
        for payload in payloads:
            self.assertEqual(payload["temperature"], 0)
            self.assertFalse(payload["ignore_eos"])
            self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
            self.assertNotIn("min_tokens", payload)
        for profile in self.profiles[1:]:
            self.artifact(profile)
        self.assertEqual(quality.summarize(self.root), 0)

    def test_startup_failure_still_cleans_up_and_saves_evidence(self):
        config = read(ROOT / "configs/bf16.yaml")
        config["tokenizer"] = "/synthetic/base"
        with patch.object(quality, "wait_port"), \
             patch.object(quality.subprocess, "Popen", return_value=SimpleNamespace(poll=lambda: 1)), \
             patch.object(quality.bench, "stop_process") as stop:
            result = quality.run_model(config, {"config": {}}, self.inputs, self.root / "bf16", 2048)
        stop.assert_called_once()
        self.assertEqual(result["status"], "failed")
        self.assertTrue((self.root / "bf16/deployment.json").exists())
        self.assertIn("Server exited", result["error"])

    def test_managed_comparison_continues_after_one_model_precheck_fails(self):
        snapshot_protocol = read(ROOT / "configs/workloads.yaml")
        def verify(config, source):
            if config["profile"] == "fp8-official":
                raise ValueError("synthetic missing official FP8 weights")
            return {"config": {}, "base_source_verification": "synthetic"}
        calls = []
        def evaluate(config, model, snapshot, directory, max_model_len):
            calls.append(config["profile"])
            original = self.artifact(config["profile"])
            shutil.copytree(original, directory)
        def command(command, **kwargs):
            if "--output" in command:
                write(Path(command[command.index("--output") + 1]), {"synthetic": True})
            return SimpleNamespace(stdout="synthetic pip freeze", returncode=0)
        output = self.root / "managed"
        requests = [dict(id=t["id"], input_tokens=7) for t in self.tasks]
        with patch.object(quality, "load_snapshot", return_value=(read(self.inputs / "manifest.json"), [], snapshot_protocol)), \
             patch.object(quality.sys, "platform", "linux"), \
             patch.object(quality.bench, "check_runtime_versions", return_value={"versions": {"vllm": "0.14.1"}}), \
             patch.object(quality.bench, "validate"), patch.object(quality.bench, "verify_model", side_effect=verify), \
             patch.object(quality, "prepare_inputs", return_value=requests), \
             patch.object(quality, "run_model", side_effect=evaluate), \
             patch.object(quality.subprocess, "run", side_effect=command), \
             contextlib.redirect_stdout(io.StringIO()):
            status = quality.main(["--snapshot", str(self.inputs), "--output", str(output)])
        self.assertEqual(status, 1)
        self.assertEqual(calls, ["bf16", "awq", "fp8-local"])
        summary = read(output / "summary.json")
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["models"]["fp8-local"]["correct"], 100)
        self.assertIsNone(summary["models"]["fp8-official"]["accuracy"])
        self.assertEqual(quality.main(["--summarize", str(output)]), 1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit), patch.object(quality, "load_snapshot",
                 return_value=(read(self.inputs / "manifest.json"), [], snapshot_protocol)), \
             patch.object(quality.sys, "platform", "linux"), \
             patch.object(quality.bench, "check_runtime_versions", return_value={"versions": {"vllm": "0.14.1"}}), \
             patch.object(quality.bench, "validate"):
            quality.main(["--snapshot", str(self.inputs), "--output", str(output)])

    def test_port_in_use_does_not_start_or_stop_an_unrelated_server(self):
        # Windows permits duplicate SO_REUSEADDR binds; simulate the Linux
        # EADDRINUSE result instead of relying on host-specific socket behavior.
        with patch.object(quality.socket, "socket") as socket_factory:
            socket_factory.return_value.__enter__.return_value.bind.side_effect = OSError("address in use")
            with self.assertRaisesRegex(TimeoutError, "busy"):
                quality.wait_port(7081, timeout=0)


if __name__ == "__main__":
    unittest.main()
