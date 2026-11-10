"""CPU-only checks for the W31 launch boundary (no vLLM imports)."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("w31_serve", MODULE_PATH)
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)


class ServeTests(unittest.TestCase):
    def test_build_command_keeps_paths_as_argv_and_maps_baseline_flags(self):
        config = serve.validate_config({"model": "/models/Qwen 3;literal", "enforce_eager": False})
        command = serve.build_command(config)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:3], ["-m", "vllm.entrypoints.openai.api_server"])
        self.assertEqual(command[command.index("--model") + 1], str(Path("/models/Qwen 3;literal").resolve()))
        self.assertNotIn("--middleware", command)
        self.assertNotIn("--quantization", command)
        self.assertIn("--enable-force-include-usage", command)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "4")
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "2048")
        self.assertIn("--no-enforce-eager", command)
        self.assertNotIn("--enforce-eager", command)
        self.assertNotIn("--backend", command)
        self.assertNotIn("--serve", command)

    def test_rejects_misspellings_types_and_invalid_budgets(self):
        for override in (
            {"max_num_seq": 4}, {"port": 0}, {"port": True}, {"max_model_len": -1},
            {"max_num_seqs": True}, {"enforce_eager": "true"}, {"gpu_memory_utilization": float("nan")},
            {"gpu_memory_utilization": True}, {"max_num_seqs": 8, "max_num_batched_tokens": 4},
            {"seed": 2**32}, {"model": " "}, {"dtype": "bf16"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                serve.validate_config(override)

    def test_logged_subprocess_preserves_stdout_stderr_and_records_actual_spawn(self):
        log = io.StringIO()
        output = io.StringIO()
        started = []
        command = [sys.executable, "-c", "import sys; print('stdout-marker'); print('stderr-marker', file=sys.stderr)"]
        with redirect_stdout(output):
            returncode, interrupted = serve.run_logged(command, log, on_started=lambda process: started.append(process.pid))
        self.assertEqual(returncode, 0)
        self.assertFalse(interrupted)
        self.assertEqual(len(started), 1)
        self.assertIn("stdout-marker", log.getvalue())
        self.assertIn("stderr-marker", log.getvalue())
        self.assertEqual(output.getvalue(), log.getvalue())

    def test_environment_is_offline_spawn_without_project_path_injection(self):
        with mock.patch.dict(serve.os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "fork", "HF_HUB_OFFLINE": "0", "PYTHONPATH": "existing-user-path"}):
            environment = serve.server_environment()
        self.assertEqual(environment["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["PYTHONPATH"], "existing-user-path")

    def test_version_guard_accepts_only_pinned_release(self):
        for version in ("0.14.1", "0.14.1+cu128"):
            serve.require_vllm_version(version)
        for version in (None, "0.11.0", "0.14.2", "0.14.1rc1"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "requires vLLM 0.14.1"):
                serve.require_vllm_version(version)

    def test_dry_run_records_plan_without_loading_or_starting_vllm(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with mock.patch.object(serve, "package_versions", return_value={"vllm": None}), \
                 mock.patch.object(serve, "gpu_inventory", return_value={"available": False}), \
                 mock.patch.object(serve, "run_logged") as run_logged, redirect_stdout(io.StringIO()):
                result = serve.main(["--dry-run", "--model", "qwen3-8b-awq", "--output-dir", str(output)])
            self.assertEqual(result, 0)
            run_logged.assert_not_called()
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "dry_run")
            self.assertFalse(manifest["execution_started"])
            self.assertIsNone(manifest["server_ready"])
            self.assertEqual(manifest["config"]["model"], "qwen3-8b-awq")
            self.assertEqual(manifest["config"]["served_model_name"], "qwen3-8b")
            self.assertEqual(manifest["runtime_config"]["quantization"], "awq_marlin")
            self.assertEqual(manifest["environment"]["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
            self.assertEqual(manifest["environment"]["HF_HUB_OFFLINE"], "1")
            self.assertFalse(manifest["fixed_server_settings"]["enable_prefix_caching"])
            self.assertIn("no server was started", (output / "startup.log").read_text(encoding="utf-8"))

    def test_actual_execution_rejected_before_spawn_when_version_wrong(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with mock.patch.object(serve, "package_versions", return_value={"vllm": "0.11.0"}), \
                 mock.patch.object(serve, "gpu_inventory", return_value={"available": False}), \
                 mock.patch.object(serve, "run_logged") as run_logged, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = serve.main(["--output-dir", str(output)])
            self.assertEqual(result, 1)
            run_logged.assert_not_called()
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "rejected")
            self.assertFalse(manifest["execution_started"])


if __name__ == "__main__":
    unittest.main()
