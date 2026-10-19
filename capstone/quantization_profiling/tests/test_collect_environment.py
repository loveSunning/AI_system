"""Dependency-free checks for the environment collector; no GPU is required."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_environment.py"
SPEC = importlib.util.spec_from_file_location("quantization_collect_environment", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def healthy_checks(free_gib=20):
    """A GPU visible to PyTorch, independent of the physical GPU inventory."""
    return {
        "packages": {
            "status": "ok",
            "packages": {
                "torch": {"status": "ok", "version": "test"},
                "vllm": {"status": "ok", "version": "test"},
            },
        },
        "torch": {
            "status": "ok",
            "runtime": {
                "cuda_available": True,
                "devices": [{
                    "logical_index": 0,
                    "name": "NVIDIA GeForce RTX 4090 D",
                    "free_bytes": free_gib * collector.GIB,
                }],
            },
        },
        "gpu": {
            "status": "ok",
            "devices": [{"name": "NVIDIA GeForce RTX 4090 D"}],
        },
        "nvcc": {"status": "missing"},
        "nsys": {"status": "missing"},
        "ncu": {"status": "missing"},
    }


class ProbeTests(unittest.TestCase):
    def test_gpu_csv_preserves_unknown_memory_and_converts_mib(self):
        rows = collector.parse_gpu_csv(
            "0, GPU-test, NVIDIA GeForce RTX 4090 D, 123.4, 24564, N/A, 20480.5\n"
            "1, GPU-other, Other GPU, 123.4, [N/A], -1, N/A\n\n"
        )
        self.assertEqual(rows[0]["total_bytes"], 24564 * 1024 ** 2)
        self.assertEqual(rows[0]["free_bytes"], int(20480.5 * 1024 ** 2))
        self.assertIsNone(rows[0]["used_bytes"])
        self.assertIsNone(rows[1]["total_bytes"])
        self.assertIsNone(rows[1]["used_bytes"])
        self.assertIsNone(rows[1]["free_bytes"])

    def test_nonfinite_gpu_memory_is_unknown(self):
        device = collector.parse_gpu_csv(
            "0, GPU-test, NVIDIA GeForce RTX 4090 D, 123.4, NaN, inf, -inf\n"
        )[0]
        for field in ("total_bytes", "used_bytes", "free_bytes"):
            self.assertIsNone(device[field])

    def test_command_failures_are_structured(self):
        for exception, expected in (
            (FileNotFoundError(), "missing"),
            (subprocess.TimeoutExpired(["tool"], 1), "timeout"),
        ):
            with self.subTest(status=expected):
                with patch.object(collector.subprocess, "run", side_effect=exception):
                    result = collector.run_command(["tool"], 1)
                self.assertEqual(result["status"], expected)
                self.assertIn("error", result)
                self.assertIn("elapsed_seconds", result)

    def test_torch_probe_accepts_import_messages_before_marker(self):
        runtime = {"cuda_available": False, "devices": []}
        with patch.object(collector, "run_command", return_value={
            "status": "ok",
            "stdout": "Import diagnostic\n" + collector.TORCH_MARKER + json.dumps(runtime),
        }):
            result = collector.probe_torch(1)
        self.assertEqual(result["runtime"], runtime)

    def test_torch_probe_bad_json_or_timeout_remains_diagnostic(self):
        for command_result, expected in (
            ({"status": "ok", "stdout": collector.TORCH_MARKER + "{bad"}, "error"),
            ({"status": "ok", "stdout": "no marker"}, "error"),
            ({"status": "timeout", "error": "timed out"}, "timeout"),
        ):
            with self.subTest(result=command_result):
                with patch.object(collector, "run_command", return_value=command_result.copy()):
                    result = collector.probe_torch(1)
                self.assertEqual(result["status"], expected)
                self.assertNotIn("runtime", result)
                self.assertIn("error", result)

    def test_checks_run_concurrently_and_isolate_one_probe_failure(self):
        # Every probe must enter before any can finish. Serial execution breaks
        # the barrier; no throughput or wall-clock threshold is asserted.
        barrier = threading.Barrier(6)
        entered = set()
        lock = threading.Lock()

        def finish(name):
            with lock:
                entered.add(name)
            barrier.wait(timeout=5)
            if name == "nvcc":
                raise RuntimeError("isolated toolkit failure")
            return {"status": "ok", "probe": name}

        with contextlib.ExitStack() as stack:
            for name in ("packages", "gpu", "torch"):
                stack.enter_context(patch.object(
                    collector, "probe_" + name,
                    side_effect=lambda *args, name=name: finish(name),
                ))
            stack.enter_context(patch.object(
                collector, "probe_tool", side_effect=lambda name, timeout: finish(name),
            ))
            result = collector.collect_checks(timeout=1, workers=6)

        self.assertEqual(entered, {"packages", "gpu", "torch", "nvcc", "nsys", "ncu"})
        self.assertEqual(result["nvcc"]["status"], "error")
        self.assertIn("isolated toolkit failure", result["nvcc"]["error"])
        for name in entered - {"nvcc"}:
            self.assertEqual(result[name]["status"], "ok")


class AssessmentTests(unittest.TestCase):
    def assess(self, checks, min_free=20):
        return collector.assess(checks, "4090D", min_free, "Linux")

    def test_normalized_target_and_exact_memory_threshold_pass_without_nvcc(self):
        result = self.assess(healthy_checks())
        self.assertTrue(result["prechecks_passed"], result["issues"])
        self.assertFalse(self.assess(healthy_checks(19.99))["prechecks_passed"])

    def test_physical_target_hidden_from_torch_cannot_pass(self):
        checks = healthy_checks()
        checks["torch"]["runtime"].update(cuda_available=False, devices=[])
        with patch.dict(collector.os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            result = self.assess(checks)
        self.assertFalse(result["prechecks_passed"])
        self.assertTrue(any("PyTorch-visible" in issue for issue in result["issues"]))

    def test_visible_other_gpu_does_not_satisfy_hidden_physical_target(self):
        checks = healthy_checks()
        checks["torch"]["runtime"]["devices"][0]["name"] = "NVIDIA A100"
        self.assertFalse(self.assess(checks)["prechecks_passed"])

    def test_each_required_package_must_be_installed(self):
        for name in ("torch", "vllm"):
            with self.subTest(package=name):
                checks = healthy_checks()
                checks["packages"]["packages"][name] = {"status": "missing"}
                result = self.assess(checks)
                self.assertFalse(result["prechecks_passed"])
                self.assertTrue(any(name + " is unavailable" in issue for issue in result["issues"]))


class CommandLineTests(unittest.TestCase):
    def run_main(self, checks, args):
        with patch.object(collector, "collect_checks", return_value=checks), \
                patch.object(collector.platform, "system", return_value="Linux"), \
                patch.object(collector, "print_report"), \
                contextlib.redirect_stdout(io.StringIO()):
            return collector.main(args)

    def test_defaults_and_json_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "environment.json"
            code = self.run_main(healthy_checks(), ["--strict", "--output", str(output)])
            snapshot = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(snapshot["assessment"]["expected_gpu"], "4090D")
        self.assertEqual(snapshot["assessment"]["min_free_gib"], 20)
        self.assertTrue(snapshot["assessment"]["prechecks_passed"])
        self.assertEqual(snapshot["checks"]["nvcc"]["status"], "missing")
        self.assertIn("CUDA_VISIBLE_DEVICES", snapshot["system"]["environment"])

    def test_nonfinite_numeric_arguments_are_rejected_before_probing(self):
        for option in ("--timeout", "--min-free-gib"):
            for value in ("nan", "inf"):
                with self.subTest(option=option, value=value):
                    with patch.object(collector, "collect_checks") as probes, \
                            contextlib.redirect_stderr(io.StringIO()), \
                            self.assertRaises(SystemExit) as error:
                        collector.main([option, value])
                    self.assertEqual(error.exception.code, 2)
                    probes.assert_not_called()

    def test_failed_prechecks_only_change_exit_status_in_strict_mode(self):
        checks = healthy_checks(10)
        self.assertEqual(self.run_main(checks, []), 0)
        self.assertEqual(self.run_main(checks, ["--strict"]), 1)


if __name__ == "__main__":
    unittest.main()

