"""CPU-only startup instrumentation checks against a fake pinned vLLM Worker."""
from contextlib import contextmanager
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import profiling_trace as trace


class FakeNVTX:
    def __init__(self):
        self.names = []
        self.active = []

    @contextmanager
    def annotate(self, message, **kwargs):
        self.names.append(message)
        self.active.append(message)
        try:
            yield
        finally:
            self.active.pop()


class BaseWorker:
    def __init__(self, rank=2, fail=False):
        self.rank = rank
        self.calls = []
        if fail:
            raise ValueError("worker construction failed")

    def init_device(self, *args, **kwargs):
        self.calls.append(("init_device", args, kwargs))
        if kwargs.get("fail"):
            raise RuntimeError("device initialization failed")
        return "device ready"

    def load_model(self, *args, **kwargs):
        self.calls.append(("load_model", args, kwargs))
        return "weights loaded"

    def determine_available_memory(self, *args, **kwargs):
        self.calls.append(("determine_available_memory", args, kwargs))
        return 123

    def initialize_from_config(self, *args, **kwargs):
        self.calls.append(("initialize_from_config", args, kwargs))
        return "cache ready"

    def compile_or_warm_up_model(self, *args, **kwargs):
        self.calls.append(("compile_or_warm_up_model", args, kwargs))
        return "warmup ready"


class WorkerStartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="worker-startup-test-")
        self.addCleanup(temporary.cleanup)
        self.events_dir = Path(temporary.name) / "worker_events"
        self.nvtx = FakeNVTX()
        modules = {name: ModuleType(name) for name in (
            "vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_worker")}
        modules["vllm.v1.worker.gpu_worker"].Worker = BaseWorker
        for active in (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"QUANT_PROFILE_EVENT_DIR": str(self.events_dir)}),
            patch.object(trace, "require_nvtx", return_value=self.nvtx),
            patch("sys.stderr", new=io.StringIO()),
        ):
            active.start()
            self.addCleanup(active.stop)
        self.addCleanup(trace.close_logging)

    def load_worker_module(self):
        spec = importlib.util.spec_from_file_location("test_startup_worker", SCRIPTS / "profiling_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def records(self):
        trace.close_logging()
        return [json.loads(line) for line in
                (self.events_dir / f"worker_{os.getpid()}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_import_constructor_and_device_stages_preserve_base_contract(self):
        module = self.load_worker_module()
        worker = module.ProfilingWorker(rank=3)
        self.assertEqual(worker.init_device("cuda:0", check=True), "device ready")
        self.assertEqual(worker.load_model(mode="local"), "weights loaded")
        self.assertEqual(worker.determine_available_memory(), 123)
        self.assertEqual(worker.initialize_from_config("kv"), "cache ready")
        self.assertEqual(worker.compile_or_warm_up_model(), "warmup ready")
        self.assertEqual(worker.calls, [
            ("init_device", ("cuda:0",), {"check": True}),
            ("load_model", (), {"mode": "local"}),
            ("determine_available_memory", (), {}),
            ("initialize_from_config", ("kv",), {}),
            ("compile_or_warm_up_model", (), {}),
        ])
        self.assertEqual(self.nvtx.names, ["worker/import_gpu_worker", "worker/base_init",
            "worker/init_device", "worker/load_model", "worker/profile_memory",
            "worker/initialize_kv_cache", "worker/compile_and_graph_capture"])
        self.assertFalse(self.nvtx.active)
        records = self.records()
        import_start = next(x for x in records if x.get("stage") == "worker/import_gpu_worker"
                            and x["event"] == "stage_start")
        self.assertTrue(import_start["already_imported"])
        self.assertTrue(all("utc" in x and "monotonic_ns" in x and "pid" in x for x in records))
        self.assertTrue(any(x["event"] == "worker_module_entry" for x in records))
        self.assertTrue(all(x["status"] == "ok" and x["cpu_wall_seconds"] >= 0
                            for x in records if x["event"] == "stage_end"))

    def test_failed_gpu_worker_import_is_logged_and_balanced(self):
        original_import = builtins.__import__
        def fail_worker_import(name, *args, **kwargs):
            if name == "vllm.v1.worker.gpu_worker":
                raise ImportError("GPU worker module unavailable")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=fail_worker_import):
            with self.assertRaisesRegex(ImportError, "GPU worker module unavailable"):
                self.load_worker_module()
        self.assertFalse(self.nvtx.active)
        failed = next(x for x in self.records() if x.get("stage") == "worker/import_gpu_worker"
                      and x["event"] == "stage_end")
        self.assertEqual(failed["status"], "failed")

    def test_missing_nvtx_is_reported_before_gpu_worker_import(self):
        with patch.object(trace, "require_nvtx", side_effect=RuntimeError("NVTX unavailable")):
            with self.assertRaisesRegex(RuntimeError, "NVTX unavailable"):
                self.load_worker_module()
        self.assertEqual(self.nvtx.names, [])
        records = self.records()
        failed = next(x for x in records if x.get("stage") == "worker/nvtx_init"
                      and x["event"] == "stage_end")
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(any(x.get("stage") == "worker/import_gpu_worker" for x in records))

    def test_device_failure_is_logged_and_does_not_leave_open_range(self):
        module = self.load_worker_module()
        worker = module.ProfilingWorker()
        with self.assertRaisesRegex(RuntimeError, "device initialization failed"):
            worker.init_device(fail=True)
        self.assertFalse(self.nvtx.active)
        records = self.records()
        failed = next(x for x in records if x.get("stage") == "worker/init_device"
                      and x["event"] == "stage_end")
        self.assertEqual(failed["status"], "failed")

    def test_constructor_failure_is_visible_before_worker_ready(self):
        module = self.load_worker_module()
        with self.assertRaisesRegex(ValueError, "worker construction failed"):
            module.ProfilingWorker(fail=True)
        self.assertFalse(self.nvtx.active)
        records = self.records()
        failed = next(x for x in records if x.get("stage") == "worker/base_init"
                      and x["event"] == "stage_end")
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(any(x["event"] == "worker_instrumentation_ready" for x in records))


if __name__ == "__main__":
    unittest.main()
