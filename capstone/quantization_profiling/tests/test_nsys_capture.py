"""Coverage gates distinguish graph resources from actual GPU replay."""
from contextlib import closing
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "capture_check", Path(__file__).resolve().parents[1] / "scripts/check_nsys_capture.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class CaptureCoverageTests(unittest.TestCase):
    def make_db(self, kind):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "capture.sqlite"
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript('''CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE CUDA_GRAPH_EVENTS(start INTEGER, end INTEGER);
                INSERT INTO CUDA_GRAPH_EVENTS VALUES(10, 10);
                CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);
                INSERT INTO NVTX_EVENTS VALUES(1, 2, 'worker/load_model', NULL);
                INSERT INTO NVTX_EVENTS VALUES(40, 50, 'worker/prefill', NULL);
                INSERT INTO NVTX_EVENTS VALUES(60, 70, 'worker/decode', NULL);
                INSERT INTO NVTX_EVENTS VALUES(40, 80, 'controller/measure', NULL);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(nameId INTEGER, start INTEGER, end INTEGER);
                INSERT INTO StringIds VALUES(1, 'cudaGraphLaunch_v10000');''')
            if kind in ("node", "loading_only", "startup_graph"):
                db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, graphNodeId INTEGER)")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?, ?, 1)",
                           (45 if kind == "node" else 5, 48 if kind == "node" else 8))
            if kind == "graph":
                db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER, end INTEGER)")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(45, 48)")
            if kind in ("node", "graph"):
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1, 44, 45)")
            if kind == "startup_graph":
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1, 4, 5)")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(45, 48, 0)")
        return path

    def test_resource_metadata_alone_is_rejected(self):
        result = check.inspect_capture(self.make_db("metadata"))
        self.assertFalse(result["key_data_present"])
        self.assertEqual(result["graph_resource_records"], 1)

    def test_node_replay_and_measure_ranges_are_accepted(self):
        result = check.inspect_capture(self.make_db("node"), True, True)
        self.assertTrue(result["key_data_present"])
        self.assertEqual(result["measure_gpu_records"], 1)

    def test_whole_graph_replay_does_not_require_kernel_rows(self):
        result = check.inspect_capture(self.make_db("graph"), True, True)
        self.assertTrue(result["key_data_present"])
        self.assertEqual(result["kernel_records"], 0)

    def test_loading_activity_does_not_substitute_for_measured_execution(self):
        result = check.inspect_capture(self.make_db("loading_only"), True)
        self.assertFalse(result["key_data_present"])
        self.assertTrue(any("inside controller/measure" in error for error in result["errors"]))

    def test_internal_warmup_graph_does_not_prove_measured_graph_replay(self):
        result = check.inspect_capture(self.make_db("startup_graph"), True, True)
        self.assertFalse(result["key_data_present"])
        self.assertEqual(result["graph_launch_api_records"], 1)
        self.assertEqual(result["measure_graph_launch_api_records"], 0)
        self.assertTrue(any("CUDA Graph replay" in error for error in result["errors"]))


    def test_offline_steady_range_does_not_require_loading_in_capture(self):
        path = self.make_db("node")
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE NVTX_EVENTS SET text='offline/measure' WHERE text='controller/measure'")
            db.execute("DELETE FROM NVTX_EVENTS WHERE text='worker/load_model'")
        result = check.inspect_capture(path, True, True, measure_range="offline/measure",
                                       require_load_model=False)
        self.assertTrue(result["key_data_present"])
        self.assertEqual(result["measure_range"], "offline/measure")

    def test_offline_capture_rejects_reported_cuda_event_loss(self):
        path = self.make_db("node")
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE DIAGNOSTIC_EVENT(text TEXT)")
            db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES('Not all CUDA events were collected')")
        result = check.inspect_capture(path, fail_on_incomplete_cuda=True)
        self.assertFalse(result["key_data_present"])
        self.assertTrue(any("event-loss" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
