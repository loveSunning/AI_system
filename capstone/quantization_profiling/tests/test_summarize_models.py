"""Synthetic artifact tests; these fixtures are not GPU benchmark results."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_utils import read, rows, write, write_rows
from summarize import completion, main, request_path


class SummarizeModelsTests(unittest.TestCase):
    def fixture(self, directory, profiles=("bf16", "awq", "fp8-official", "fp8-local"), cases=("w128_o128",),
                rounds=1, requests=50, legacy=False):
        run = {"run_id": "synthetic_only", "manifest_sha256": "frozen-sha",
               "protocol": {"requests": requests, "rounds": rounds, "warmup": 5, "concurrency": 1},
               "cases": list(cases)}
        if not legacy:
            run.update(schema_version=2, models=[{"profile": profile, "config": {}, "model": {}} for profile in profiles],
                       snapshot_protocol={"requests": 100, "rounds": 3}, selected_requests_sha256="subset-sha")
        results = []
        for profile in profiles:
            for case_id in cases:
                artifact = case_id if legacy else f"{profile}/{case_id}"
                item = {"case": case_id, "status": "passed", "rounds": [
                    {"round": number, "phase": "measure", "total_requests": requests,
                     "successful_requests": requests, "failed_requests": 0,
                     "output_tokens_per_second": 50.0, "latency_p95_seconds": 2.7,
                     "time_to_first_text_seconds_p95": 0.04} for number in range(1, rounds + 1)]}
                if not legacy:
                    item.update(model_profile=profile, artifact_directory=artifact)
                results.append(item)
                write_rows(directory / artifact / "requests.jsonl", [{"case": case_id, "phase": "measure", "id": "request-0"}])
        summary = {"cases": results}
        write(directory / "run.json", run)
        write(directory / "summary.json", summary)
        return run, summary

    def render(self, directory):
        with contextlib.redirect_stdout(io.StringIO()):
            main([str(directory)])
        return (directory / "Week1_Report.md").read_text(encoding="utf-8")

    def test_four_models_one_round_fifty_requests_and_separate_export(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory)
            self.assertTrue(completion(run, summary)[0])
            report = self.render(directory)
            self.assertIn("# 多模型量化性能实验报告", report)
            self.assertIn("1 轮，每轮 50 个正式请求", report)
            self.assertIn("本次请求子集 SHA256：subset-sha", report)
            self.assertIn("本次所选模型/负载全部满足有效协议：True", report)
            self.assertIn("无法据此评估多轮之间的波动", report)
            self.assertIn("|BF16|w128_o128|passed|1|50/50|", report)
            self.assertIn("|AWQ/W4A16|w128_o128|passed|1|50/50|", report)
            self.assertIn("|官方 FP8|w128_o128|passed|1|50/50|", report)
            self.assertIn("|本地 FP8|w128_o128|passed|1|50/50|", report)
            exported = rows(directory / "performance.jsonl")
            self.assertEqual(len(exported), 4)
            self.assertEqual({row["model_profile"] for row in exported}, {"bf16", "awq", "fp8-official", "fp8-local"})
            self.assertEqual({row["run_id"] for row in exported}, {"synthetic_only"})

    def test_old_three_model_runs_still_pass_and_missing_official_result_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory, profiles=("bf16", "awq", "fp8-local"))
            self.assertTrue(completion(run, summary)[0])
            self.assertNotIn("|官方 FP8|", self.render(directory))
            run, summary = self.fixture(directory)
            summary["cases"] = [entry for entry in summary["cases"] if entry["model_profile"] != "fp8-official"]
            write(directory / "summary.json", summary)
            report = self.render(directory)
            self.assertIn("有效协议：False", report)
            self.assertIn("fp8-official/w128_o128：缺少实验结果", report)

    def test_missing_round_or_request_does_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory, rounds=2)
            summary["cases"][0]["rounds"].pop()
            self.assertFalse(completion(run, summary)[0])
            write(directory / "summary.json", summary)
            self.assertIn("有效协议：False", self.render(directory))
            run, summary = self.fixture(directory)
            summary["cases"][1]["rounds"][0]["total_requests"] = 49
            summary["cases"][1]["rounds"][0]["successful_requests"] = 49
            self.assertFalse(completion(run, summary)[0])
            summary["cases"][1]["rounds"][0].update(total_requests=50, successful_requests=49, failed_requests=1)
            self.assertFalse(completion(run, summary)[0])

    def test_selected_single_case_does_not_require_five_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory, profiles=("awq",), cases=("w1024_o512",), rounds=2)
            self.assertTrue(completion(run, summary)[0])
            report = self.render(directory)
            self.assertIn("# AWQ/W4A16 实验报告", report)
            self.assertIn("有效协议：True", report)
            self.assertNotIn("五组", report)

    def test_legacy_bf16_three_round_report_and_legacy_artifact_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory, profiles=("bf16",), rounds=3, requests=100, legacy=True)
            self.assertTrue(completion(run, summary)[0])
            report = self.render(directory)
            self.assertIn("# 第一周 BF16 实验报告", report)
            self.assertIn("3 轮，每轮 100 个正式请求", report)
            self.assertIn("|BF16|w128_o128|passed|3|100/100|", report)
            self.assertIn("有效协议：True", report)
            self.assertNotIn("请求子集", report)
            self.assertEqual(rows(directory / "performance.jsonl")[0]["model_profile"], "bf16")

    def test_failed_model_is_visible_with_successful_other_models(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory)
            summary["cases"][1].update(status="failed", rounds=[], failure_kind="oom", error="CUDA out of memory")
            write(directory / "summary.json", summary)
            report = self.render(directory)
            self.assertIn("|AWQ/W4A16|w128_o128|failed|未测|", report)
            self.assertIn("|BF16|w128_o128|passed|1|50/50|", report)
            self.assertIn("|本地 FP8|w128_o128|passed|1|50/50|", report)
            self.assertIn("有效协议：False", report)
            self.assertIn("CUDA out of memory", report)
            self.assertEqual(len(rows(directory / "performance.jsonl")), 4)

    def test_missing_or_duplicate_matrix_records_and_empty_precheck_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory)
            summary["cases"][2] = dict(summary["cases"][1])
            self.assertFalse(completion(run, summary)[0])
            write(directory / "summary.json", {"cases": [], "error": "No visible GPU"})
            report = self.render(directory)
            self.assertIn("有效协议：False", report)
            self.assertIn("No visible GPU", report)
            self.assertIn("缺少实验结果", report)
            self.assertEqual(rows(directory / "performance.jsonl"), [])
            run["cases"] = []
            self.assertFalse(completion(run, {"cases": []})[0])

    def test_artifact_directory_cannot_escape_run(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "run"
            directory.mkdir()
            for invalid in ("../outside", "/outside", "C:/outside", "bf16/../../outside", "C:outside"):
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    request_path(directory, {"case": "w128_o128", "artifact_directory": invalid})
            self.assertEqual(request_path(directory, {"case": "w128_o128", "artifact_directory": "bf16/w128_o128"}),
                             (directory / "bf16/w128_o128/requests.jsonl").resolve())



    def test_weight_verification_statuses_are_explicit_and_separate_from_request_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, summary = self.fixture(directory)
            statuses = (("passed", "sha256"), ("skipped", "metadata_only"),
                        ("failed", "sha256"), ("unknown", "sha256"))
            for descriptor, (status, mode) in zip(run["models"], statuses):
                descriptor["model"]["weight_verification"] = {"status": status, "mode": mode}
            write(directory / "run.json", run)
            report = self.render(directory)
            self.assertIn("|BF16|passed|sha256|", report)
            self.assertIn("|AWQ/W4A16|skipped|metadata_only|", report)
            self.assertIn("|官方 FP8|failed|sha256|", report)
            self.assertIn("|本地 FP8|unknown|sha256|", report)
            self.assertIn("权重、Tokenizer 哈希与转换清单未校验", report)
            self.assertIn("校验状态独立于下方请求协议完成性", report)
            self.assertTrue(completion(run, summary)[0])
            self.assertIn("有效协议：True", report)

    def test_missing_legacy_verification_or_config_request_never_implies_passed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, _ = self.fixture(directory, profiles=("bf16",))
            run["models"][0]["config"]["verify_weights"] = True
            run["models"][0]["model"]["base_source_verification"] = "exact_frozen_files"
            write(directory / "run.json", run)
            report = self.render(directory)
            self.assertIn("|BF16|unknown|未记录|", report)
            self.assertNotIn("|BF16|passed|sha256|", report)
            self.assertIn("无法据此认定权重或 Tokenizer 校验通过", report)
            self.fixture(directory, profiles=("bf16",), legacy=True)
            report = self.render(directory)
            self.assertIn("|BF16|unknown|未记录|", report)
            self.assertIn("有效协议：True", report)

    def test_precheck_failure_overrides_stale_passed_verification_and_escapes_error(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, _ = self.fixture(directory, profiles=("fp8-local",))
            run["models"][0].update(validation_error="SHA256 mismatch | bad\ncontent",
                model={"weight_verification": {"status": "passed", "mode": "sha256"}})
            write(directory / "run.json", run)
            report = self.render(directory)
            self.assertIn("|本地 FP8|failed|sha256|模型预检失败：SHA256 mismatch \\| bad content|", report)
            self.assertNotIn("|本地 FP8|passed|sha256|", report)

    def test_passed_without_sha256_mode_is_reported_as_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, _ = self.fixture(directory, profiles=("awq",))
            run["models"][0]["model"]["weight_verification"] = {"status": "passed", "mode": "metadata_only"}
            write(directory / "run.json", run)
            report = self.render(directory)
            self.assertIn("|AWQ/W4A16|unknown|metadata_only|", report)
            self.assertNotIn("|AWQ/W4A16|passed|", report)

    def test_execution_settings_distinguish_marlin_w8a16_from_auto(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run, _ = self.fixture(directory, profiles=("fp8-official",))
            run["models"][0]["config"] = {"enforce_eager": False, "fp8_backend": "marlin"}
            write(directory / "run.json", run)
            report = self.render(directory)
            self.assertIn("|官方 FP8|False|marlin|W8A16（Marlin）|", report)
            self.assertIn("实际 CUDA kernel 仍需 Nsight 确认", report)


if __name__ == "__main__":
    unittest.main()
