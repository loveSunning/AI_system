"""Synthetic files and fake processes verify model comparisons without CUDA."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark as bench
from experiment_utils import ROOT, digest, read, write, write_rows


FP8_CONFIG = {"quant_method": "compressed-tensors", "config_groups": {"group_0": {
    "weights": {"num_bits": 8, "type": "float", "strategy": "block"},
    "input_activations": {"num_bits": 8, "type": "float", "dynamic": True}}}}


NATIVE_FP8_CONFIG = {"quant_method": "fp8", "fmt": "e4m3",
                     "activation_scheme": "dynamic", "weight_block_size": [128, 128]}

class FakeGPU:
    def __init__(self, *args):
        self.phase = "before_load"
        self.samples = [{"phase": "loaded", "gpus": [{"uuid": "synthetic", "used_mib": 10}]}]
    def __enter__(self): return self
    def __exit__(self, *args): pass


class BenchmarkModelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = read(ROOT / "configs/workloads.yaml")
        self.case = self.protocol["cases"][0]
        self.source_path = self.root / "Qwen3-8B"
        self.source_path.mkdir()
        self.base_config = {"model_type": "qwen3", "vocab_size": 100,
                            "num_hidden_layers": 2, "hidden_size": 16}
        write(self.source_path / "config.json", self.base_config)
        write(self.source_path / "tokenizer.json", {"synthetic_vocab": True})
        write(self.source_path / "tokenizer_config.json", {"chat_template": "frozen"})
        (self.source_path / "model.safetensors").write_bytes(b"synthetic bf16")
        self.source = {"path": str(self.source_path), "config": self.base_config,
            "files": {p.name: digest(p) for p in self.source_path.iterdir()}}

    def config(self, profile):
        filename = "fp8.yaml" if profile == "fp8-local" else profile + ".yaml"
        config = read(ROOT / "configs" / filename)
        config["model"] = str(self.source_path if profile == "bf16" else self.root / profile)
        config["tokenizer"] = str(self.source_path)
        return config

    def quantized(self, profile):
        config = self.config(profile)
        directory = Path(config["model"])
        directory.mkdir()
        quant = ({"quant_method": "awq", "bits": 4, "group_size": 128} if profile == "awq" else
                 NATIVE_FP8_CONFIG if profile == "fp8-official" else FP8_CONFIG)
        write(directory / "config.json", dict(self.base_config, quantization_config=quant))
        (directory / "model.safetensors").write_bytes(b"different quantized bytes " + profile.encode())
        return config

    def items(self, count=100):
        return [dict(id=f"{self.case['id']}-{i:04d}", case=self.case["id"],
                     phase="warmup" if i < 5 else "measure",
                     input_tokens=128, output_tokens=128,
                     prompt_token_ids=[i] * 128) for i in range(count + 5)]

    def frozen(self):
        directory = self.root / "frozen"
        directory.mkdir()
        protocol = dict(self.protocol, requests=100, rounds=3, cases=[self.case])
        write(directory / "workloads.yaml", protocol)
        write_rows(directory / "requests.jsonl", self.items())
        write(directory / "model.json", self.source)
        write(directory / "manifest.json", {"files": {p.name: digest(p) for p in directory.iterdir()}})
        return directory, protocol

    def test_old_hundred_snapshot_selects_fifty_without_mutating_original(self):
        items = self.items()
        original = copy.deepcopy(items)
        frozen = dict(self.protocol, requests=100, rounds=3)
        protocol, selected, sha = bench.select_requests(items[::-1], frozen, [self.case])
        self.assertEqual((protocol["requests"], protocol["rounds"]), (50, 1))
        self.assertEqual(len(selected), 55)
        self.assertEqual(selected[-1]["id"], self.case["id"] + "-0054")
        self.assertEqual(items, original)
        self.assertEqual((frozen["requests"], frozen["rounds"]), (100, 3))
        repeated = bench.select_requests(items, frozen, [self.case], 50, 2)
        self.assertEqual(sha, repeated[2])
        self.assertEqual(repeated[0]["rounds"], 2)
        self.assertNotEqual(sha, bench.select_requests(items, frozen, [self.case], 51, 1)[2])

    def test_unavailable_requests_duplicates_and_bad_lengths_are_rejected(self):
        protocol = dict(self.protocol, requests=100)
        with self.assertRaisesRegex(ValueError, "new larger snapshot"):
            bench.select_requests(self.items(), protocol, [self.case], 101)
        duplicate = self.items(); duplicate[-1]["id"] = duplicate[-2]["id"]
        with self.assertRaisesRegex(ValueError, "count/ID"):
            bench.select_requests(duplicate, protocol, [self.case])
        changed = self.items(); changed[-1]["prompt_token_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "token length"):
            bench.select_requests(changed, protocol, [self.case])

    def test_all_model_plan_defaults_and_repeat_selection(self):
        snapshot, frozen = self.frozen()
        before = digest(snapshot / "manifest.json")
        output = io.StringIO()
        with patch.dict(bench.os.environ, {"MODEL_ROOT": str(self.root)}), contextlib.redirect_stdout(output):
            self.assertEqual(bench.main(["--snapshot", str(snapshot), "--all-models", "--plan-only"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["models"], ["bf16", "awq", "fp8-official", "fp8-local"])
        self.assertEqual((plan["rounds"], plan["formal_requests_per_round"]), (1, 50))
        self.assertEqual(len(plan["selected_ids"]), 55)
        self.assertEqual(len(plan["commands"]), 4)
        awq = plan["commands"][1]["command"]
        self.assertEqual(awq[awq.index("--quantization") + 1], "awq_marlin")
        official = plan["commands"][2]["command"]
        self.assertEqual(official[official.index("--quantization") + 1], "fp8")
        self.assertTrue(official[official.index("--model") + 1].endswith("Qwen3-8B-FP8"))
        self.assertNotIn("--quantization", plan["commands"][3]["command"])
        for entry in plan["commands"]:
            command = entry["command"]
            self.assertEqual(command[command.index("--tokenizer") + 1], str(self.source_path))
        self.assertEqual(digest(snapshot / "manifest.json"), before)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            bench.main(["--snapshot", str(snapshot), "--model", "QWEN3-8B", "--model", "Qwen3-8B-FP8",
                        "--model", "qwen3-8b-fp8-local", "--rounds", "2", "--requests", "7", "--plan-only"])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["models"], ["bf16", "fp8-official", "fp8-local"])
        self.assertEqual((plan["rounds"], len(plan["selected_ids"])), (2, 12))

    def test_custom_config_expands_alias_before_profile_inference_and_null_tokenizer(self):
        config = self.quantized("awq")
        target = self.root / "Qwen3-8B-AWQ"
        Path(config["model"]).rename(target)
        config.pop("profile")
        config.update(model="qwen3-8b-awq", quantization="auto", tokenizer=None)
        path = self.root / "custom.json"
        write(path, config)
        with patch.dict(bench.os.environ, {"MODEL_ROOT": str(self.root)}):
            result = bench.load_configs(SimpleNamespace(config=path), self.source)[0]
        self.assertEqual(result["profile"], "awq")
        self.assertEqual(result["model"], str(target))
        self.assertEqual(result["tokenizer"], str(self.source_path))

    def test_official_alias_custom_config_and_wrong_fp8_format(self):
        config = self.quantized("fp8-official")
        target = self.root / "Qwen3-8B-FP8"
        Path(config["model"]).rename(target)
        config.pop("profile")
        config.update(model="Qwen3-8B-FP8", tokenizer=None)
        path = self.root / "official.json"
        write(path, config)
        with patch.dict(bench.os.environ, {"MODEL_ROOT": str(self.root)}):
            loaded = bench.load_configs(SimpleNamespace(config=path), self.source)[0]
        self.assertEqual(loaded["profile"], "fp8-official")
        with contextlib.redirect_stdout(io.StringIO()):
            model = bench.verify_model(loaded, self.source)
        self.assertEqual(model["quant_method"], "fp8")
        self.assertEqual(model["base_source_verification"], "architecture_only; original base revision unproven")
        bench.validate(loaded, self.protocol, model)
        wrong = {"config": dict(self.base_config, quantization_config=FP8_CONFIG)}
        with self.assertRaisesRegex(ValueError, "Official FP8"):
            bench.validate(loaded, self.protocol, wrong)
        write(target / "config.json", wrong["config"])
        with self.assertRaisesRegex(ValueError, "Official FP8"):
            bench.verify_model(loaded, self.source)

    def test_bf16_changed_source_fails_but_quantized_shards_have_independent_hashes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            base = bench.verify_model(self.config("bf16"), self.source)
            for profile in ("awq", "fp8-official", "fp8-local"):
                config = self.quantized(profile)
                model = bench.verify_model(config, self.source)
                bench.validate(config, self.protocol, model)
                self.assertEqual(model["profile"], profile)
                self.assertNotEqual(model["files"]["model.safetensors"], base["files"]["model.safetensors"])
                self.assertEqual(model["tokenizer_path"], base["tokenizer_path"])
            (self.source_path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since freeze"):
                bench.verify_model(self.config("bf16"), self.source)

    def test_changed_tokenizer_or_wrong_model_architecture_is_rejected(self):
        config = self.quantized("awq")
        with contextlib.redirect_stdout(io.StringIO()):
            write(self.source_path / "tokenizer.json", {"different_vocab": True})
            with self.assertRaisesRegex(ValueError, "Reference tokenizer changed"):
                bench.verify_model(config, self.source)
            actual = read(Path(config["model"]) / "config.json")
            actual["hidden_size"] = 999
            write(Path(config["model"]) / "config.json", actual)
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                bench.verify_model(config, self.source)

    def test_fp8_conversion_manifest_checks_source_and_export_content(self):
        config = self.quantized("fp8-local")
        directory = Path(config["model"])
        manifest = {"status": "complete", "source_sha256": self.source["files"],
                    "output_sha256": {p.name: digest(p) for p in directory.iterdir()}}
        write(directory / "quantization_manifest.json", manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            model = bench.verify_model(config, self.source)
            self.assertEqual(model["base_source_verification"], "conversion_manifest_matches_frozen_source")
            manifest["status"] = "failed"
            write(directory / "quantization_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "not complete"):
                bench.verify_model(config, self.source)
            manifest["status"] = "complete"
            manifest["source_sha256"] = dict(self.source["files"], **{"config.json": "bad"})
            write(directory / "quantization_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "source differs"):
                bench.verify_model(config, self.source)
            manifest["source_sha256"] = self.source["files"]
            write(directory / "quantization_manifest.json", manifest)
            (directory / "model.safetensors").write_bytes(b"corrupted export")
            with self.assertRaisesRegex(ValueError, "differ from conversion manifest"):
                bench.verify_model(config, self.source)

    def test_resolved_engine_format_is_checked_without_claiming_cuda_kernel(self):
        for profile, method in (("bf16", "None"), ("awq", "awq_marlin"),
                                ("fp8-official", "fp8"), ("fp8-local", "compressed-tensors")):
            config = self.config(profile)
            model = {"config": self.base_config if profile == "bf16" else dict(self.base_config,
                quantization_config=({"quant_method": "awq"} if profile == "awq" else
                                     NATIVE_FP8_CONFIG if profile == "fp8-official" else FP8_CONFIG))}
            line = f"INFO Initializing a V1 LLM engine (v0.11.0) with config: dtype=torch.bfloat16, quantization={method}, seed=42"
            verified = bench.verify_engine(line, config, model)
            self.assertEqual(verified["quantization"], method)
            self.assertFalse(verified["cuda_kernel_verified"])
            with self.assertRaisesRegex(ValueError, "quantization"):
                bench.verify_engine(line.replace("quantization=" + method, "quantization=wrong"), config, model)
            with self.assertRaisesRegex(ValueError, "dtype"):
                bench.verify_engine(line.replace("torch.bfloat16", "torch.float16"), config, model)
        with self.assertRaisesRegex(ValueError, "initialization"):
            bench.verify_engine("Starting: --dtype bfloat16", self.config("bf16"), {"config": self.base_config})

    def test_managed_case_repeats_chosen_subset_and_separates_model_logs(self):
        selected = bench.select_requests(self.items(), dict(self.protocol, requests=100), [self.case], 50, 2)[1]
        config = self.config("awq")
        def launch(command, stdout, **kwargs):
            stdout.write("Initializing a V1 LLM engine with config: dtype=torch.bfloat16, quantization=awq_marlin, seed=42\n")
            stdout.flush()
            return SimpleNamespace(poll=lambda: None)
        calls = []
        def phase(args, cfg, group, round_id, stage, concurrency, output):
            calls.append((round_id, stage, [x["id"] for x in group]))
            return {"round": round_id, "phase": stage, "successful_requests": len(group),
                    "total_requests": len(group), "failed_requests": 0}
        with patch.object(bench, "GPURecorder", FakeGPU), patch.object(bench.subprocess, "Popen", side_effect=launch), \
             patch.object(bench.Client, "health", return_value={}), patch.object(bench, "phase", side_effect=phase), \
             patch.object(bench.time, "sleep"), patch.object(bench, "stop_process") as stop, \
             contextlib.redirect_stdout(io.StringIO()):
            record = bench.run_case(config, dict(self.protocol, rounds=2), self.case, selected, self.root,
                                    {"config": dict(self.base_config, quantization_config={"quant_method": "awq"})})
        self.assertEqual(record["status"], "passed")
        self.assertEqual([len(x[2]) for x in calls], [5, 50, 5, 50])
        self.assertEqual(calls[1][2], calls[3][2])
        self.assertEqual(record["artifact_directory"], "awq/" + self.case["id"])
        self.assertTrue((self.root / record["artifact_directory"] / "summary.json").exists())
        stop.assert_called_once()

    def test_startup_oom_still_saves_failure_and_cleans_owned_process(self):
        config = self.config("fp8-local")
        def launch(command, stdout, **kwargs):
            stdout.write("CUDA out of memory\n"); stdout.flush()
            return SimpleNamespace(poll=lambda: 1)
        with patch.object(bench, "GPURecorder", FakeGPU), patch.object(bench.subprocess, "Popen", side_effect=launch), \
             patch.object(bench, "stop_process") as stop:
            record = bench.run_case(config, self.protocol, self.case, [], self.root, {"config": {}})
        self.assertEqual((record["status"], record["failure_kind"]), ("failed", "oom"))
        self.assertIn("CUDA out of memory", record["error"])
        self.assertEqual(read(self.root / record["artifact_directory"] / "summary.json")["status"], "failed")
        stop.assert_called_once()

    def test_unified_runtime_accepts_complete_stack_and_rejects_mixed_versions(self):
        versions = {"torch": "2.9.1+cu128", "transformers": "4.57.3", "vllm": "0.14.1", "compressed-tensors": "0.13.0"}
        with patch.object(bench.importlib.metadata, "version", side_effect=versions.__getitem__):
            result = bench.check_runtime_versions()
        self.assertEqual(result["name"], "unified-quantization-inference")
        self.assertEqual(result["versions"]["compressed-tensors"], "0.13.0")
        for package, wrong in (("torch", "2.9.0+cu128"), ("transformers", "4.57.6"),
                               ("vllm", "0.11.2"), ("compressed-tensors", "0.12.2")):
            mixed = dict(versions, **{package: wrong})
            with self.subTest(package=package), patch.object(bench.importlib.metadata, "version", side_effect=mixed.__getitem__):
                with self.assertRaises(ValueError):
                    bench.check_runtime_versions()

    def test_bad_model_precheck_does_not_skip_remaining_models(self):
        snapshot, frozen = self.frozen()
        configs = [self.config(profile) for profile in ("bf16", "awq", "fp8-official", "fp8-local")]
        def environment(command, **kwargs):
            if "--output" in command:
                write(Path(command[command.index("--output") + 1]), {"checks": {"torch": {"runtime": {
                    "devices": [{"name": "RTX 4090D", "bf16_supported": True}]}}}})
            return SimpleNamespace(stdout="synthetic", returncode=0)
        def verify(config, source):
            if config["profile"] == "awq": raise ValueError("missing AWQ files")
            actual = self.base_config if config["profile"] == "bf16" else dict(
                self.base_config, quantization_config=NATIVE_FP8_CONFIG if config["profile"] == "fp8-official" else FP8_CONFIG)
            return {"config": actual}
        def case_run(config, protocol, case, items, directory, model):
            return {"case": case["id"], "model_profile": config["profile"], "rounds": [], "status": "passed"}
        with patch.object(bench, "ROOT", self.root), patch.object(bench.sys, "platform", "linux"), \
             patch.object(bench, "load_configs", return_value=configs), \
             patch.object(bench.importlib.metadata, "version", side_effect=lambda package: {"torch": "2.8.0+cu129", "transformers": "4.57.6", "vllm": "0.11.0+cu129"}[package]), \
             patch.object(bench.socket, "socket"), patch.object(bench.subprocess, "run", side_effect=environment), \
             patch.object(bench, "verify_model", side_effect=verify), patch.object(bench, "run_case", side_effect=case_run) as runner, \
             contextlib.redirect_stdout(io.StringIO()):
            status = bench.main(["--snapshot", str(snapshot), "--all-models"])
        self.assertEqual(status, 1)
        self.assertEqual([call.args[0]["profile"] for call in runner.call_args_list], ["bf16", "fp8-official", "fp8-local"])
        run_path = next((self.root / "log").iterdir())
        results = read(run_path / "summary.json")["cases"]
        self.assertEqual([x["status"] for x in results], ["passed", "failed", "passed", "passed"])
        self.assertEqual(results[1]["failure_kind"], "model_precheck")
        run = read(run_path / "run.json")
        self.assertEqual((run["snapshot_protocol"]["requests"], run["protocol"]["requests"]), (100, 50))
        self.assertEqual(len(run["selected_ids"]), 55)


if __name__ == "__main__":
    unittest.main()
