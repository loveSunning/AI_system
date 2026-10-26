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
            self.assertEqual(base["weight_verification"]["status"], "passed")
            self.assertEqual(base["weight_verification"]["mode"], "sha256")
            for profile in ("awq", "fp8-official", "fp8-local"):
                config = self.quantized(profile)
                model = bench.verify_model(config, self.source)
                bench.validate(config, self.protocol, model)
                self.assertEqual(model["profile"], profile)
                self.assertEqual(model["weight_verification"]["status"], "passed")
                self.assertEqual(model["weight_verification"]["mode"], "sha256")
                self.assertNotEqual(model["files"]["model.safetensors"], base["files"]["model.safetensors"])
                self.assertEqual(model["tokenizer_path"], base["tokenizer_path"])
            (self.source_path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since freeze"):
                bench.verify_model(self.config("bf16"), self.source)


    def test_metadata_inspection_skips_weight_tokenizer_hashes_and_conversion_manifest(self):
        configs = [self.config("bf16")] + [self.quantized(profile) for profile in
                   ("awq", "fp8-official", "fp8-local")]
        # These changes are deliberately unverified in metadata-only mode.
        (self.source_path / "model.safetensors").write_bytes(b"changed since freeze")
        write(self.source_path / "tokenizer.json", {"different_vocab": True})
        (Path(configs[-1]["model"]) / "quantization_manifest.json").write_text("invalid JSON", encoding="utf-8")
        with patch.object(bench, "digest", side_effect=AssertionError("metadata inspection must not hash files")), \
             patch.object(bench, "hash_model_file", side_effect=AssertionError("metadata inspection must not read model files")), \
             patch.object(bench, "read", wraps=read) as reader, contextlib.redirect_stdout(io.StringIO()):
            for config in configs:
                with self.subTest(profile=config["profile"]):
                    model = bench.verify_model(config, self.source, verify_weights=False)
                    self.assertEqual(model["files"], {})
                    self.assertEqual(model["tokenizer_sha256"], {})
                    self.assertEqual(model["weight_verification"]["status"], "skipped")
                    self.assertEqual(model["weight_verification"]["mode"], "metadata_only")
                    self.assertEqual(model["base_source_verification"],
                                     "architecture_only; weight_and_tokenizer_hashes_skipped")
                    self.assertNotIn("conversion_manifest_sha256", model)
        self.assertEqual([Path(call.args[0]).name for call in reader.call_args_list], ["config.json"] * 4)

    def test_metadata_inspection_still_requires_weights_architecture_and_valid_runtime(self):
        for profile in ("bf16", "awq", "fp8-official", "fp8-local"):
            config = self.config(profile) if profile == "bf16" else self.quantized(profile)
            directory = Path(config["model"])
            actual = read(directory / "config.json")
            write(directory / "config.json", dict(actual, hidden_size=999))
            with self.subTest(profile=profile), patch.object(bench, "digest") as hash_file:
                with self.assertRaisesRegex(ValueError, "architecture differs"):
                    bench.verify_model(config, self.source, verify_weights=False)
                hash_file.assert_not_called()
            write(directory / "config.json", actual)
            (directory / "model.safetensors").unlink()
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "No local model weights"):
                bench.verify_model(config, self.source, verify_weights=False)
        config = self.config("fp8-official")
        config["fp8_backend"] = "unsupported"
        with patch.object(bench, "digest") as hash_file, self.assertRaises(ValueError):
            bench.verify_model(config, self.source, verify_weights=False)
        hash_file.assert_not_called()

    def test_explicit_full_verification_checks_changed_weights_and_tokenizer(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = bench.verify_model(self.config("bf16"), self.source, verify_weights=True)
            self.assertEqual(model["weight_verification"]["status"], "passed")
            self.assertEqual(model["weight_verification"]["mode"], "sha256")
            (self.source_path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since freeze"):
                bench.verify_model(self.config("bf16"), self.source, verify_weights=True)
            config = self.quantized("awq")
            write(self.source_path / "tokenizer.json", {"different_vocab": True})
            with self.assertRaisesRegex(ValueError, "Reference tokenizer changed"):
                bench.verify_model(config, self.source, verify_weights=True)

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
            return SimpleNamespace(pid=12345, poll=lambda: None)
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
        self.assertEqual(record["startup"]["status"], "ready")
        self.assertEqual(record["server_pid"], 12345)
        self.assertEqual(record["startup"]["health_attempts"], 1)
        for field in ("spawn_seconds", "health_probe_seconds", "retry_sleep_seconds",
                      "health_wait_seconds", "engine_verification_seconds", "total_seconds"):
            self.assertGreaterEqual(record["startup"][field], 0)
        self.assertEqual([len(x[2]) for x in calls], [5, 50, 5, 50])
        self.assertEqual(calls[1][2], calls[3][2])
        self.assertEqual(record["artifact_directory"], "awq/" + self.case["id"])
        self.assertTrue((self.root / record["artifact_directory"] / "summary.json").exists())
        stop.assert_called_once()

    def test_startup_oom_still_saves_failure_and_cleans_owned_process(self):
        config = self.config("fp8-local")
        def launch(command, stdout, **kwargs):
            stdout.write("CUDA out of memory\n"); stdout.flush()
            return SimpleNamespace(pid=12345, poll=lambda: 1)
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
        def verify(config, source, *, verify_weights=True):
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



    def test_cli_defaults_to_metadata_and_full_verification_failure_never_passes(self):
        snapshot, _ = self.frozen()
        config_path = self.root / "bf16.json"
        write(config_path, self.config("bf16"))
        def environment(command, **kwargs):
            if "--output" in command:
                write(Path(command[command.index("--output") + 1]), {"checks": {"torch": {"runtime": {
                    "devices": [{"name": "RTX 4090D", "bf16_supported": True}]}}}})
            return SimpleNamespace(stdout="synthetic", returncode=0)
        def case_run(config, protocol, case, items, directory, model):
            return {"case": case["id"], "model_profile": config["profile"], "rounds": [], "status": "passed"}
        def safe_digest(path):
            path = Path(path)
            if path.parent == self.source_path:
                raise AssertionError("default benchmark must not hash model or tokenizer files")
            return digest(path)
        common = ["--snapshot", str(snapshot), "--config", str(config_path)]
        with patch.object(bench, "ROOT", self.root), patch.object(bench.sys, "platform", "linux"), \
             patch.object(bench, "check_runtime_versions", return_value={"versions": {"vllm": "0.14.1"}}), \
             patch.object(bench.socket, "socket"), patch.object(bench.subprocess, "run", side_effect=environment), \
             patch.object(bench, "run_case", side_effect=case_run) as runner, \
             contextlib.redirect_stdout(io.StringIO()):
            for flags in ([], ["--no-verify-weights"]):
                with self.subTest(flags=flags), patch.object(bench, "digest", side_effect=safe_digest), \
                     patch.object(bench, "hash_model_file", side_effect=AssertionError("weight hashing disabled")):
                    self.assertEqual(bench.main(common + flags), 0)
                    model = runner.call_args.args[-1]
                    self.assertEqual(model["weight_verification"]["status"], "skipped")
                    self.assertEqual(model["files"], {})
            runner.reset_mock()
            (self.source_path / "model.safetensors").write_bytes(b"changed after freeze")
            self.assertEqual(bench.main(common + ["--verify-weights"]), 1)
            runner.assert_not_called()
        runs = [read(path / "run.json") for path in (self.root / "log").iterdir()]
        skipped = [run for run in runs if not run["models"][0].get("validation_error")]
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(not run["models"][0]["config"]["verify_weights"] for run in skipped))
        failed = next(run for run in runs if run["models"][0].get("validation_error"))
        self.assertTrue(failed["models"][0]["config"]["verify_weights"])
        self.assertIn("changed since freeze", failed["models"][0]["validation_error"])
        status = (failed["models"][0].get("model") or {}).get("weight_verification", {}).get("status")
        self.assertNotEqual(status, "passed")

    def test_bf16_verified_requires_successful_sha256_verification(self):
        config = self.config("bf16")
        def launch(command, stdout, **kwargs):
            stdout.write("Initializing a V1 LLM engine with config: dtype=torch.bfloat16, quantization=None, seed=42\n")
            stdout.flush()
            return SimpleNamespace(pid=12345, poll=lambda: None)
        def phase(args, cfg, group, round_id, stage, concurrency, output):
            return {"round": round_id, "phase": stage, "successful_requests": len(group),
                    "total_requests": len(group), "failed_requests": 0}
        for iteration, (verification, expected) in enumerate(((None, False), ({"status": "skipped", "mode": "metadata_only"}, False),
                ({"status": "failed", "mode": "sha256"}, False),
                ({"status": "passed", "mode": "metadata_only"}, False),
                ({"status": "passed", "mode": "sha256"}, True))):
            model = {"config": self.base_config}
            if verification is not None:
                model["weight_verification"] = verification
            with self.subTest(verification=verification), patch.object(bench, "GPURecorder", FakeGPU), \
                 patch.object(bench.subprocess, "Popen", side_effect=launch), \
                 patch.object(bench.Client, "health", return_value={}), \
                 patch.object(bench, "phase", side_effect=phase), \
                 patch.object(bench.time, "sleep"), patch.object(bench, "stop_process"), \
                 contextlib.redirect_stdout(io.StringIO()):
                record = bench.run_case(config, dict(self.protocol, rounds=1), self.case,
                                        self.items(1), self.root / str(iteration), model)
            self.assertEqual(record.get("bf16_verified", False), expected)

    def test_eager_cli_overrides_are_applied_without_changing_configs(self):
        snapshot, _ = self.frozen()
        config_file = ROOT / "configs/fp8.yaml"
        original_hash = digest(config_file)
        for flags, expected in (([], True), (["--enforce-eager"], True),
                                (["--enforce-eager", "true"], True),
                                (["--enforce-eager", "false"], False),
                                (["--enforce-eager=false"], False),
                                (["--enforce-eager", "0"], False),
                                (["--no-enforce-eager"], False)):
            output = io.StringIO()
            with self.subTest(flags=flags), contextlib.redirect_stdout(output):
                self.assertEqual(bench.main(["--snapshot", str(snapshot), "--model", "fp8-local",
                                             "--requests", "7", "--plan-only", *flags]), 0)
            entry = json.loads(output.getvalue())["commands"][0]
            self.assertEqual(entry["enforce_eager"], expected)
            self.assertIn("--enforce-eager" if expected else "--no-enforce-eager", entry["command"])
        self.assertEqual(digest(config_file), original_hash)
        config = self.config("bf16")
        config["enforce_eager"] = False
        custom = self.root / "custom.json"
        write(custom, config)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            bench.main(["--snapshot", str(snapshot), "--config", str(custom), "--plan-only"])
        self.assertFalse(json.loads(output.getvalue())["commands"][0]["enforce_eager"])

    def test_origin_marlin_plan_records_environment_and_keeps_same_request_subset(self):
        snapshot, _ = self.frozen()
        plans = []
        for backend in ("auto", "marlin"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                bench.main(["--snapshot", str(snapshot), "--model", "fp8-origin",
                            "--fp8-backend", backend, "--enforce-eager", "false", "--plan-only"])
            plans.append(json.loads(output.getvalue()))
        self.assertEqual(plans[1]["models"], ["fp8-official"])
        self.assertEqual(plans[0]["selected_requests_sha256"], plans[1]["selected_requests_sha256"])
        for plan, backend, flag in zip(plans, ("auto", "marlin"), ("0", "1")):
            entry = plan["commands"][0]
            self.assertEqual(entry["fp8_backend_requested"], backend)
            self.assertEqual(entry["environment_overrides"], {"VLLM_TEST_FORCE_FP8_MARLIN": flag})
            self.assertEqual(entry["command"][entry["command"].index("--fp8-backend")+1], backend)
            self.assertFalse(entry["enforce_eager"])

    def test_compressed_tensors_marlin_fails_before_hashing_or_launch(self):
        config = self.quantized("fp8-local")
        config["fp8_backend"] = "marlin"
        with patch.object(bench, "digest") as hash_file, patch.object(bench.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "does not honor"):
                bench.verify_model(config, self.source)
        hash_file.assert_not_called()
        launch.assert_not_called()
        config = self.config("bf16")
        config["enforce_eager"] = "false"
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            bench.validate(config, self.protocol)

    def test_marlin_requires_initialization_evidence_but_not_claims_of_kernel_verification(self):
        config = self.quantized("fp8-official")
        config["fp8_backend"] = "marlin"
        model = {"config": read(Path(config["model"]) / "config.json")}
        startup = "Initializing a V1 LLM engine with config: dtype=torch.bfloat16, quantization=fp8, seed=42"
        with self.assertRaisesRegex(ValueError, "Marlin initialization"):
            bench.verify_engine(startup, config, model)
        result = bench.verify_engine(startup + "\nWeight-only FP8 compression will be used leveraging the Marlin kernel.", config, model)
        self.assertEqual(result["execution_precision"], "W8A16")
        self.assertTrue(result["backend_initialization_verified"])
        self.assertFalse(result["cuda_kernel_verified"])

    def test_server_environment_is_scoped_and_execution_settings_are_persisted(self):
        configs = [self.config("fp8-official"), self.config("bf16")]
        configs[0].update(fp8_backend="marlin", enforce_eager=False)
        captured = []
        def launch(command, stdout, **kwargs):
            captured.append(kwargs["env"])
            method = "fp8" if "--fp8-backend" in command else "None"
            stdout.write(f"Initializing a V1 LLM engine with config: dtype=torch.bfloat16, quantization={method}, seed=42\n")
            if method == "fp8":
                stdout.write("Weight-only FP8 compression will be used leveraging the Marlin kernel.\n")
            stdout.flush()
            return SimpleNamespace(pid=12345, poll=lambda: None)
        def phase(args, cfg, group, round_id, stage, concurrency, output):
            return {"round": round_id, "phase": stage, "successful_requests": len(group),
                    "total_requests": len(group), "failed_requests": 0}
        with patch.dict(bench.os.environ, {"VLLM_TEST_FORCE_FP8_MARLIN": "1"}), \
             patch.object(bench, "GPURecorder", FakeGPU), patch.object(bench.subprocess, "Popen", side_effect=launch), \
             patch.object(bench.Client, "health", return_value={}), patch.object(bench, "phase", side_effect=phase), \
             patch.object(bench.time, "sleep"), patch.object(bench, "stop_process"), \
             contextlib.redirect_stdout(io.StringIO()):
            for cfg in configs:
                actual = dict(self.base_config, quantization_config=NATIVE_FP8_CONFIG) if cfg["profile"] == "fp8-official" else self.base_config
                record = bench.run_case(cfg, dict(self.protocol, rounds=1), self.case, self.items(), self.root, {"config": actual})
                self.assertEqual(record["status"], "passed")
                self.assertEqual(record["enforce_eager"], cfg["enforce_eager"])
            self.assertEqual(bench.os.environ["VLLM_TEST_FORCE_FP8_MARLIN"], "1")
        self.assertEqual([env["VLLM_TEST_FORCE_FP8_MARLIN"] for env in captured], ["1", "0"])

    def test_legacy_runtime_rejects_explicit_marlin_before_starting_services(self):
        snapshot, _ = self.frozen()
        runtime = {"versions": {"vllm": "0.11.0"}}
        with patch.object(bench.sys, "platform", "linux"), \
             patch.object(bench, "check_runtime_versions", return_value=runtime), \
             patch.object(bench.subprocess, "Popen") as launch, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            bench.main(["--snapshot", str(snapshot), "--model", "fp8-origin", "--fp8-backend", "marlin"])
        self.assertEqual(error.exception.code, 2)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
