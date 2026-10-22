"""Format declarations are checked before importing CUDA runtimes."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_profiles import (canonical_profile, inspect_model_config, resolve_model_alias,
                            resolve_runtime_options, profile_from_model_name, validate_model_profile,
                            validate_fp8_backend, fp8_backend_environment)


def fp8_config():
    return {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"group_0": {
        "weights": {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]},
        "input_activations": {"num_bits": 8, "type": "float", "dynamic": True, "group_size": 128},
    }}}}


class ModelProfileTests(unittest.TestCase):
    def test_aliases_are_case_insensitive_and_paths_are_preserved(self):
        with patch.dict(os.environ, {"MODEL_ROOT": "/custom/weights"}):
            self.assertEqual(canonical_profile("Qwen3-8B-AWQ"), "awq")
            self.assertEqual(resolve_model_alias("QWEN3-8B-FP8-LOCAL"), str(Path("/custom/weights") / "Qwen3-8B-FP8-local"))
            self.assertEqual(resolve_model_alias("/other/model-awq"), "/other/model-awq")
            self.assertEqual(resolve_model_alias("Qwen/Qwen3-8B"), "Qwen/Qwen3-8B")
        with self.assertRaises(ValueError):
            canonical_profile("not-a-profile")

    def test_official_fp8_alias_and_recipe_are_distinct_from_local_fp8(self):
        with patch.dict(os.environ, {"MODEL_ROOT": "/custom/weights"}):
            self.assertEqual(canonical_profile("QWEN3-8B-FP8"), "fp8-official")
            self.assertEqual(resolve_model_alias("fp8-official"), str(Path("/custom/weights") / "Qwen3-8B-FP8"))
            self.assertEqual(profile_from_model_name("/models/Qwen3-8B-FP8"), "fp8-official")
            self.assertIsNone(profile_from_model_name("/models/other-checkpoint"))
            self.assertEqual(canonical_profile("fp8"), "fp8-local")  # Historical alias.
        native = {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                  "activation_scheme": "dynamic", "weight_block_size": [128, 128]}}
        self.assertEqual(validate_model_profile(native, "fp8-official"), "fp8-official")
        self.assertEqual(validate_model_profile(native, "fp8-local"), "fp8-local")
        self.assertEqual(resolve_runtime_options(native, quantization="fp8")["quantization"], "fp8")
        with self.assertRaisesRegex(ValueError, "Official FP8"):
            validate_model_profile(fp8_config(), "fp8-official")
        for field, value in (("activation_scheme", "static"), ("weight_block_size", [64, 128]), ("fmt", "e5m2")):
            bad = copy.deepcopy(native)
            bad["quantization_config"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "FP8"):
                validate_model_profile(bad, "fp8-official")

    def test_fp8_metadata_preserves_loader_format(self):
        config = fp8_config()
        self.assertEqual(inspect_model_config(config), "fp8-local")
        options = resolve_runtime_options(config)
        self.assertEqual(options["quant_method"], "compressed-tensors")
        self.assertEqual(options["quantization"], "auto")
        self.assertEqual(options["dtype"], "bfloat16")
        with self.assertRaises(ValueError):
            resolve_runtime_options(config, quantization="fp8")
        native = {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}
        self.assertEqual(resolve_runtime_options(native, quantization="fp8")["quantization"], "fp8")

    def test_int4_or_static_recipes_are_not_mislabeled_fp8(self):
        for field, value in (("num_bits", 4), ("type", "int")):
            config = fp8_config()
            config["quantization_config"]["config_groups"]["group_0"]["weights"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "FP8"):
                inspect_model_config(config)
        config = fp8_config()
        config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["dynamic"] = False
        with self.assertRaisesRegex(ValueError, "dynamic FP8"):
            inspect_model_config(config)
        mixed = fp8_config()
        bad_group = copy.deepcopy(mixed["quantization_config"]["config_groups"]["group_0"])
        bad_group["weights"]["num_bits"] = 4
        mixed["quantization_config"]["config_groups"]["group_1"] = bad_group
        with self.assertRaises(ValueError):
            inspect_model_config(mixed)

    def test_unsupported_metadata_and_loader_mismatches_are_explicit(self):
        for config in (
            {"quantization_config": {"quant_method": "gptq"}},
            {"quantization_config": "awq"},
            {"quantization_config": {"bits": 4}},
            {"quantization_config": {"quant_method": "compressed-tensors"}},
            {"quantization_config": {"quant_method": "awq", "bits": 8}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                inspect_model_config(config)
        with self.assertRaisesRegex(ValueError, "Unquantized"):
            resolve_runtime_options({}, quantization="awq")

    def test_awq_dtype_matches_selected_kernel(self):
        config = {"quantization_config": {"quant_method": "awq", "bits": 4}}
        marlin = resolve_runtime_options(config)
        self.assertEqual((marlin["dtype"], marlin["quantization"]), ("bfloat16", "awq_marlin"))
        original = resolve_runtime_options(config, quantization="awq")
        self.assertEqual((original["dtype"], original["quantization"]), ("float16", "awq"))
        with self.assertRaisesRegex(ValueError, "float16 only"):
            resolve_runtime_options(config, dtype="bfloat16", quantization="awq")
        with self.assertRaises(ValueError):
            resolve_runtime_options(config, dtype="float32")


    def test_fp8_origin_alias_and_native_marlin_contract(self):
        self.assertEqual(canonical_profile("FP8-ORIGIN"), "fp8-official")
        self.assertTrue(resolve_model_alias("fp8-origin").endswith("Qwen3-8B-FP8"))
        native = {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                  "activation_scheme": "dynamic", "weight_block_size": [128, 128]}}
        validate_fp8_backend(native, "marlin")
        validate_fp8_backend(fp8_config(), "auto")
        with self.assertRaisesRegex(ValueError, "does not honor"):
            validate_fp8_backend(fp8_config(), "marlin")
        with self.assertRaisesRegex(ValueError, "requires an FP8"):
            validate_fp8_backend({}, "marlin")

    def test_backend_environment_is_isolated_and_rejects_conflicting_modes(self):
        original = {"VLLM_TEST_FORCE_FP8_MARLIN": "1", "CUDA_VISIBLE_DEVICES": "2"}
        auto = fp8_backend_environment("auto", original)
        marlin = fp8_backend_environment("marlin", original)
        self.assertEqual(auto["VLLM_TEST_FORCE_FP8_MARLIN"], "0")
        self.assertEqual(marlin["VLLM_TEST_FORCE_FP8_MARLIN"], "1")
        self.assertEqual(auto["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(original["VLLM_TEST_FORCE_FP8_MARLIN"], "1")
        for env, message in (({"VLLM_BATCH_INVARIANT": "1"}, "BATCH_INVARIANT"),
                             ({"VLLM_MARLIN_INPUT_DTYPE": "fp8"}, "W8A16")):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, message):
                fp8_backend_environment("marlin", env)


if __name__ == "__main__":
    unittest.main()
