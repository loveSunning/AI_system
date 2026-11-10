"""Checks local checkpoint boundaries without loading a tokenizer or GPU."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_config import prepare_model, resolve_model_alias, resolve_runtime_options


AWQ = {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True, "version": "GEMM"}


def checkpoint(path, quant=None):
    path.mkdir(parents=True)
    metadata = {"model_type": "qwen3", "torch_dtype": "bfloat16"}
    if quant is not None:
        metadata["quantization_config"] = quant
    (path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "tokenizer_config.json").write_text('{"chat_template": "test-template"}', encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"test file inventory, not model tensors")
    return path


def config(model, **overrides):
    return {"model": str(model), "dtype": "bfloat16", "quantization": "auto", **overrides}


class ModelConfigTests(unittest.TestCase):
    def test_aliases_use_this_projects_model_root(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"MODEL_ROOT": directory}):
            self.assertEqual(resolve_model_alias("qwen3-8b"), (Path(directory) / "Qwen3-8B").resolve())
            self.assertEqual(resolve_model_alias("qwen3-8b-awq"), (Path(directory) / "Qwen3-8B-AWQ").resolve())
            with self.assertRaisesRegex(ValueError, "FP8"):
                resolve_model_alias("qwen3-8b-fp8")

    def test_dry_run_allows_missing_weights_but_execution_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            resolved, metadata = prepare_model(config(path))
            self.assertEqual(resolved["tokenizer"], str(path.resolve()))
            self.assertEqual(metadata["validation"], "planning_only")
            with self.assertRaisesRegex(ValueError, "directory does not exist"):
                prepare_model(config(path), require_files=True)

    def test_local_unquantized_checkpoint_records_hashes_and_rejects_quantizing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "model")
            resolved, metadata = prepare_model(config(path), require_files=True)
            self.assertEqual(resolved["dtype"], "bfloat16")
            self.assertEqual(resolved["quantization"], "auto")
            self.assertEqual(metadata["profile"], "unquantized")
            self.assertEqual(len(metadata["config_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "does not quantize"):
                prepare_model(config(path, quantization="awq_marlin"), require_files=True)

    def test_awq_auto_uses_marlin_and_original_awq_needs_fp16(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "model", AWQ)
            resolved, metadata = prepare_model(config(path), require_files=True)
            self.assertEqual(resolved["quantization"], "awq_marlin")
            self.assertEqual(metadata["profile"], "awq")
            with self.assertRaisesRegex(ValueError, "float16 only"):
                prepare_model(config(path, quantization="awq"))
            resolved, _ = prepare_model(config(path, quantization="awq", dtype="auto"))
            self.assertEqual(resolved["dtype"], "float16")

    def test_rejects_wrong_alias_and_unsupported_or_partial_quant_metadata(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"MODEL_ROOT": directory}):
            checkpoint((Path(directory) / "Qwen3-8B-AWQ").resolve())
            with self.assertRaisesRegex(ValueError, "expects awq"):
                prepare_model(config("qwen3-8b-awq"))
            for number, quant in enumerate(({"quant_method": "fp8"}, {**AWQ, "bits": 8}, {**AWQ, "zero_point": 1}, {**AWQ, "version": "GEMV"}, {"quant_method": "awq"})):
                with self.subTest(quant=quant):
                    path = checkpoint(Path(directory) / str(number), quant)
                    with self.assertRaises(ValueError):
                        prepare_model(config(path))

    def test_separate_tokenizer_and_missing_index_shard_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "model")
            other = checkpoint(Path(directory) / "tokenizer")
            resolved, _ = prepare_model(config(path, tokenizer=str(other)), require_files=True)
            self.assertEqual(resolved["tokenizer"], str(other.resolve()))
            (other / "tokenizer_config.json").unlink()
            with self.assertRaisesRegex(ValueError, "Tokenizer requires"):
                prepare_model(config(path, tokenizer=str(other)), require_files=True)
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing or invalid model shard"):
                prepare_model(config(path), require_files=True)

    def test_dtype_option_checks_do_not_import_gpu_libraries(self):
        self.assertEqual(resolve_runtime_options("awq", "auto", "auto"), ("bfloat16", "awq_marlin"))
        with self.assertRaisesRegex(ValueError, "float16 or bfloat16"):
            resolve_runtime_options("awq", "float32", "auto")


if __name__ == "__main__":
    unittest.main()
