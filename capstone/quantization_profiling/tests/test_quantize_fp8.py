"""FP8 export validation uses synthetic safetensors; no model/GPU dependencies."""
import builtins
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import quantize_fp8 as quant


def float_bytes(dtype, values):
    if dtype == "BF16":
        return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16)
                        for v in values)
    return b"".join(struct.pack("<f" if dtype == "F32" else "<e", v) for v in values)


def tensor(dtype, shape, values=None):
    count = math.prod(shape)
    if values is None:
        size = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E4M3FN": 1}[dtype]
        data = bytes(count * size)
    else:
        data = float_bytes(dtype, values)
    return dtype, shape, data


def save_tensors(directory, tensors, filename="model.safetensors"):
    header = {"__metadata__": {"format": "pt"}}
    chunks = []
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    (directory / filename).write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def fp8_scheme():
    return {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "float", "strategy": "block", "symmetric": True,
                    "dynamic": False, "block_structure": [128, 128]},
        "input_activations": {"num_bits": 8, "type": "float", "strategy": "group", "symmetric": True,
                              "dynamic": True, "group_size": 128},
        "output_activations": None,
    }


class FP8Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.output = self.root / "fp8"
        self.logs = self.root / "log"
        self.source.mkdir()
        self.source_config = {"model_type": "qwen3", "num_hidden_layers": 2, "torch_dtype": "bfloat16"}
        self.expected = [f"model.layers.{i}.{p}.weight" for i in range(2) for p in quant.PROJECTIONS]
        self.source_tensors = {n: tensor("BF16", [256, 128]) for n in self.expected}
        self.source_tensors.update({
            "model.embed_tokens.weight": tensor("BF16", [8, 128]),
            "lm_head.weight": tensor("BF16", [8, 128]),
            "model.norm.weight": tensor("BF16", [128]),
        })
        quant.write_json(self.source / "config.json", self.source_config)
        quant.write_json(self.source / "tokenizer.json", {})
        quant.write_json(self.source / "tokenizer_config.json", {"chat_template": "enable_thinking"})
        save_tensors(self.source, self.source_tensors)
        self.shapes = {n: value[1] for n, value in self.source_tensors.items()}

    def args(self, *extra):
        return ["--model", str(self.source), "--output", str(self.output),
                "--log-dir", str(self.logs), *extra]

    def exported_tensors(self, scale_dtype="F32"):
        values = deepcopy(self.source_tensors)
        for name in self.expected:
            values[name] = tensor("F8_E4M3", self.shapes[name])
            scale_name = name.removesuffix(".weight") + ".weight_scale"
            values[scale_name] = tensor(scale_dtype, [2, 1], [0.125, 1.0])
        return values

    def save_export(self, tensors=None, config=None):
        self.output.mkdir(exist_ok=True)
        if config is None:
            config = dict(self.source_config, quantization_config={
                "quant_method": "compressed-tensors", "quantization_status": "compressed",
                "ignore": ["lm_head"], "config_groups": {"group_0": fp8_scheme()},
            })
        quant.write_json(self.output / "config.json", config)
        quant.write_json(self.output / "tokenizer.json", {})
        quant.write_json(self.output / "tokenizer_config.json", {"chat_template": "enable_thinking"})
        save_tensors(self.output, tensors if tensors is not None else self.exported_tensors())

    def complete_manifest(self):
        manifest = {
            "schema_version": 1, "status": "complete", "expected_linear_weights": self.expected,
            "original_shapes": self.shapes, "output_sha256": quant.fingerprints(self.output),
        }
        quant.write_json(self.output / "quantization_manifest.json", manifest)
        return manifest

    def run_main(self, *extra):
        with redirect_stdout(io.StringIO()) as stream:
            status = quant.main(self.args(*extra))
        return status, stream.getvalue()

    def test_paths_are_separate_and_resolved(self):
        for output in (self.source, self.source / "nested", self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate"):
                quant.separate_paths(self.source, output)
        resolved = quant.parse_args(["--model", str(self.source / ".." / "source"),
                                    "--output", str(self.output)])
        self.assertEqual(resolved.model, self.source)
        quant.separate_paths(self.source, self.output)

    def test_existing_output_is_not_overwritten_even_when_empty(self):
        self.output.mkdir()
        for populated in (False, True):
            with self.subTest(populated=populated):
                if populated:
                    (self.output / "keep.txt").write_text("keep", encoding="utf-8")
                status, message = self.run_main("--dry-run")
                self.assertEqual(status, 1)
                self.assertIn("Output exists", message)
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")
        self.assertFalse(self.logs.exists())

    def test_dry_run_writes_nothing_and_never_imports_gpu_libraries(self):
        before = {str(p.relative_to(self.root)): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        original_import = builtins.__import__
        blocked = {"torch", "transformers", "llmcompressor", "compressed_tensors", "safetensors"}
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise AssertionError("Dry run imported " + name)
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            status, text = self.run_main("--dry-run")
        self.assertEqual(status, 0, text)
        plan = json.loads(text)
        self.assertEqual(plan["expected_linear_count"], 14)
        self.assertEqual(plan["export_format"], "compressed-tensors")
        self.assertIsNone(plan["calibration_dataset"])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.logs.exists())
        after = {str(p.relative_to(self.root)): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_source_requires_every_layer_and_unquantized_weights(self):
        config, catalog, expected = quant.inspect_source(self.source)
        self.assertEqual(expected, self.expected)
        self.assertEqual(len(catalog), 17)
        for field in ("quantization_config", "compression_config"):
            with self.subTest(field=field):
                quant.write_json(self.source / "config.json", dict(self.source_config, **{field: {"x": 1}}))
                with self.assertRaisesRegex(ValueError, "already quantized"):
                    quant.inspect_source(self.source)
        quant.write_json(self.source / "config.json", self.source_config)
        missing = deepcopy(self.source_tensors)
        missing.pop(self.expected[-1])
        save_tensors(self.source, missing)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            quant.inspect_source(self.source)
        quantized = deepcopy(self.source_tensors)
        quantized[self.expected[0]] = tensor("F8_E4M3", self.shapes[self.expected[0]])
        save_tensors(self.source, quantized)
        with self.assertRaisesRegex(ValueError, "nonfloating"):
            quant.inspect_source(self.source)

    def test_every_projection_and_scale_is_verified_for_three_scale_dtypes(self):
        for dtype in ("BF16", "F16", "F32"):
            with self.subTest(dtype=dtype):
                self.save_export(self.exported_tensors(dtype))
                result = quant.verify_export(self.output, self.expected, self.shapes)
                self.assertEqual(result["quantized_linear_count"], 14)
                self.assertTrue(result["config_and_tensors_verified"])
                self.assertFalse(result["target_gpu_inference_verified"])

    def test_rejects_fake_bf16_export_and_quantized_lm_head(self):
        for name, dtype in ((self.expected[-1], "BF16"), ("lm_head.weight", "F8_E4M3")):
            with self.subTest(name=name):
                tensors = self.exported_tensors()
                tensors[name] = tensor(dtype, self.shapes[name])
                self.save_export(tensors)
                with self.assertRaises(ValueError):
                    quant.verify_export(self.output, self.expected, self.shapes)

    def test_rejects_missing_projection_missing_scale_and_wrong_scale_shape(self):
        for change in ("projection", "scale", "shape"):
            with self.subTest(change=change):
                tensors = self.exported_tensors()
                scale = self.expected[-1].removesuffix(".weight") + ".weight_scale"
                if change == "projection":
                    tensors.pop(self.expected[-1])
                elif change == "scale":
                    tensors.pop(scale)
                else:
                    tensors[scale] = tensor("F32", [1, 2], [1.0, 1.0])
                self.save_export(tensors)
                with self.assertRaises(ValueError):
                    quant.verify_export(self.output, self.expected, self.shapes)

    def test_scales_must_be_positive_and_finite_in_all_storage_dtypes(self):
        for dtype in ("BF16", "F16", "F32"):
            for bad in (0.0, -0.5, float("nan"), float("inf")):
                with self.subTest(dtype=dtype, bad=bad):
                    tensors = self.exported_tensors(dtype)
                    scale = self.expected[-1].removesuffix(".weight") + ".weight_scale"
                    tensors[scale] = tensor(dtype, [2, 1], [1.0, bad])
                    self.save_export(tensors)
                    with self.assertRaisesRegex(ValueError, "Nonfinite or nonpositive"):
                        quant.verify_export(self.output, self.expected, self.shapes)

    def test_recipe_must_use_exact_block_and_dynamic_activation_semantics(self):
        good = fp8_scheme()
        quant.validate_scheme(good)
        for section, key, value in (("weights", "block_structure", [64, 128]),
                                    ("weights", "dynamic", True),
                                    ("weights", "num_bits", 4),
                                    ("input_activations", "dynamic", False),
                                    ("input_activations", "group_size", 64)):
            with self.subTest(section=section, key=key):
                bad = deepcopy(good)
                bad[section][key] = value
                with self.assertRaises(ValueError):
                    quant.validate_scheme(bad)
        self.save_export()
        config = quant.read_json(self.output / "config.json")
        config["quantization_config"]["quant_method"] = "fp8"
        quant.write_json(self.output / "config.json", config)
        with self.assertRaisesRegex(ValueError, "compressed-tensors"):
            quant.verify_export(self.output, self.expected, self.shapes)

    def test_failure_is_recorded_and_partial_output_cannot_be_retried(self):
        with patch.object(quant, "versions", return_value={p: "test" for p in quant.PACKAGES}), \
             patch.object(quant, "execute_quantization", side_effect=RuntimeError("CUDA out of memory")):
            status, text = self.run_main()
        self.assertEqual(status, 1, text)
        manifest = quant.read_json(self.output / "quantization_manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("out of memory", manifest["error"])
        self.assertEqual(set(manifest["source_sha256"]), set(quant.fingerprints(self.source)))
        self.assertTrue(list(self.logs.glob("quantize_fp8_*.log")))
        self.assertEqual(self.run_main("--verify-only")[0], 1)
        self.assertEqual(self.run_main("--dry-run")[0], 1)

    def test_interruption_is_recorded(self):
        with patch.object(quant, "versions", return_value={p: "test" for p in quant.PACKAGES}), \
             patch.object(quant, "execute_quantization", side_effect=KeyboardInterrupt):
            status, text = self.run_main()
        self.assertEqual(status, 130, text)
        self.assertEqual(quant.read_json(self.output / "quantization_manifest.json")["status"], "interrupted")

    def test_verify_only_detects_file_changes_without_loading_model(self):
        self.save_export()
        self.complete_manifest()
        with patch.object(quant, "execute_quantization", side_effect=AssertionError("must not run")):
            status, text = self.run_main("--verify-only")
            self.assertEqual(status, 0, text)
            quant.write_json(self.output / "tokenizer.json", {"modified": True})
            status, text = self.run_main("--verify-only")
        self.assertEqual(status, 1)
        self.assertIn("SHA256 mismatch", text)

    def test_verify_only_requires_hash_coverage_of_every_export_file(self):
        self.save_export()
        manifest = self.complete_manifest()
        manifest["output_sha256"].pop("model.safetensors")
        quant.write_json(self.output / "quantization_manifest.json", manifest)
        status, text = self.run_main("--verify-only")
        self.assertEqual(status, 1, "Incomplete hashes must not pass verification: " + text)

    def test_manifest_cannot_hide_a_missing_projection_from_verification(self):
        self.save_export()
        for field in ("expected_linear_weights", "original_shapes"):
            with self.subTest(field=field):
                manifest = self.complete_manifest()
                if field == "expected_linear_weights":
                    manifest[field] = self.expected[:-1]
                else:
                    manifest[field] = {n: s for n, s in self.shapes.items() if n != self.expected[-1]}
                quant.write_json(self.output / "quantization_manifest.json", manifest)
                status, text = self.run_main("--verify-only")
                self.assertEqual(status, 1, "Incomplete projection metadata must fail: " + text)

    def test_invalid_export_gets_failed_manifest_after_conversion(self):
        def bad_convert(args, logger, manifest):
            tensors = self.exported_tensors()
            tensors[self.expected[-1]] = tensor("BF16", self.shapes[self.expected[-1]])
            self.save_export(tensors)
        with patch.object(quant, "versions", return_value={p: "test" for p in quant.PACKAGES}), \
             patch.object(quant, "execute_quantization", side_effect=bad_convert):
            status, text = self.run_main()
        self.assertEqual(status, 1, text)
        manifest = quant.read_json(self.output / "quantization_manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("not saved as E4M3", manifest["error"])
        self.assertNotIn("output_sha256", manifest)

    def test_success_manifest_is_created_only_after_tensor_validation(self):
        template = "enable_thinking"
        def convert(args, logger, manifest):
            self.save_export()
            manifest["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
        tokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(get_chat_template=lambda: template))
        with patch.object(quant, "versions", return_value={p: "test" for p in quant.PACKAGES}), \
             patch.object(quant, "execute_quantization", side_effect=convert), \
             patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=tokenizer)}):
            status, text = self.run_main()
        self.assertEqual(status, 0, text)
        manifest = quant.read_json(self.output / "quantization_manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(set(manifest["output_sha256"]), set(quant.fingerprints(self.output)))
        self.assertFalse(manifest["verification"]["target_gpu_inference_verified"])


if __name__ == "__main__":
    unittest.main()
