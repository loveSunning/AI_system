"""Synthetic tokenizer and tiny safetensors verify workload counts without a GPU."""
import contextlib
from collections import Counter
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_workloads as prepare
from benchmark import select_requests
from experiment_utils import ROOT, digest, load_snapshot, read, rows, write


class Tokenizer:
    def get_chat_template(self):
        return "{{ enable_thinking }} user assistant"

    def apply_chat_template(self, messages, **kwargs):
        return "BEGIN " + messages[0]["content"] + " END ASSISTANT THINK_OFF"

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


class PrepareWorkloadsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "Qwen3-8B"
        self.model.mkdir()
        write(self.model / "config.json", {"model_type": "qwen3", "num_hidden_layers": 2,
              "num_attention_heads": 2, "num_key_value_heads": 1, "hidden_size": 16})
        write(self.model / "tokenizer.json", {"synthetic": True})
        header = json.dumps({"test.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
        (self.model / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
        self.protocol = read(ROOT / "configs/workloads.yaml")
        self.protocol.update(requests=3, cases=self.protocol["cases"][:2])
        self.config = self.root / "workloads.yaml"
        write(self.config, self.protocol)
        self.output = self.root / "frozen"
        self.auto = SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer()))

    def args(self, *extra):
        return ["--model", str(self.model), "--config", str(self.config), "--output", str(self.output), *extra]

    def generate(self, *extra):
        output = io.StringIO()
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=self.auto)}), \
             contextlib.redirect_stdout(output):
            prepare.main(self.args(*extra))
        return output.getvalue()

    def test_default_uses_config_and_preserves_existing_snapshot(self):
        config_hash = digest(self.config)
        self.generate()
        _, items, frozen = load_snapshot(self.output)
        self.assertEqual(frozen, self.protocol)
        self.assertEqual(len(items), 2 * (3 + 5))
        self.assertEqual(Counter(item["phase"] for item in items), {"warmup": 10, "measure": 6})
        self.assertEqual(len(rows(self.output / "quality_holdout.jsonl")), 100)
        self.assertEqual(digest(self.config), config_hash)
        self.auto.from_pretrained.assert_called_once_with(str(self.model.resolve()), local_files_only=True)
        manifest_hash = digest(self.output / "manifest.json")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            prepare.main(self.args("--requests", "200"))
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(digest(self.output / "manifest.json"), manifest_hash)
        self.assertEqual(load_snapshot(self.output)[2]["requests"], 3)

    def test_two_hundred_per_case_is_frozen_and_replayed_by_benchmark(self):
        config_hash = digest(self.config)
        output = self.generate("--requests", "200")
        self.assertIn("200 formal + 5 warmup requests per case (410 total)", output)
        _, items, frozen = load_snapshot(self.output)
        self.assertEqual(frozen["requests"], 200)
        self.assertEqual(read(self.config)["requests"], 3)
        self.assertEqual(digest(self.config), config_hash)
        self.assertEqual(len(items), 410)
        self.assertEqual(len({item["id"] for item in items}), 410)
        for case in frozen["cases"]:
            subset = [item for item in items if item["case"] == case["id"]]
            self.assertEqual(Counter(item["phase"] for item in subset), {"warmup": 5, "measure": 200})
            self.assertTrue(all(len(item["prompt_token_ids"]) == case["input_tokens"] for item in subset))
        full, selected, _ = select_requests(items, frozen, frozen["cases"], requests=200, rounds=1)
        self.assertEqual((full["requests"], len(selected)), (200, 410))
        smaller, selected, _ = select_requests(items, frozen, frozen["cases"], requests=50, rounds=2)
        self.assertEqual((smaller["requests"], smaller["rounds"], len(selected)), (50, 2, 110))
        with self.assertRaisesRegex(ValueError, "snapshot has"):
            select_requests(items, frozen, frozen["cases"], requests=201)

    def test_invalid_cli_counts_fail_before_reading_files_or_importing_transformers(self):
        for value in ("0", "-1", "1.5", "invalid"):
            with self.subTest(value=value), patch.object(prepare, "read") as reader, \
                 patch.dict(sys.modules, {"transformers": None}), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                prepare.main(self.args("--requests", value))
            self.assertEqual(error.exception.code, 2)
            reader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_config_counts_fail_without_creating_snapshot(self):
        for value in (0, -1, 1.5, "200", True, None):
            write(self.config, dict(self.protocol, requests=value))
            with self.subTest(value=value), patch.dict(sys.modules, {"transformers": None}), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                prepare.main(self.args())
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(self.output.exists())

    def test_help_exposes_requests_without_transformers(self):
        output = io.StringIO()
        with patch.dict(sys.modules, {"transformers": None}), contextlib.redirect_stdout(output), \
             self.assertRaises(SystemExit) as error:
            prepare.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--requests REQUESTS", output.getvalue())
        self.assertIn("excluding warmup", output.getvalue())


if __name__ == "__main__":
    unittest.main()
