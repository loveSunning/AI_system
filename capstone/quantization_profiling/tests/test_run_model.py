"""Offline request/protocol checks; fake engines never read weights or use CUDA."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_model.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("quantization_run_model", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeTokenizer:
    chat_template = "{{ enable_thinking }} user assistant"
    pad_token_id = 0
    eos_token_id = 99
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self):
        self.render_calls = []
        self.encode_calls = []

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, **kwargs):
        self.render_calls.append(kwargs)
        return "<bos>" + "|".join(message["content"] for message in messages) + "<assistant>"

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append(add_special_tokens)
        return ([5] if add_special_tokens else []) + list(range(10, 10 + len(text)))

    def pad(self, batch, **kwargs):
        rows = batch["input_ids"]
        width = max(map(len, rows))
        tensor = FakeTensor([[0] * (width - len(row)) + row for row in rows])
        mask = FakeTensor([[0] * (width - len(row)) + [1] * len(row) for row in rows])
        return FakeBatch(input_ids=tensor, attention_mask=mask)

    def decode(self, tokens, **kwargs):
        return " ".join(f"T{token}" for token in tokens if token not in (0, 99))


class FakeVector:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        return FakeVector(self.values[item])

    def tolist(self):
        return self.values


class FakeTensor:
    device = types.SimpleNamespace(type="cpu")

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(FakeVector(row) for row in self.rows)


class FakeBatch(dict):
    def to(self, device):
        return self


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="model-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_dir = self.root / "local model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(
            '{"model_type": "qwen3"}', encoding="utf-8"
        )
        (self.model_dir / "model.safetensors").write_bytes(b"fake weights")
        self.tokenizer = FakeTokenizer()

    def args(self, *extra):
        return runner.parse_args(["--model", str(self.model_dir), *extra])

    def invoke(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main(["--model", str(self.model_dir), *extra])

    def fake_transformers(self):
        module = types.ModuleType("transformers")
        module.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=self.tokenizer)
        )
        module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=MagicMock())
        module.GenerationConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
        module.set_seed = MagicMock()
        return module

    def assert_cli_error(self, *args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.args(*args)
        self.assertEqual(error.exception.code, 2)

    def test_help_does_not_import_gpu_libraries(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend", result.stdout)

    def test_invalid_numbers_and_conflicting_modes_are_rejected(self):
        for arguments in (
            ("--temperature", "nan"), ("--gpu-memory-utilization", "inf"),
            ("--top-p", "0"), ("--max-new-tokens", "0"), ("--seed", "-1"),
            ("--max-model-len", "100"), ("--serve", "--thinking", "off"),
            ("--serve", "--backend", "transformers"), ("--serve", "--prompt=test"),
            ("--raw-prompt", "--thinking", "off"),
            ("--backend", "vllm", "--device", "cpu"),
            ("--max-num-batched-tokens", "0"),
            ("--max-num-seqs", "5", "--max-num-batched-tokens", "4"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_cli_error(*arguments)

    def test_missing_directory_and_incomplete_download_are_reported(self):
        args = self.args()
        args.model = str(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "directory does not exist"):
            runner.resolve_paths(args)
        (self.model_dir / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "weights"):
            runner.resolve_paths(self.args())
        runner.resolve_paths(self.args("--tokenize-only"))

    def test_quantized_transformers_model_is_rejected_before_import(self):
        (self.model_dir / "config.json").write_text(
            '{"quantization_config": {"quant_method": "awq"}}', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "prequantized"):
            runner.resolve_paths(self.args("--backend", "transformers"))
        runner.resolve_paths(self.args("--backend", "transformers", "--tokenize-only"))

    def test_awq_offline_and_server_choose_same_bf16_marlin_loader(self):
        (self.model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "quantization_config": {
                "quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True,
            },
        }), encoding="utf-8")
        args = self.args()
        runner.resolve_paths(args)
        self.assertEqual(args.model_profile, "awq")
        self.assertEqual((args.dtype, args.quantization), ("bfloat16", "awq_marlin"))
        offline = runner.vllm_engine_kwargs(args)
        command = runner.server_command(args)
        self.assertEqual(offline["quantization"], command[command.index("--quantization") + 1])
        self.assertEqual(offline["dtype"], command[command.index("--dtype") + 1])
        args = self.args("--quantization", "awq")
        runner.resolve_paths(args)
        self.assertEqual(args.dtype, "float16")
        with self.assertRaisesRegex(ValueError, "float16 only"):
            runner.resolve_paths(self.args("--quantization", "awq", "--dtype", "bfloat16"))

    def test_compressed_fp8_never_forces_the_native_fp8_loader(self):
        metadata = {"quant_method": "compressed-tensors", "config_groups": {"group_0": {
            "weights": {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]},
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
        }}}
        (self.model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "quantization_config": metadata,
        }), encoding="utf-8")
        args = self.args()
        runner.resolve_paths(args)
        self.assertEqual(args.model_profile, "fp8-local")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertNotIn("quantization", runner.vllm_engine_kwargs(args))
        self.assertNotIn("--quantization", runner.server_command(args))
        with self.assertRaisesRegex(ValueError, "compressed-tensors"):
            runner.resolve_paths(self.args("--quantization", "fp8"))
        output = self.root / "fp8-tokens.json"
        with patch.dict(sys.modules, {"transformers": self.fake_transformers()}):
            self.assertEqual(self.invoke("--tokenize-only", "--output", str(output)), 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(record["model_profile"], "fp8-local")
        self.assertEqual(record["quantization_config"], metadata)

    def test_all_three_aliases_start_the_downloaded_local_model(self):
        for alias, dirname, quant in (
            ("qwen3-8b", "Qwen3-8B", None),
            ("qwen3-8b-awq", "Qwen3-8B-AWQ", {"quant_method": "awq", "bits": 4}),
            ("qwen3-8b-fp8-local", "Qwen3-8B-FP8-local", {"quant_method": "fp8", "fmt": "e4m3"}),
        ):
            with self.subTest(alias=alias):
                directory = self.root / dirname
                directory.mkdir()
                (directory / "config.json").write_text(json.dumps({"model_type": "qwen3", "quantization_config": quant}), encoding="utf-8")
                (directory / "model.safetensors").write_bytes(b"fake")
                with patch.dict(runner.os.environ, {"MODEL_ROOT": str(self.root)}), \
                     patch.object(runner.subprocess, "call", return_value=0) as call, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["--model", alias, "--serve"]), 0)
                command = call.call_args.args[0]
                self.assertEqual(command[command.index("--model") + 1], str(directory.resolve()))
                self.assertEqual(command[command.index("--tokenizer") + 1], str(directory.resolve()))

    def test_repeated_prompts_are_independent_with_same_system(self):
        conversations = runner.load_conversations(self.args(
            "--prompt", "one", "--prompt", "two", "--system", "teacher"
        ))
        self.assertEqual(len(conversations), 2)
        self.assertEqual([item[-1]["content"] for item in conversations], ["one", "two"])
        self.assertEqual(conversations[0][0], {"role": "system", "content": "teacher"})

    def test_text_file_is_one_prompt_and_messages_validate_last_role(self):
        source = self.root / "prompt.txt"
        source.write_text("line one\nline two", encoding="utf-8")
        conversation = runner.load_conversations(self.args("--prompt-file", str(source)))
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0][-1]["content"], "line one\nline two")
        source.write_text('[{"role":"assistant","content":"hello"}]', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "last message"):
            runner.load_conversations(self.args("--messages-file", str(source)))
        source.write_text('[{"role":"user","content":"hello"}]', encoding="utf-8")
        self.assertEqual(len(runner.load_conversations(self.args("--messages-file", str(source)))), 1)

    def test_template_switch_and_no_duplicate_special_tokens(self):
        args = self.args("--thinking", "off", "--prompt", "hello")
        requests = runner.prepare_requests(self.tokenizer, runner.load_conversations(args), args)
        self.assertFalse(self.tokenizer.render_calls[0]["enable_thinking"])
        self.assertFalse(self.tokenizer.encode_calls[0])
        self.assertEqual(requests[0]["input_tokens"], len(requests[0]["input_token_ids"]))

    def test_unknown_thinking_switch_and_missing_chat_template_are_not_silent(self):
        self.tokenizer.chat_template = "no variable here"
        args = self.args("--thinking", "off")
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            runner.prepare_requests(self.tokenizer, runner.load_conversations(args), args)
        self.tokenizer.chat_template = None
        args = self.args()
        with self.assertRaisesRegex(ValueError, "no chat template"):
            runner.prepare_requests(self.tokenizer, runner.load_conversations(args), args)

    def test_raw_prompt_and_total_token_budget(self):
        args = self.args("--raw-prompt", "--prompt", "abc", "--max-model-len", "8",
                         "--max-new-tokens", "4")
        requests = runner.prepare_requests(self.tokenizer, runner.load_conversations(args), args)
        self.assertEqual(requests[0]["input_tokens"], 4)
        self.assertEqual(requests[0]["prompt"], "abc")
        args.max_new_tokens = 5
        with self.assertRaisesRegex(ValueError, "exceeding"):
            runner.prepare_requests(self.tokenizer, runner.load_conversations(args), args)

    def test_greedy_normalizes_sampling_and_preserves_eos(self):
        args = self.args("--temperature", "0")
        sample = runner.sampling_kwargs(args)
        self.assertEqual((sample["temperature"], sample["top_p"], sample["top_k"]), (0, 1, -1))
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(
            eos_token_id=[99, 98], bos_token_id=5
        ))
        params = runner.generation_kwargs(args, self.tokenizer, model)
        self.assertFalse(params["do_sample"])
        self.assertNotIn("temperature", params)
        self.assertEqual(params["eos_token_id"], [99, 98])

    def test_tokenize_only_saves_record_without_loading_either_engine(self):
        module = self.fake_transformers()
        output = self.root / "records" / "tokens.json"
        with patch.dict(sys.modules, {"transformers": module}), \
             patch.object(runner, "run_vllm") as vllm_run, \
             patch.object(runner, "run_transformers") as transformers_run:
            status = self.invoke("--tokenize-only", "--thinking", "off", "--output", str(output))
        self.assertEqual(status, 0)
        vllm_run.assert_not_called()
        transformers_run.assert_not_called()
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(record["tokenize_only"])
        self.assertIsNone(record["timings"])
        self.assertNotIn("generated_text", record["requests"][0])
        self.assertEqual(len(record["model_config_sha256"]), 64)

    def test_vllm_uses_exact_token_ids_and_records_only_completion(self):
        transformers = self.fake_transformers()
        module = types.ModuleType("vllm")
        module.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
        engine = MagicMock()
        engine.generate.return_value = [
            types.SimpleNamespace(outputs=[types.SimpleNamespace(
                text="answer", token_ids=[31, 32], finish_reason="length"
            )])
        ]
        module.LLM = MagicMock(return_value=engine)
        output = self.root / "vllm.json"
        with patch.dict(sys.modules, {"transformers": transformers, "vllm": module}), \
             patch.dict(runner.os.environ, {}, clear=False):
            status = self.invoke("--thinking", "off", "--output", str(output))
        self.assertEqual(status, 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(record["requests"][0]["generated_text"], "answer")
        self.assertEqual(record["requests"][0]["output_tokens"], 2)
        metrics = record["timings"]
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 2 / metrics["generation_seconds"])
        self.assertEqual(engine.generate.call_args.args[0][0]["prompt_token_ids"],
                         record["requests"][0]["input_token_ids"])
        self.assertFalse(module.LLM.call_args.kwargs["enable_prefix_caching"])
        self.assertTrue(module.LLM.call_args.kwargs["enable_chunked_prefill"])
        self.assertEqual(module.LLM.call_args.kwargs["max_num_batched_tokens"], 2048)
        self.assertEqual(module.LLM.call_args.kwargs["generation_config"], "vllm")

    def test_transformers_cuts_at_padded_width_and_removes_post_eos_padding(self):
        transformers = self.fake_transformers()
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.inference_mode = contextlib.nullcontext
        model = MagicMock()
        model.generation_config = types.SimpleNamespace(eos_token_id=99, bos_token_id=5)
        model.get_input_embeddings.return_value.weight.device = FakeTensor.device

        def generate(input_ids, attention_mask, generation_config):
            return FakeTensor([
                input_ids.rows[0] + [31, 99, 0],
                input_ids.rows[1] + [41, 42, 43],
            ])

        model.generate.side_effect = generate
        transformers.AutoModelForCausalLM.from_pretrained.return_value = model
        output = self.root / "transformers.json"
        with patch.dict(sys.modules, {"transformers": transformers, "torch": torch}):
            status = self.invoke("--backend", "transformers", "--device", "cpu",
                                 "--prompt", "a", "--prompt", "a longer question",
                                 "--output", str(output))
        self.assertEqual(status, 0)
        records = json.loads(output.read_text(encoding="utf-8"))["requests"]
        self.assertEqual(records[0]["output_token_ids"], [31, 99])
        self.assertEqual(records[0]["generated_text"], "T31")
        self.assertEqual(records[0]["finish_reason"], "stop")
        self.assertEqual(records[1]["output_token_ids"], [41, 42, 43])
        self.assertEqual(records[1]["finish_reason"], "length")
        self.assertEqual(self.tokenizer.padding_side, "left")

    def test_generation_throughput_uses_actual_output_and_excludes_loading(self):
        metrics = runner.inference_metrics([
            {"input_tokens": 10, "output_tokens": 3},
            {"input_tokens": 20, "output_tokens": 7},
        ], 2.0)
        self.assertEqual(metrics["output_tokens_per_second"], 5.0)
        self.assertEqual(metrics["requests_per_second"], 1.0)
        self.assertEqual(metrics["input_tokens"], 30)
        self.assertEqual(metrics["inference_seconds"], 2.0)

    def test_server_concurrency_alias_is_passed_to_vllm_scheduler(self):
        args = self.args("--serve", "--concurrency", "4")
        command = runner.server_command(args)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "4")
        self.assertEqual(command[command.index("--middleware") + 1],
                         "profiling_middleware.ProfilingMiddleware")
        self.assertIn("--enable-force-include-usage", command)

    def test_default_server_bind_address_and_port(self):
        args = self.args("--serve")
        self.assertEqual((args.host, args.port), ("0.0.0.0", 7081))
        command = runner.server_command(args)
        self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")
        self.assertEqual(command[command.index("--port") + 1], "7081")

    def test_server_uses_active_interpreter_and_propagates_exit_code(self):
        with patch.object(runner.subprocess, "call", return_value=7) as call, \
             patch.dict(runner.os.environ, {}, clear=False):
            status = self.invoke("--serve", "--port", "8123", "--served-model-name", "test-model")
        self.assertEqual(status, 7)
        command = call.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "vllm.entrypoints.openai.api_server"])
        self.assertIn(str(self.model_dir.resolve()), command)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("8123", command)
        self.assertIn("test-model", command)
        self.assertNotIn("--thinking", command)
        self.assertEqual(call.call_args.kwargs, {})


if __name__ == "__main__":
    unittest.main()
