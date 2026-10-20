#!/usr/bin/env python3
"""Run local text causal language models with vLLM or Transformers.

GPU libraries are imported only after CLI/path validation. --tokenize-only
prepares requests without loading weights. --serve launches the versioned
vLLM API entry point with a small response-profiling middleware.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback


DEFAULT_PROMPT = "请用三句话解释大模型推理中的 prefill 和 decode。"
GENERATION_FLAGS = {
    "--prompt", "--prompt-file", "--messages-file", "--system", "--raw-prompt",
    "--thinking", "--max-new-tokens", "--temperature", "--top-p", "--top-k",
    "--repetition-penalty", "--output", "--tokenize-only",
}


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a finite number")
    return number


def parse_args(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run a local causal LM; defaults target a short single-GPU request.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--model", default=str(
        Path(os.environ.get("MODEL_ROOT", "/workspace/weights_pub")) / "Qwen3-8B"
    ), help="local model directory; repository IDs are not downloaded")
    parser.add_argument("--tokenizer", help="local tokenizer directory; defaults to --model")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt", action="append", help="user text; repeat for independent requests")
    inputs.add_argument("--prompt-file", type=Path, help="UTF-8 file, treated as one user prompt")
    inputs.add_argument("--messages-file", type=Path, help="JSON list of text chat messages")
    parser.add_argument("--system", help="system instruction for --prompt/--prompt-file")
    parser.add_argument("--raw-prompt", action="store_true", help="bypass the chat template (base LMs)")
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                        help="enable_thinking template switch; auto preserves the template default")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="generation only")
    parser.add_argument("--temperature", type=finite_float, default=0.7,
                        help="generation only; 0 selects greedy decoding")
    parser.add_argument("--top-p", type=finite_float, default=0.8, help="generation only")
    parser.add_argument("--top-k", type=int, default=20, help="generation only; 0 disables top-k")
    parser.add_argument("--repetition-penalty", type=finite_float, default=1.0, help="generation only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="input + output token limit; also an engine setting for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=finite_float, default=0.75,
                        help="vLLM: fraction of each GPU's total memory, not its free memory")
    parser.add_argument("--max-num-seqs", "--concurrency", type=int, default=1,
                        help="vLLM scheduler sequence limit (not the HTTP connection limit)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048,
                        help="vLLM token budget per scheduler step, including prefill")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM GPU count")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
                        help="vLLM: disable CUDA Graph capture for the initial smoke test")
    parser.add_argument("--quantization", default="auto",
                        help="vLLM quantization method, e.g. awq; auto reads model config")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Transformers only")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="allow custom Python code from the local model repository")
    parser.add_argument("--tokenize-only", action="store_true", help="inspect prompts/tokens; no weights")
    parser.add_argument("--output", type=Path, help="write an inference JSON record (overwrites this file)")
    parser.add_argument("--serve", action="store_true", help="vLLM only: start an HTTP API server")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=7081, help="HTTP port")
    parser.add_argument("--served-model-name", default="local-model", help="model name used by API clients")
    parser.add_argument("--debug", action="store_true", help="print full traceback on failure")
    args = parser.parse_args(raw)

    for name in ("max_new_tokens", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "tensor_parallel_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_new_tokens > args.max_model_len and not args.serve:
        parser.error("--max-new-tokens must not exceed --max-model-len")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("require temperature >= 0, 0 < top-p <= 1, top-k >= 0")
    if args.repetition_penalty <= 0 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("require repetition-penalty > 0 and 0 < gpu-memory-utilization <= 1")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.raw_prompt and (args.system or args.messages_file or args.thinking != "auto"):
        parser.error("--raw-prompt cannot combine with --system, --messages-file or explicit --thinking")
    if args.messages_file and args.system:
        parser.error("put the system instruction inside --messages-file")
    if args.serve:
        if args.backend != "vllm":
            parser.error("--serve requires --backend vllm")
        supplied = {argument.split("=", 1)[0] for argument in raw}
        invalid = sorted(supplied & GENERATION_FLAGS)
        if invalid:
            parser.error("--serve: send generation/template settings in the API request; remove "
                         + ", ".join(invalid))
    if args.backend == "vllm" and args.max_num_batched_tokens < args.max_num_seqs:
        parser.error("--max-num-batched-tokens must be at least --max-num-seqs")
    if args.backend == "vllm" and args.device != "cuda":
        parser.error("--device cpu is supported only by --backend transformers")
    if args.backend == "transformers" and (
        args.tensor_parallel_size != 1 or args.quantization != "auto"
    ):
        parser.error("tensor parallelism/explicit quantization in this script require vLLM")
    return args


def local_directory(value, description):
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"{description} directory does not exist: {directory}. "
                         "Pass a downloaded local path.")
    return str(directory)


def resolve_paths(args):
    args.model = local_directory(args.model, "Model")
    args.tokenizer = local_directory(args.tokenizer or args.model, "Tokenizer")
    config_path = Path(args.model) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")
    if args.backend == "transformers" and config.get("quantization_config") and not args.tokenize_only:
        raise ValueError("This Transformers entry point supports unquantized models. "
                         "Use --backend vllm for prequantized AWQ/FP8 weights.")
    if not args.tokenize_only:
        # Enumerate only model weight names; no weight data is read here.
        has_weights = any(Path(args.model).glob("*.safetensors")) or any(
            Path(args.model).glob("*.bin"))
        if not has_weights:
            raise ValueError("No .safetensors/.bin weights found in the model directory.")
    return config


def load_conversations(args):
    if args.messages_file:
        messages = json.loads(args.messages_file.read_text(encoding="utf-8"))
        if not isinstance(messages, list) or not messages:
            raise ValueError("--messages-file must contain a nonempty JSON list")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in (
                "system", "user", "assistant"
            ) or not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("Each message needs role=system/user/assistant and nonempty text content")
        if messages[-1]["role"] != "user":
            raise ValueError("The last message must have role=user")
        return [messages]
    prompts = args.prompt
    if args.prompt_file:
        prompts = [args.prompt_file.read_text(encoding="utf-8")]
    if prompts is None:
        prompts = [DEFAULT_PROMPT]
    conversations = []
    for prompt in prompts:
        if not prompt.strip():
            raise ValueError("Prompts must not be empty")
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        conversations.append(messages)
    return conversations


def prepare_requests(tokenizer, conversations, args):
    prepared = []
    if not args.raw_prompt and not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template. Use --raw-prompt for a base model.")
    if args.thinking != "auto":
        # Do not silently claim the switch is applied to an unrelated template.
        template = tokenizer.get_chat_template()
        if "enable_thinking" not in template:
            raise ValueError("This template does not declare enable_thinking; use --thinking auto.")
    for messages in conversations:
        if args.raw_prompt:
            text = messages[-1]["content"]
            token_ids = tokenizer.encode(text, add_special_tokens=True)
        else:
            kwargs = {}
            if args.thinking != "auto":
                kwargs["enable_thinking"] = args.thinking == "on"
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            # The rendered template already contains its own special tokens.
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("A prompt produced no input tokens")
        total = len(token_ids) + args.max_new_tokens
        if total > args.max_model_len:
            raise ValueError(f"Request needs {len(token_ids)} input + {args.max_new_tokens} "
                             f"output tokens, exceeding --max-model-len {args.max_model_len}. "
                             "Shorten the prompt/output or explicitly increase the limit.")
        prepared.append({"messages": messages, "prompt": text,
                         "input_token_ids": token_ids, "input_tokens": len(token_ids)})
    return prepared


def vllm_engine_kwargs(args):
    kwargs = {
        "model": args.model, "tokenizer": args.tokenizer, "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code, "seed": args.seed,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs, "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_batched_tokens": args.max_num_batched_tokens, "enable_chunked_prefill": True,
        "enforce_eager": args.enforce_eager, "enable_prefix_caching": False,
        "generation_config": "vllm",
    }
    if args.quantization != "auto":
        kwargs["quantization"] = args.quantization
    return kwargs


def sampling_kwargs(args):
    greedy = args.temperature == 0
    return {
        "max_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": 1.0 if greedy else args.top_p,
        "top_k": -1 if greedy or args.top_k == 0 else args.top_k,
        "repetition_penalty": args.repetition_penalty, "seed": args.seed,
    }


def inference_metrics(requests, generation_seconds):
    """Aggregate actual output tokens over the generation batch, excluding model load."""
    output_tokens = sum(request["output_tokens"] for request in requests)
    return {
        "request_count": len(requests),
        "input_tokens": sum(request["input_tokens"] for request in requests),
        "output_tokens": output_tokens,
        "inference_seconds": generation_seconds,
        "output_tokens_per_second": output_tokens / generation_seconds if generation_seconds > 0 else None,
        "requests_per_second": len(requests) / generation_seconds if generation_seconds > 0 else None,
        "timing_scope": "generation_batch_wall_time_excluding_model_load",
    }


def run_vllm(args, requests):
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    engine = LLM(**vllm_engine_kwargs(args))
    load_seconds = time.perf_counter() - started
    params = SamplingParams(**sampling_kwargs(args))
    started = time.perf_counter()
    outputs = engine.generate(
        [{"prompt_token_ids": request["input_token_ids"]} for request in requests],
        sampling_params=params, use_tqdm=False,
    )
    generation_seconds = time.perf_counter() - started
    if len(outputs) != len(requests):
        raise RuntimeError("vLLM returned a different number of requests")
    for request, output in zip(requests, outputs):
        completion = output.outputs[0]
        request.update({
            "generated_text": completion.text,
            "output_token_ids": list(completion.token_ids),
            "output_tokens": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
        })
    return {"load_seconds": load_seconds, "generation_seconds": generation_seconds,
            **inference_metrics(requests, generation_seconds)}


def generation_kwargs(args, tokenizer, model):
    # Fresh config prevents hidden generation_config.json settings from changing the protocol.
    kwargs = {
        "max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty, "num_beams": 1, "use_cache": True,
        "eos_token_id": model.generation_config.eos_token_id,
        "bos_token_id": model.generation_config.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    return kwargs


def trim_generated(token_ids, eos_token_id):
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return token_ids[:index + 1], "stop"
    return token_ids, "length"


def run_transformers(args, tokenizer, requests):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable. Check the active interpreter/driver, "
                           "or explicitly use --device cpu --dtype float32 for a small model.")
    set_seed(args.seed)
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": args.device},
        local_files_only=True, trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad_token or eos_token for batching")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer.pad(
        {"input_ids": [request["input_token_ids"] for request in requests]},
        padding=True, return_tensors="pt",
    ).to(model.get_input_embeddings().weight.device)
    generation_config = GenerationConfig(**generation_kwargs(args, tokenizer, model))
    cuda_device = inputs["input_ids"].device
    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=generation_config)
    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
    generation_seconds = time.perf_counter() - started
    # generate() returns [padded input | new output]; slice at the padded width.
    input_width = inputs["input_ids"].shape[1]
    if len(sequences) != len(requests):
        raise RuntimeError("Transformers returned a different number of requests")
    for request, sequence in zip(requests, sequences):
        token_ids, reason = trim_generated(
            sequence[input_width:].tolist(), generation_config.eos_token_id
        )
        request.update({
            "generated_text": tokenizer.decode(token_ids, skip_special_tokens=True),
            "output_token_ids": token_ids, "output_tokens": len(token_ids),
            "finish_reason": reason,
        })
    return {"load_seconds": load_seconds, "generation_seconds": generation_seconds,
            **inference_metrics(requests, generation_seconds)}


def server_command(args):
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    flags = {
        "--model": args.model, "--tokenizer": args.tokenizer, "--dtype": args.dtype,
        "--host": args.host, "--port": args.port, "--served-model-name": args.served_model_name,
        "--max-model-len": args.max_model_len,
        "--gpu-memory-utilization": args.gpu_memory_utilization,
        "--max-num-seqs": args.max_num_seqs, "--tensor-parallel-size": args.tensor_parallel_size,
        "--max-num-batched-tokens": args.max_num_batched_tokens,
        "--seed": args.seed, "--generation-config": "vllm",
    }
    for flag, value in flags.items():
        command.extend([flag, str(value)])
    command.extend(["--no-enable-prefix-caching", "--enable-chunked-prefill"])
    command.extend(["--middleware", "profiling_middleware.ProfilingMiddleware",
                    "--enable-force-include-usage"])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.quantization != "auto":
        command.extend(["--quantization", args.quantization])
    return command


def package_versions():
    result = {}
    for package in ("torch", "vllm", "transformers", "tokenizers", "accelerate"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        config = resolve_paths(args)
        if args.backend == "vllm" and not args.tokenize_only:
            # No torch.cuda calls in the parent before vLLM starts its workers.
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if args.serve:
            command = server_command(args)
            # Make the sibling middleware importable in the separate API process.
            scripts_path = str(Path(__file__).resolve().parent)
            python_path = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (scripts_path, python_path)))
            print("Starting:", shlex.join(command), flush=True)
            print(f"Scheduler concurrency: {args.max_num_seqs}; API responses include profiling metrics.",
                  flush=True)
            return subprocess.call(command)

        from transformers import AutoTokenizer

        conversations = load_conversations(args)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        requests = prepare_requests(tokenizer, conversations, args)
        print(f"Backend: {args.backend}; model: {args.model}; thinking: {args.thinking}", flush=True)
        print("Input tokens:", [request["input_tokens"] for request in requests], flush=True)
        timings = None
        if args.tokenize_only:
            for index, request in enumerate(requests, 1):
                print(f"\n--- Prompt {index} ---\n{request['prompt']}")
        else:
            runner = run_vllm if args.backend == "vllm" else run_transformers
            timings = runner(args, requests) if args.backend == "vllm" else runner(args, tokenizer, requests)
            for index, request in enumerate(requests, 1):
                print(f"\n--- Response {index} ---\n{request['generated_text']}")
                print(f"input_tokens={request['input_tokens']} output_tokens={request['output_tokens']} "
                      f"finish_reason={request['finish_reason']}")
            print("Inference metrics (model load reported separately):", json.dumps(timings))
        template = None if args.raw_prompt else tokenizer.get_chat_template()
        report = {
            "backend": args.backend, "model": args.model, "tokenizer": args.tokenizer,
            "tokenize_only": args.tokenize_only, "parameters": {
                key: value for key, value in vars(args).items()
                if key not in {"output", "prompt_file", "messages_file", "prompt"}
            },
            "model_type": config.get("model_type"),
            "model_config_sha256": hashlib.sha256(
                (Path(args.model) / "config.json").read_bytes()
            ).hexdigest(),
            "chat_template_sha256": None if args.raw_prompt else hashlib.sha256(
                template.encode("utf-8")
            ).hexdigest(),
            "versions": package_versions(), "timings": timings, "requests": requests,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Saved:", args.output)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        if "out of memory" in str(error).lower() or "kv cache" in str(error).lower():
            print("For a first run, shorten input/output, keep --max-num-seqs 1 and eager mode; "
                  "check other GPU processes. vLLM needs enough memory for weights + KV + workspace.",
                  file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
