from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

TRITON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRITON_ROOT.parents[1]
OUT_BENCHMARK_ROOT = REPO_ROOT / "out" / "triton" / "benchmarks"
PYTHON_ROOT = TRITON_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import torch
import triton

from triton_playground.ops import (
    flash_attention,
    flash_attention_v2,
    flash_attention_v2_feature_reason,
    flash_attention_v2_is_available,
    torch_flash_attention_v2_reference,
)


DTYPES = {
    "float16": torch.float16,
}

QUANTILES = [0.5, 0.2, 0.8]


def do_bench_ms(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    median_ms, low_ms, high_ms = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=iters,
        quantiles=QUANTILES,
    )
    return float(median_ms), float(low_ms), float(high_ms)


def attention_tflops(ms: float, batch: int, heads: int, seq: int, dim: int, causal: bool, mode: str) -> float:
    flops_per_matmul = 2.0 * batch * heads * seq * seq * dim
    total_flops = 2.0 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "backward":
        total_flops *= 2.5
    return total_flops / (ms * 1e-3) / 1e12


def materialized_memory_mib(batch: int, heads: int, seq: int, dtype: torch.dtype) -> tuple[float, float]:
    elem_size = torch.empty((), device="cuda", dtype=dtype).element_size()
    scores_mib = batch * heads * seq * seq * 4 / 1024**2
    probs_mib = batch * heads * seq * seq * elem_size / 1024**2
    return scores_mib, probs_mib


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "op",
        "impl",
        "shape",
        "dtype",
        "config",
        "warmup",
        "iters",
        "avg_ms",
        "min_ms",
        "max_ms",
        "throughput",
        "unit",
        "reference",
        "device",
        "notes",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def assert_attention_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float = 3e-2, atol: float = 3e-2) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    requires_grad = args.mode == "backward"
    shape = (args.batch, args.heads, args.seq, args.dim)
    q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=requires_grad)

    if not args.skip_correctness:
        if args.mode == "forward":
            expected = torch_flash_attention_v2_reference(q, k, v, causal=args.causal)
            actual = flash_attention_v2(q, k, v, causal=args.causal, warp_specialize=args.warp_specialize)
            assert_attention_close(actual, expected)
        else:
            q_ref = q.detach().clone().requires_grad_(True)
            k_ref = k.detach().clone().requires_grad_(True)
            v_ref = v.detach().clone().requires_grad_(True)
            q_tri = q.detach().clone().requires_grad_(True)
            k_tri = k.detach().clone().requires_grad_(True)
            v_tri = v.detach().clone().requires_grad_(True)
            dout = torch.randn_like(q_ref)
            torch_flash_attention_v2_reference(q_ref, k_ref, v_ref, causal=args.causal).backward(dout)
            flash_attention_v2(q_tri, k_tri, v_tri, causal=args.causal, warp_specialize=args.warp_specialize).backward(dout)
            torch.testing.assert_close(q_tri.grad.float(), q_ref.grad.float(), rtol=6e-2, atol=6e-2)
            torch.testing.assert_close(k_tri.grad.float(), k_ref.grad.float(), rtol=6e-2, atol=6e-2)
            torch.testing.assert_close(v_tri.grad.float(), v_ref.grad.float(), rtol=6e-2, atol=6e-2)

    return q, k, v


def benchmark_provider(
    provider: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    if args.mode == "backward":
        dout = torch.randn_like(q)

        def bench_backward(fn):
            q.grad = None
            k.grad = None
            v.grad = None
            fn().backward(dout)

        if provider == "flash_attention_v2":
            return do_bench_ms(
                lambda: bench_backward(
                    lambda: flash_attention_v2(q, k, v, causal=args.causal, warp_specialize=args.warp_specialize)
                ),
                args.warmup,
                args.iters,
            )
        if provider == "flash_attention_v1":
            return do_bench_ms(
                lambda: bench_backward(
                    lambda: flash_attention(
                        q,
                        k,
                        v,
                        causal=args.causal,
                        block_m=args.v1_block_m,
                        block_n=args.v1_block_n,
                        block_d=args.dim,
                    )
                ),
                args.warmup,
                args.iters,
            )
        if provider == "torch_attention":
            return do_bench_ms(
                lambda: bench_backward(lambda: torch_flash_attention_v2_reference(q, k, v, causal=args.causal)),
                args.warmup,
                args.iters,
            )
        raise ValueError(f"provider {provider} does not support backward mode")

    if provider == "flash_attention_v2":
        return do_bench_ms(
            lambda: flash_attention_v2(q, k, v, causal=args.causal, warp_specialize=args.warp_specialize),
            args.warmup,
            args.iters,
        )
    if provider == "flash_attention_v1":
        return do_bench_ms(
            lambda: flash_attention(
                q,
                k,
                v,
                causal=args.causal,
                block_m=args.v1_block_m,
                block_n=args.v1_block_n,
                block_d=args.dim,
            ),
            args.warmup,
            args.iters,
        )
    if provider == "torch_attention":
        return do_bench_ms(lambda: torch_flash_attention_v2_reference(q, k, v, causal=args.causal), args.warmup, args.iters)
    raise ValueError(f"unknown provider: {provider}")


def config_text(provider: str, args: argparse.Namespace) -> str:
    if provider == "flash_attention_v2":
        return (
            "official_tutorial_v2_style;"
            f"autotune=BLOCK_M/BLOCK_N;exp2;stage_causal;warp_specialize={args.warp_specialize}"
        )
    if provider == "flash_attention_v1":
        return f"BLOCK_M={args.v1_block_m};BLOCK_N={args.v1_block_n};BLOCK_D={args.dim};online_softmax_v1"
    return "torch_matmul_softmax_matmul"


def notes_text(provider: str, scores_mib: float, probs_mib: float) -> str:
    if provider == "flash_attention_v2":
        return "Triton tutorial 06 style FlashAttention v2; no scores/probs materialization; no dropout path"
    if provider == "flash_attention_v1":
        return "project FlashAttention-1 style learning implementation"
    return f"materialized scores_fp32={scores_mib:.2f}MiB;probs={probs_mib:.2f}MiB"


def result_row(
    provider: str,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    args: argparse.Namespace,
    scores_mib: float,
    probs_mib: float,
) -> dict[str, object]:
    return {
        "stage": "W14",
        "op": "flash_attention_v2",
        "impl": provider,
        "shape": f"B={args.batch};H={args.heads};S={args.seq};D={args.dim}",
        "dtype": args.dtype,
        "config": f"{config_text(provider, args)};causal={args.causal};mode={args.mode}",
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{attention_tflops(median_ms, args.batch, args.heads, args.seq, args.dim, args.causal, args.mode):.3f}",
        "unit": "TFLOP/s_est",
        "reference": "torch_attention",
        "device": torch.cuda.get_device_name(),
        "notes": notes_text(provider, scores_mib, probs_mib),
    }


def run_single(args: argparse.Namespace) -> None:
    q, k, v = make_inputs(args)
    scores_mib, probs_mib = materialized_memory_mib(args.batch, args.heads, args.seq, DTYPES[args.dtype])
    for provider in args.providers:
        median_ms, low_ms, high_ms = benchmark_provider(provider, q, k, v, args)
        row = result_row(provider, median_ms, low_ms, high_ms, args, scores_mib, probs_mib)
        append_result(args.output, row)
        print(row)


def run_sweep(args: argparse.Namespace) -> None:
    original_seq = args.seq
    for power in range(args.min_seq_power, args.max_seq_power + 1):
        args.seq = 1 << power
        run_single(args)
    args.seq = original_seq


def run_perf_report(args: argparse.Namespace) -> None:
    x_vals = [1 << power for power in range(args.min_seq_power, args.max_seq_power + 1)]
    styles = [("red", "-"), ("blue", "-"), ("green", "-")]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq"],
            x_vals=x_vals,
            x_log=True,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            styles=styles[: len(args.providers)],
            ylabel="TFLOP/s_est",
            plot_name=f"w14-flash-attention-v2-{args.mode}-causal={args.causal}",
            args={
                "batch": args.batch,
                "heads": args.heads,
                "dim": args.dim,
                "dtype_name": args.dtype,
                "causal": args.causal,
                "mode": args.mode,
                "warmup": args.warmup,
                "iters": args.iters,
                "skip_correctness": args.skip_correctness,
                "warp_specialize": args.warp_specialize,
                "v1_block_m": args.v1_block_m,
                "v1_block_n": args.v1_block_n,
                "seed": args.seed,
            },
        )
    )
    def benchmark(
        seq: int,
        provider: str,
        batch: int,
        heads: int,
        dim: int,
        dtype_name: str,
        causal: bool,
        mode: str,
        warmup: int,
        iters: int,
        skip_correctness: bool,
        warp_specialize: bool,
        v1_block_m: int,
        v1_block_n: int,
        seed: int,
    ):
        local_args = argparse.Namespace(
            batch=batch,
            heads=heads,
            seq=seq,
            dim=dim,
            dtype=dtype_name,
            causal=causal,
            mode=mode,
            warmup=warmup,
            iters=iters,
            skip_correctness=skip_correctness,
            warp_specialize=warp_specialize,
            v1_block_m=v1_block_m,
            v1_block_n=v1_block_n,
            seed=seed,
        )
        q, k, v = make_inputs(local_args)
        median_ms, low_ms, high_ms = benchmark_provider(provider, q, k, v, local_args)
        return (
            attention_tflops(median_ms, batch, heads, seq, dim, causal, mode),
            attention_tflops(high_ms, batch, heads, seq, dim, causal, mode),
            attention_tflops(low_ms, batch, heads, seq, dim, causal, mode),
        )

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.run(print_data=True, show_plots=args.show_plots, save_path=str(args.plot_dir))
    except TypeError as exc:
        if "save_path" not in str(exc):
            raise
        benchmark.run(print_data=True, show_plots=args.show_plots)


def parse_providers(raw: str) -> list[str]:
    providers = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"flash_attention_v2", "flash_attention_v1", "torch_attention"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the tutorial-style FlashAttention v2 implementation.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, choices=[16, 32, 64, 128, 256], default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--mode", choices=["forward", "backward"], default="forward")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("flash_attention_v2,flash_attention_v1,torch_attention"))
    parser.add_argument("--warp-specialize", action="store_true")
    parser.add_argument("--v1-block-m", type=int, default=16)
    parser.add_argument("--v1-block-n", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w14_flash_attention_v2.csv")
    parser.add_argument("--sweep", action="store_true", help="Run a power-of-two sequence length sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-seq-power", type=int, default=7)
    parser.add_argument("--max-seq-power", type=int, default=11)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not flash_attention_v2_is_available():
        raise RuntimeError(flash_attention_v2_feature_reason())
    if args.batch <= 0 or args.heads <= 0 or args.seq <= 0:
        raise ValueError("batch, heads, and seq must be positive")
    if args.seq < 128 or args.seq % 128 != 0:
        raise ValueError("seq must be a multiple of 128 for flash_attention_v2")
    if args.v1_block_m <= 0 or args.v1_block_n <= 0:
        raise ValueError("v1 block sizes must be positive")
    if args.mode == "backward":
        args.providers = [provider for provider in args.providers if provider in {"flash_attention_v2", "flash_attention_v1", "torch_attention"}]
    if not args.providers:
        raise ValueError("at least one provider must be enabled")

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
