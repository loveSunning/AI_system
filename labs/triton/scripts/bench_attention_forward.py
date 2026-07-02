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

from triton_playground.ops import torch_attention, triton_stepwise_attention


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
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


def tflops(ms: float, B: int, H: int, S: int, D: int) -> float:
    flops = 4.0 * B * H * S * S * D
    return flops / (ms * 1e-3) / 1e12


def materialized_memory_mib(B: int, H: int, S: int, dtype: torch.dtype) -> tuple[float, float]:
    elem_size = torch.empty((), device="cuda", dtype=dtype).element_size()
    scores_mib = B * H * S * S * 4 / 1024**2
    probs_mib = B * H * S * S * elem_size / 1024**2
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


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = DTYPES[args.dtype]
    torch.manual_seed(0)
    q = torch.randn((args.batch, args.heads, args.seq, args.dim), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if not args.skip_correctness:
        expected = torch_attention(q, k, v, causal=args.causal)[0]
        actual = triton_stepwise_attention(
            q,
            k,
            v,
            causal=args.causal,
            block_m=args.block_m,
            block_n=args.block_n,
            block_d=args.block_d,
            softmax_block=args.softmax_block,
        )[0]
        assert_close(actual, expected)

    return q, k, v


def benchmark_provider(
    provider: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    if provider == "triton_stepwise":
        return do_bench_ms(
            lambda: triton_stepwise_attention(
                q,
                k,
                v,
                causal=args.causal,
                block_m=args.block_m,
                block_n=args.block_n,
                block_d=args.block_d,
                softmax_block=args.softmax_block,
            )[0],
            args.warmup,
            args.iters,
        )
    if provider == "torch_attention":
        return do_bench_ms(lambda: torch_attention(q, k, v, causal=args.causal)[0], args.warmup, args.iters)
    raise ValueError(f"unknown provider: {provider}")


def config_text(provider: str, args: argparse.Namespace) -> str:
    if provider == "triton_stepwise":
        return (
            f"BLOCK_M={args.block_m};BLOCK_N={args.block_n};BLOCK_D={args.block_d};"
            f"softmax_block={args.softmax_block or 'next_power_of_2(S)'};causal={args.causal}"
        )
    return f"torch_matmul_softmax_matmul;causal={args.causal}"


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
        "op": "attention_forward",
        "impl": provider,
        "shape": f"B={args.batch};H={args.heads};S={args.seq};D={args.dim}",
        "dtype": args.dtype,
        "config": config_text(provider, args),
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{tflops(median_ms, args.batch, args.heads, args.seq, args.dim):.3f}",
        "unit": "TFLOP/s_est",
        "reference": "torch_attention",
        "device": torch.cuda.get_device_name(),
        "notes": f"materialized scores_fp32={scores_mib:.2f}MiB;probs={probs_mib:.2f}MiB",
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

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq"],
            x_vals=x_vals,
            x_log=True,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            styles=[("blue", "-"), ("green", "-")][: len(args.providers)],
            ylabel="TFLOP/s_est",
            plot_name="w14-attention-forward",
            args={
                "batch": args.batch,
                "heads": args.heads,
                "dim": args.dim,
                "dtype_name": args.dtype,
                "causal": args.causal,
                "block_m": args.block_m,
                "block_n": args.block_n,
                "block_d": args.block_d,
                "softmax_block": args.softmax_block,
                "warmup": args.warmup,
                "iters": args.iters,
                "skip_correctness": args.skip_correctness,
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
        block_m: int,
        block_n: int,
        block_d: int,
        softmax_block: int | None,
        warmup: int,
        iters: int,
        skip_correctness: bool,
    ):
        local_args = argparse.Namespace(
            batch=batch,
            heads=heads,
            seq=seq,
            dim=dim,
            dtype=dtype_name,
            causal=causal,
            block_m=block_m,
            block_n=block_n,
            block_d=block_d,
            softmax_block=softmax_block,
            warmup=warmup,
            iters=iters,
            skip_correctness=skip_correctness,
        )
        q, k, v = make_inputs(local_args)
        median_ms, low_ms, high_ms = benchmark_provider(provider, q, k, v, local_args)
        return (
            tflops(median_ms, batch, heads, seq, dim),
            tflops(high_ms, batch, heads, seq, dim),
            tflops(low_ms, batch, heads, seq, dim),
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
    allowed = {"triton_stepwise", "torch_attention"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark W14 stepwise materialized attention forward.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("triton_stepwise,torch_attention"))
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-d", type=int, default=32)
    parser.add_argument("--softmax-block", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w14_attention_forward.csv")
    parser.add_argument("--sweep", action="store_true", help="Run a power-of-two sequence length sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-seq-power", type=int, default=7)
    parser.add_argument("--max-seq-power", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.batch <= 0 or args.heads <= 0 or args.seq <= 0 or args.dim <= 0:
        raise ValueError("batch, heads, seq, and dim must be positive")
    if args.softmax_block is not None and args.softmax_block < args.seq and not args.sweep:
        raise ValueError("softmax_block must be >= seq")
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
