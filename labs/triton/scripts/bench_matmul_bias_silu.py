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
import torch.nn.functional as F
import triton

from triton_playground.ops import matmul_bias_silu, torch_matmul_bias_silu


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


def tflops(ms: float, m: int, n: int, k: int) -> float:
    return (2.0 * m * n * k) / (ms * 1e9)


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


def assert_close(torch_module, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float16:
        torch_module.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    else:
        torch_module.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def maybe_compile_expression(enable_compile: bool):
    if not enable_compile or not hasattr(torch, "compile"):
        return None

    @torch.compile
    def compiled(a, b, bias):
        return F.silu(torch.matmul(a, b) + bias)

    return compiled


def make_inputs(m: int, n: int, k: int, dtype: torch.dtype, check_correctness: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    bias = torch.randn((n,), device="cuda", dtype=dtype)

    if check_correctness:
        actual = matmul_bias_silu(a, b, bias)
        expected = torch_matmul_bias_silu(a, b, bias)
        assert_close(torch, actual, expected)
    return a, b, bias


def benchmark_provider(
    provider: str,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    compiled_expr,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    if provider == "triton_fused":
        return do_bench_ms(
            lambda: matmul_bias_silu(
                a,
                b,
                bias,
                block_size_m=args.block_size_m,
                block_size_n=args.block_size_n,
                block_size_k=args.block_size_k,
                group_size_m=args.group_size_m,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            args.warmup,
            args.iters,
        )
    if provider == "torch_expression":
        return do_bench_ms(lambda: torch_matmul_bias_silu(a, b, bias), args.warmup, args.iters)
    if provider == "torch_compile":
        if compiled_expr is None:
            raise RuntimeError("torch.compile is not available; remove torch_compile from --providers")
        return do_bench_ms(lambda: compiled_expr(a, b, bias), args.warmup, args.iters)
    raise ValueError(f"unknown provider: {provider}")


def config_text(provider: str, args: argparse.Namespace) -> str:
    if provider == "triton_fused":
        return (
            f"BLOCK_M={args.block_size_m};BLOCK_N={args.block_size_n};BLOCK_K={args.block_size_k};"
            f"GROUP_M={args.group_size_m};num_warps={args.num_warps};num_stages={args.num_stages}"
        )
    if provider == "torch_expression":
        return "torch_silu_matmul_bias"
    return "torch_compile_silu_matmul_bias"


def result_row(
    provider: str,
    m: int,
    n: int,
    k: int,
    dtype_name: str,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "stage": "W12",
        "op": "matmul_bias_silu",
        "impl": provider,
        "shape": f"{m}x{n}x{k}",
        "dtype": dtype_name,
        "config": config_text(provider, args),
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{tflops(median_ms, m, n, k):.3f}",
        "unit": "TFLOP/s",
        "reference": "torch_matmul_bias_silu",
        "device": torch.cuda.get_device_name(),
        "notes": "triton.testing.do_bench median/p20/p80; matmul FLOPs only",
    }


def run_single(args: argparse.Namespace, m: int, n: int, k: int) -> None:
    dtype = DTYPES[args.dtype]
    a, b, bias = make_inputs(m, n, k, dtype, check_correctness=not args.skip_correctness)
    compiled_expr = maybe_compile_expression("torch_compile" in args.providers and not args.no_compile)

    if compiled_expr is not None:
        for _ in range(args.compile_warmup):
            compiled_expr(a, b, bias)
        torch.cuda.synchronize()

    for provider in args.providers:
        if provider == "torch_compile" and args.no_compile:
            continue
        median_ms, low_ms, high_ms = benchmark_provider(provider, a, b, bias, compiled_expr, args)
        row = result_row(provider, m, n, k, args.dtype, median_ms, low_ms, high_ms, args)
        append_result(args.output, row)
        print(row)


def run_sweep(args: argparse.Namespace) -> None:
    for power in range(args.min_power, args.max_power + 1):
        size = 1 << power
        run_single(args, size, size, args.k)


def run_perf_report(args: argparse.Namespace) -> None:
    x_vals = [1 << power for power in range(args.min_power, args.max_power + 1)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["size"],
            x_vals=x_vals,
            x_log=True,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            styles=[("blue", "-"), ("green", "-"), ("red", "-")][: len(args.providers)],
            ylabel="TFLOP/s",
            plot_name="w12-matmul-bias-silu",
            args={
                "k": args.k,
                "dtype_name": args.dtype,
                "warmup": args.warmup,
                "iters": args.iters,
                "block_size_m": args.block_size_m,
                "block_size_n": args.block_size_n,
                "block_size_k": args.block_size_k,
                "group_size_m": args.group_size_m,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
                "skip_correctness": args.skip_correctness,
                "no_compile": args.no_compile,
                "compile_warmup": args.compile_warmup,
            },
        )
    )
    def benchmark(
        size: int,
        provider: str,
        k: int,
        dtype_name: str,
        warmup: int,
        iters: int,
        block_size_m: int,
        block_size_n: int,
        block_size_k: int,
        group_size_m: int,
        num_warps: int,
        num_stages: int,
        skip_correctness: bool,
        no_compile: bool,
        compile_warmup: int,
    ):
        local_args = argparse.Namespace(
            dtype=dtype_name,
            warmup=warmup,
            iters=iters,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            block_size_k=block_size_k,
            group_size_m=group_size_m,
            num_warps=num_warps,
            num_stages=num_stages,
            providers=args.providers,
            skip_correctness=skip_correctness,
            no_compile=no_compile,
            compile_warmup=compile_warmup,
        )
        dtype = DTYPES[dtype_name]
        a, b, bias = make_inputs(size, size, k, dtype, check_correctness=not skip_correctness)
        compiled_expr = maybe_compile_expression(provider == "torch_compile" and not no_compile)
        if compiled_expr is not None:
            for _ in range(compile_warmup):
                compiled_expr(a, b, bias)
            torch.cuda.synchronize()
        median_ms, low_ms, high_ms = benchmark_provider(provider, a, b, bias, compiled_expr, local_args)
        return tflops(median_ms, size, size, k), tflops(high_ms, size, size, k), tflops(low_ms, size, size, k)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.run(print_data=True, show_plots=args.show_plots, save_path=str(args.plot_dir))
    except TypeError as exc:
        if "save_path" not in str(exc):
            raise
        benchmark.run(print_data=True, show_plots=args.show_plots)


def parse_providers(raw: str) -> list[str]:
    providers = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"triton_fused", "torch_expression", "torch_compile"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused Triton matmul+bias+SiLU against PyTorch baselines.")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("triton_fused,torch_expression,torch_compile"))
    parser.add_argument("--block-size-m", type=int, default=128)
    parser.add_argument("--block-size-n", type=int, default=128)
    parser.add_argument("--block-size-k", type=int, default=32)
    parser.add_argument("--group-size-m", type=int, default=8)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--compile-warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w12_matmul_bias_silu.csv")
    parser.add_argument("--sweep", action="store_true", help="Run square M=N sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-power", type=int, default=10)
    parser.add_argument("--max-power", type=int, default=13)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.no_compile:
        args.providers = [provider for provider in args.providers if provider != "torch_compile"]
    if not args.providers:
        raise ValueError("at least one provider must be enabled")

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.m, args.n, args.k)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
