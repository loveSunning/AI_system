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

from triton_playground.kernels.grouped_gemm import default_num_sms
from triton_playground.ops import grouped_gemm


QUANTILES = [0.5, 0.2, 0.8]


def do_bench_ms(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    median_ms, low_ms, high_ms = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=iters,
        quantiles=QUANTILES,
    )
    return float(median_ms), float(low_ms), float(high_ms)


def tflops(ms: float, shapes: list[tuple[int, int, int]]) -> float:
    flops = sum(2.0 * m * n * k for m, n, k in shapes)
    return flops / (ms * 1e9)


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


def make_shapes(args: argparse.Namespace, group_size: int) -> list[tuple[int, int, int]]:
    shapes = []
    for index in range(group_size):
        if args.pattern == "same":
            m, n, k = args.m, args.n, args.k
        elif args.pattern == "vary_n":
            m, n, k = args.m, args.n + index * args.step, args.k
        elif args.pattern == "vary_m":
            m, n, k = args.m + index * args.step, args.n, args.k
        else:
            delta = index * args.step
            m, n, k = args.m + delta, args.n + delta, args.k + delta
        shapes.append((m, n, k))
    return shapes


def make_inputs(shapes: list[tuple[int, int, int]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    group_a = []
    group_b = []
    for m, n, k in shapes:
        group_a.append(torch.randn((m, k), device="cuda", dtype=torch.float16))
        group_b.append(torch.randn((k, n), device="cuda", dtype=torch.float16))
    expected = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]
    actual = grouped_gemm(group_a, group_b)
    for actual_i, expected_i in zip(actual, expected):
        torch.testing.assert_close(actual_i, expected_i, rtol=1e-2, atol=1e-1)
    return group_a, group_b


def benchmark_provider(
    provider: str,
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    if provider == "triton":
        return do_bench_ms(
            lambda: grouped_gemm(
                group_a,
                group_b,
                block_size_m=args.block_size_m,
                block_size_n=args.block_size_n,
                block_size_k=args.block_size_k,
                num_sms=args.num_sms,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            args.warmup,
            args.iters,
        )
    if provider == "torch_loop":
        return do_bench_ms(lambda: [torch.matmul(a, b) for a, b in zip(group_a, group_b)], args.warmup, args.iters)
    raise ValueError(f"unknown provider: {provider}")


def shape_text(shapes: list[tuple[int, int, int]]) -> str:
    if len(shapes) <= 4:
        return ",".join(f"{m}x{n}x{k}" for m, n, k in shapes)
    first = ",".join(f"{m}x{n}x{k}" for m, n, k in shapes[:2])
    last = ",".join(f"{m}x{n}x{k}" for m, n, k in shapes[-2:])
    return f"{first},...,{last}"


def config_text(provider: str, args: argparse.Namespace) -> str:
    if provider == "triton":
        return (
            f"BLOCK_M={args.block_size_m};BLOCK_N={args.block_size_n};BLOCK_K={args.block_size_k};"
            f"NUM_SM={args.num_sms};num_warps={args.num_warps};num_stages={args.num_stages}"
        )
    return "python_loop(torch.matmul)"


def result_row(
    provider: str,
    shapes: list[tuple[int, int, int]],
    warmup: int,
    iters: int,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    notes = {
        "triton": "one grouped GEMM launch; benchmark includes metadata tensor construction",
        "torch_loop": "Python loop over torch.matmul baselines",
    }[provider]
    return {
        "stage": "W11",
        "op": "grouped_gemm",
        "impl": provider,
        "shape": shape_text(shapes),
        "dtype": "float16",
        "config": config_text(provider, args),
        "warmup": warmup,
        "iters": iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{tflops(median_ms, shapes):.3f}",
        "unit": "TFLOP/s",
        "reference": "torch_loop",
        "device": torch.cuda.get_device_name(),
        "notes": notes,
    }


def run_single(args: argparse.Namespace, group_size: int) -> None:
    shapes = make_shapes(args, group_size)
    group_a, group_b = make_inputs(shapes)
    for provider in args.providers:
        median_ms, low_ms, high_ms = benchmark_provider(provider, group_a, group_b, args)
        row = result_row(provider, shapes, args.warmup, args.iters, median_ms, low_ms, high_ms, args)
        append_result(args.output, row)
        print(row)


def run_sweep(args: argparse.Namespace) -> None:
    for group_size in range(args.min_group_size, args.max_group_size + 1, args.group_size_step):
        run_single(args, group_size)


def run_perf_report(args: argparse.Namespace) -> None:
    x_vals = list(range(args.min_group_size, args.max_group_size + 1, args.group_size_step))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["group_size"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            styles=[("blue", "-"), ("green", "-")][: len(args.providers)],
            ylabel="TFLOP/s",
            plot_name="w11-grouped-gemm",
            args={"bench_args": args},
        )
    )
    def benchmark(group_size: int, provider: str, bench_args: argparse.Namespace):
        shapes = make_shapes(bench_args, group_size)
        group_a, group_b = make_inputs(shapes)
        median_ms, low_ms, high_ms = benchmark_provider(provider, group_a, group_b, bench_args)
        return tflops(median_ms, shapes), tflops(high_ms, shapes), tflops(low_ms, shapes)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.run(print_data=True, show_plots=args.show_plots, save_path=str(args.plot_dir))
    except TypeError as exc:
        if "save_path" not in str(exc):
            raise
        benchmark.run(print_data=True, show_plots=args.show_plots)


def parse_providers(raw: str) -> list[str]:
    providers = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"triton", "torch_loop"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton grouped GEMM against a PyTorch matmul loop.")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--pattern", choices=["same", "vary_n", "vary_m", "vary_all"], default="vary_n")
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("triton,torch_loop"))
    parser.add_argument("--block-size-m", type=int, default=128)
    parser.add_argument("--block-size-n", type=int, default=128)
    parser.add_argument("--block-size-k", type=int, default=32)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w11_grouped_gemm.csv")
    parser.add_argument("--sweep", action="store_true", help="Sweep group size and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-group-size", type=int, default=1)
    parser.add_argument("--max-group-size", type=int, default=8)
    parser.add_argument("--group-size-step", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.group_size <= 0 or args.min_group_size <= 0 or args.max_group_size <= 0:
        raise ValueError("group sizes must be positive")
    if args.group_size_step <= 0:
        raise ValueError("group-size-step must be positive")
    if args.m <= 0 or args.n <= 0 or args.k <= 0 or args.step < 0:
        raise ValueError("m, n, k must be positive and step must be non-negative")
    if args.num_sms is None:
        args.num_sms = default_num_sms()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.group_size)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
