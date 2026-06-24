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

from triton_playground.kernels.fused_softmax import default_num_warps, next_power_of_2
from triton_playground.ops import fused_softmax, naive_softmax


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


def gbps(ms: float, bytes_moved: int) -> float:
    return bytes_moved * 1e-9 / (ms * 1e-3)


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


def make_input(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    expected = torch.softmax(x, dim=-1)
    actual = fused_softmax(x)
    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return x


def benchmark_provider(
    provider: str,
    x: torch.Tensor,
    block_size: int,
    num_warps: int,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    if provider == "triton":
        return do_bench_ms(lambda: fused_softmax(x, block_size=block_size, num_warps=num_warps), warmup, iters)
    if provider == "torch":
        return do_bench_ms(lambda: torch.softmax(x, dim=-1), warmup, iters)
    if provider == "naive":
        return do_bench_ms(lambda: naive_softmax(x), warmup, iters)
    raise ValueError(f"unknown provider: {provider}")


def result_row(
    provider: str,
    rows: int,
    cols: int,
    dtype_name: str,
    block_size: int,
    num_warps: int,
    warmup: int,
    iters: int,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    bytes_moved: int,
) -> dict[str, object]:
    config = {
        "triton": f"BLOCK_SIZE={block_size};num_warps={num_warps}",
        "torch": "torch_softmax",
        "naive": "torch_max_exp_sum_div",
    }[provider]
    notes = {
        "triton": "one Triton program per row",
        "torch": "PyTorch softmax baseline",
        "naive": "PyTorch naive multi-kernel baseline",
    }[provider]
    return {
        "stage": "W09",
        "op": "fused_softmax",
        "impl": provider,
        "shape": f"{rows}x{cols}",
        "dtype": dtype_name,
        "config": config,
        "warmup": warmup,
        "iters": iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{gbps(median_ms, bytes_moved):.3f}",
        "unit": "GB/s_est",
        "reference": "torch_softmax",
        "device": torch.cuda.get_device_name(),
        "notes": notes,
    }


def run_single(args: argparse.Namespace, cols: int) -> None:
    dtype = DTYPES[args.dtype]
    block_size = args.block_size if args.block_size is not None else next_power_of_2(cols)
    num_warps = args.num_warps if args.num_warps is not None else default_num_warps(block_size)
    x = make_input(args.rows, cols, dtype)
    bytes_moved = args.rows * cols * x.element_size() * 2

    for provider in ("triton", "torch", "naive"):
        median_ms, low_ms, high_ms = benchmark_provider(provider, x, block_size, num_warps, args.warmup, args.iters)
        row = result_row(
            provider,
            args.rows,
            cols,
            args.dtype,
            block_size,
            num_warps,
            args.warmup,
            args.iters,
            median_ms,
            low_ms,
            high_ms,
            bytes_moved,
        )
        append_result(args.output, row)
        print(row)


def run_sweep(args: argparse.Namespace) -> None:
    for power in range(args.min_cols_power, args.max_cols_power + 1):
        run_single(args, 1 << power)


def run_perf_report(args: argparse.Namespace) -> None:
    x_vals = [1 << power for power in range(args.min_cols_power, args.max_cols_power + 1)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["cols"],
            x_vals=x_vals,
            x_log=True,
            line_arg="provider",
            line_vals=["triton", "torch", "naive"],
            line_names=["Triton fused", "Torch softmax", "Torch naive"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="GB/s_est",
            plot_name="w09-fused-softmax",
            args={
                "rows": args.rows,
                "dtype_name": args.dtype,
                "warmup": args.warmup,
                "iters": args.iters,
            },
        )
    )
    def benchmark(cols: int, provider: str, rows: int, dtype_name: str, warmup: int, iters: int):
        dtype = DTYPES[dtype_name]
        block_size = next_power_of_2(cols)
        num_warps = default_num_warps(block_size)
        x = make_input(rows, cols, dtype)
        bytes_moved = rows * cols * x.element_size() * 2
        median_ms, low_ms, high_ms = benchmark_provider(provider, x, block_size, num_warps, warmup, iters)
        return gbps(median_ms, bytes_moved), gbps(high_ms, bytes_moved), gbps(low_ms, bytes_moved)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.run(print_data=True, show_plots=args.show_plots, save_path=str(args.plot_dir))
    except TypeError as exc:
        if "save_path" not in str(exc):
            raise
        benchmark.run(print_data=True, show_plots=args.show_plots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark W09 Triton fused softmax against PyTorch baselines.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w09_vector_add_softmax.csv")
    parser.add_argument("--sweep", action="store_true", help="Run a power-of-two column sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-cols-power", type=int, default=7)
    parser.add_argument("--max-cols-power", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.cols)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
