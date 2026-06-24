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

from triton_playground.ops import vector_add


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


def make_inputs(n_elements: int, dtype: torch.dtype, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    y = torch.randn(n_elements, device="cuda", dtype=dtype)
    actual = vector_add(x, y, block_size=block_size)
    torch.testing.assert_close(actual, x + y, rtol=0.0, atol=0.0)
    return x, y


def benchmark_provider(
    provider: str,
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    if provider == "triton":
        return do_bench_ms(lambda: vector_add(x, y, block_size=block_size), warmup, iters)
    if provider == "torch":
        return do_bench_ms(lambda: x + y, warmup, iters)
    raise ValueError(f"unknown provider: {provider}")


def result_row(
    provider: str,
    n_elements: int,
    dtype_name: str,
    block_size: int,
    warmup: int,
    iters: int,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    bytes_moved: int,
) -> dict[str, object]:
    return {
        "stage": "W09",
        "op": "vector_add",
        "impl": provider,
        "shape": str(n_elements),
        "dtype": dtype_name,
        "config": f"BLOCK_SIZE={block_size}" if provider == "triton" else "torch_add",
        "warmup": warmup,
        "iters": iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{gbps(median_ms, bytes_moved):.3f}",
        "unit": "GB/s",
        "reference": "torch_add",
        "device": torch.cuda.get_device_name(),
        "notes": "triton.testing.do_bench median/p20/p80",
    }


def run_single(args: argparse.Namespace, n_elements: int) -> None:
    dtype = DTYPES[args.dtype]
    x, y = make_inputs(n_elements, dtype, args.block_size)
    bytes_moved = n_elements * x.element_size() * 3

    for provider in ("triton", "torch"):
        median_ms, low_ms, high_ms = benchmark_provider(provider, x, y, args.block_size, args.warmup, args.iters)
        row = result_row(
            provider,
            n_elements,
            args.dtype,
            args.block_size,
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
    for power in range(args.min_power, args.max_power + 1):
        run_single(args, 1 << power)


def run_perf_report(args: argparse.Namespace) -> None:
    x_vals = [1 << power for power in range(args.min_power, args.max_power + 1)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n_elements"],
            x_vals=x_vals,
            x_log=True,
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="GB/s",
            plot_name="w09-vector-add",
            args={
                "dtype_name": args.dtype,
                "block_size": args.block_size,
                "warmup": args.warmup,
                "iters": args.iters,
            },
        )
    )
    def benchmark(n_elements: int, provider: str, dtype_name: str, block_size: int, warmup: int, iters: int):
        dtype = DTYPES[dtype_name]
        x, y = make_inputs(n_elements, dtype, block_size)
        bytes_moved = n_elements * x.element_size() * 3
        median_ms, low_ms, high_ms = benchmark_provider(provider, x, y, block_size, warmup, iters)
        return gbps(median_ms, bytes_moved), gbps(high_ms, bytes_moved), gbps(low_ms, bytes_moved)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.run(print_data=True, show_plots=args.show_plots, save_path=str(args.plot_dir))
    except TypeError as exc:
        if "save_path" not in str(exc):
            raise
        benchmark.run(print_data=True, show_plots=args.show_plots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark W09 Triton vector add against PyTorch.")
    parser.add_argument("--n-elements", type=int, default=1 << 24)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w09_vector_add_softmax.csv")
    parser.add_argument("--sweep", action="store_true", help="Run a power-of-two size sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-power", type=int, default=12)
    parser.add_argument("--max-power", type=int, default=28)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.n_elements)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
