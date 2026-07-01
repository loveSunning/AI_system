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
from triton_playground.ops import fused_softmax, online_softmax, torch_online_softmax


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


def assert_close(torch_module, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float16:
        torch_module.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        torch_module.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def make_input(rows: int, cols: int, dtype: torch.dtype, online_block_size: int, check_correctness: bool) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    if check_correctness:
        expected = torch.softmax(x, dim=-1)
        assert_close(torch, online_softmax(x, block_size=online_block_size), expected)
        assert_close(torch, torch_online_softmax(x, block_size=online_block_size), expected)
        assert_close(torch, fused_softmax(x), expected)
    return x


def benchmark_provider(
    provider: str,
    x: torch.Tensor,
    online_block_size: int,
    online_num_warps: int,
    fused_block_size: int,
    fused_num_warps: int,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    if provider == "triton_online":
        return do_bench_ms(
            lambda: online_softmax(x, block_size=online_block_size, num_warps=online_num_warps),
            warmup,
            iters,
        )
    if provider == "torch_online":
        return do_bench_ms(lambda: torch_online_softmax(x, block_size=online_block_size), warmup, iters)
    if provider == "triton_fused":
        return do_bench_ms(
            lambda: fused_softmax(x, block_size=fused_block_size, num_warps=fused_num_warps),
            warmup,
            iters,
        )
    if provider == "torch_softmax":
        return do_bench_ms(lambda: torch.softmax(x, dim=-1), warmup, iters)
    raise ValueError(f"unknown provider: {provider}")


def config_text(provider: str, online_block_size: int, online_num_warps: int, fused_block_size: int, fused_num_warps: int) -> str:
    if provider == "triton_online":
        return f"BLOCK_SIZE={online_block_size};num_warps={online_num_warps};two_pass_online"
    if provider == "torch_online":
        return f"chunk_size={online_block_size};two_pass_online"
    if provider == "triton_fused":
        return f"BLOCK_SIZE={fused_block_size};num_warps={fused_num_warps};single_block_row"
    return "torch_softmax"


def notes_text(provider: str) -> str:
    return {
        "triton_online": "two-pass block-streaming online softmax, one program per row",
        "torch_online": "PyTorch loop implementing online max/sum recurrence",
        "triton_fused": "W09 fused softmax baseline, one block covers one full row",
        "torch_softmax": "PyTorch softmax baseline",
    }[provider]


def result_row(
    provider: str,
    rows: int,
    cols: int,
    dtype_name: str,
    online_block_size: int,
    online_num_warps: int,
    fused_block_size: int,
    fused_num_warps: int,
    warmup: int,
    iters: int,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    bytes_moved: int,
) -> dict[str, object]:
    return {
        "stage": "W13",
        "op": "online_softmax",
        "impl": provider,
        "shape": f"{rows}x{cols}",
        "dtype": dtype_name,
        "config": config_text(provider, online_block_size, online_num_warps, fused_block_size, fused_num_warps),
        "warmup": warmup,
        "iters": iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{gbps(median_ms, bytes_moved):.3f}",
        "unit": "GB/s_est",
        "reference": "torch_softmax",
        "device": torch.cuda.get_device_name(),
        "notes": notes_text(provider),
    }


def run_single(args: argparse.Namespace, cols: int) -> None:
    dtype = DTYPES[args.dtype]
    online_block_size = args.block_size
    online_num_warps = args.num_warps if args.num_warps is not None else default_num_warps(online_block_size)
    fused_block_size = next_power_of_2(cols)
    fused_num_warps = default_num_warps(fused_block_size)
    x = make_input(args.rows, cols, dtype, online_block_size, check_correctness=not args.skip_correctness)

    # Logical read + write traffic, kept consistent with W09 fused_softmax for chart comparison.
    bytes_moved = args.rows * cols * x.element_size() * 2
    for provider in args.providers:
        median_ms, low_ms, high_ms = benchmark_provider(
            provider,
            x,
            online_block_size,
            online_num_warps,
            fused_block_size,
            fused_num_warps,
            args.warmup,
            args.iters,
        )
        row = result_row(
            provider,
            args.rows,
            cols,
            args.dtype,
            online_block_size,
            online_num_warps,
            fused_block_size,
            fused_num_warps,
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
    styles = [("blue", "-"), ("orange", "-"), ("green", "-"), ("red", "-")]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["cols"],
            x_vals=x_vals,
            x_log=True,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            styles=styles[: len(args.providers)],
            ylabel="GB/s_est",
            plot_name="w13-online-softmax",
            args={
                "rows": args.rows,
                "dtype_name": args.dtype,
                "online_block_size": args.block_size,
                "warmup": args.warmup,
                "iters": args.iters,
                "skip_correctness": args.skip_correctness,
            },
        )
    )
    def benchmark(
        cols: int,
        provider: str,
        rows: int,
        dtype_name: str,
        online_block_size: int,
        warmup: int,
        iters: int,
        skip_correctness: bool,
    ):
        dtype = DTYPES[dtype_name]
        online_num_warps = default_num_warps(online_block_size)
        fused_block_size = next_power_of_2(cols)
        fused_num_warps = default_num_warps(fused_block_size)
        x = make_input(rows, cols, dtype, online_block_size, check_correctness=not skip_correctness)
        bytes_moved = rows * cols * x.element_size() * 2
        median_ms, low_ms, high_ms = benchmark_provider(
            provider,
            x,
            online_block_size,
            online_num_warps,
            fused_block_size,
            fused_num_warps,
            warmup,
            iters,
        )
        return gbps(median_ms, bytes_moved), gbps(high_ms, bytes_moved), gbps(low_ms, bytes_moved)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.run(print_data=True, show_plots=args.show_plots, save_path=str(args.plot_dir))
    except TypeError as exc:
        if "save_path" not in str(exc):
            raise
        benchmark.run(print_data=True, show_plots=args.show_plots)


def parse_providers(raw: str) -> list[str]:
    providers = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"triton_online", "torch_online", "triton_fused", "torch_softmax"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark W13 online softmax against fused softmax and PyTorch.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("triton_online,torch_online,triton_fused,torch_softmax"))
    parser.add_argument("--block-size", type=int, default=1024, help="Streaming chunk size for online softmax.")
    parser.add_argument("--num-warps", type=int, default=None, help="Triton online softmax num_warps.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w13_online_softmax.csv")
    parser.add_argument("--sweep", action="store_true", help="Run a power-of-two column sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-cols-power", type=int, default=7)
    parser.add_argument("--max-cols-power", type=int, default=13)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.block_size <= 0:
        raise ValueError("block_size must be positive")
    if args.block_size & (args.block_size - 1):
        raise ValueError("block_size must be a power of two")
    if args.num_warps is not None and args.num_warps <= 0:
        raise ValueError("num_warps must be positive")
    if not args.providers:
        raise ValueError("at least one provider must be enabled")

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.cols)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
