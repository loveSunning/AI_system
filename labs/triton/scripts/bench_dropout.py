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

from triton_playground.ops import dropout_with_mask, low_memory_dropout


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}

QUANTILES = [0.5, 0.2, 0.8]


def assert_dropout_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


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


def make_inputs(n_elements: int, dtype: torch.dtype, p: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    keep_mask = torch.rand(n_elements, device="cuda") > p

    expected = torch.where(keep_mask, x / (1.0 - p), torch.zeros_like(x))
    actual = dropout_with_mask(x, keep_mask, p=p)
    assert_dropout_close(actual, expected)

    seeded_1 = low_memory_dropout(x, p=p, seed=seed)
    seeded_2 = low_memory_dropout(x, p=p, seed=seed)
    torch.testing.assert_close(seeded_1, seeded_2, rtol=0.0, atol=0.0)
    return x, keep_mask


def benchmark_provider(
    provider: str,
    x: torch.Tensor,
    keep_mask: torch.Tensor,
    p: float,
    seed: int,
    block_size: int,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    if provider == "triton_mask":
        return do_bench_ms(lambda: dropout_with_mask(x, keep_mask, p=p, block_size=block_size), warmup, iters)
    if provider == "triton_seeded_low_memory":
        return do_bench_ms(lambda: low_memory_dropout(x, p=p, seed=seed, block_size=block_size), warmup, iters)
    if provider == "torch":
        return do_bench_ms(lambda: F.dropout(x, p=p, training=True), warmup, iters)
    raise ValueError(f"unknown provider: {provider}")


def config_text(provider: str, p: float, block_size: int, seed: int) -> str:
    if provider == "triton_mask":
        return f"p={p};BLOCK_SIZE={block_size};explicit_keep_mask"
    if provider == "triton_seeded_low_memory":
        return f"p={p};BLOCK_SIZE={block_size};seed={seed}"
    return f"p={p};torch_dropout"


def bytes_moved_for(provider: str, n_elements: int, element_size: int) -> int:
    if provider == "triton_mask":
        return n_elements * (element_size * 2 + 1)
    return n_elements * element_size * 2


def result_row(
    provider: str,
    n_elements: int,
    dtype_name: str,
    p: float,
    block_size: int,
    seed: int,
    warmup: int,
    iters: int,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    bytes_moved: int,
) -> dict[str, object]:
    notes = {
        "triton_mask": "explicit keep-mask baseline from Triton tutorial",
        "triton_seeded_low_memory": "tl.rand(seed, offsets), no mask tensor materialized",
        "torch": "torch.nn.functional.dropout baseline",
    }[provider]
    return {
        "stage": "W12",
        "op": "dropout",
        "impl": provider,
        "shape": str(n_elements),
        "dtype": dtype_name,
        "config": config_text(provider, p, block_size, seed),
        "warmup": warmup,
        "iters": iters,
        "avg_ms": f"{median_ms:.6f}",
        "min_ms": f"{low_ms:.6f}",
        "max_ms": f"{high_ms:.6f}",
        "throughput": f"{gbps(median_ms, bytes_moved):.3f}",
        "unit": "GB/s_est",
        "reference": "torch_dropout",
        "device": torch.cuda.get_device_name(),
        "notes": notes,
    }


def run_single(args: argparse.Namespace, n_elements: int) -> None:
    dtype = DTYPES[args.dtype]
    x, keep_mask = make_inputs(n_elements, dtype, args.p, args.seed)

    for provider in args.providers:
        median_ms, low_ms, high_ms = benchmark_provider(
            provider,
            x,
            keep_mask,
            args.p,
            args.seed,
            args.block_size,
            args.warmup,
            args.iters,
        )
        bytes_moved = bytes_moved_for(provider, n_elements, x.element_size())
        row = result_row(
            provider,
            n_elements,
            args.dtype,
            args.p,
            args.block_size,
            args.seed,
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
            line_vals=args.providers,
            line_names=args.providers,
            styles=[("blue", "-"), ("green", "-"), ("red", "-")][: len(args.providers)],
            ylabel="GB/s_est",
            plot_name="w12-dropout",
            args={
                "dtype_name": args.dtype,
                "p": args.p,
                "seed": args.seed,
                "block_size": args.block_size,
                "warmup": args.warmup,
                "iters": args.iters,
            },
        )
    )
    def benchmark(n_elements: int, provider: str, dtype_name: str, p: float, seed: int, block_size: int, warmup: int, iters: int):
        dtype = DTYPES[dtype_name]
        x, keep_mask = make_inputs(n_elements, dtype, p, seed)
        median_ms, low_ms, high_ms = benchmark_provider(provider, x, keep_mask, p, seed, block_size, warmup, iters)
        bytes_moved = bytes_moved_for(provider, n_elements, x.element_size())
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
    allowed = {"triton_mask", "triton_seeded_low_memory", "torch"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton explicit-mask dropout and low-memory seeded dropout.")
    parser.add_argument("--n-elements", type=int, default=1 << 24)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("triton_mask,triton_seeded_low_memory,torch"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w12_dropout.csv")
    parser.add_argument("--sweep", action="store_true", help="Run a power-of-two size sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-power", type=int, default=12)
    parser.add_argument("--max-power", type=int, default=28)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not 0.0 <= args.p < 1.0:
        raise ValueError(f"dropout probability must satisfy 0 <= p < 1, got {args.p}")

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.n_elements)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
