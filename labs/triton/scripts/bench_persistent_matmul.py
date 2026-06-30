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

from triton_playground.kernels.persistent_matmul import default_num_sms, is_likely_4090d
from triton_playground.ops import matmul_fixed, persistent_matmul, persistent_matmul_fixed


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
        torch_module.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-1)
    else:
        torch_module.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def make_inputs(m: int, n: int, k: int, dtype: torch.dtype, check_autotune: bool, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    expected = torch.matmul(a, b)
    assert_close(
        torch,
        persistent_matmul_fixed(
            a,
            b,
            block_size_m=args.block_size_m,
            block_size_n=args.block_size_n,
            block_size_k=args.block_size_k,
            group_size_m=args.group_size_m,
            num_sms=args.num_sms,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        ),
        expected,
    )
    if check_autotune:
        assert_close(torch, persistent_matmul(a, b, num_sms=args.num_sms), expected)
    return a, b


def benchmark_provider(
    provider: str,
    a: torch.Tensor,
    b: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    if provider == "persistent_autotune":
        return do_bench_ms(lambda: persistent_matmul(a, b, num_sms=args.num_sms), args.warmup, args.iters)
    if provider == "persistent_fixed":
        return do_bench_ms(
            lambda: persistent_matmul_fixed(
                a,
                b,
                block_size_m=args.block_size_m,
                block_size_n=args.block_size_n,
                block_size_k=args.block_size_k,
                group_size_m=args.group_size_m,
                num_sms=args.num_sms,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            args.warmup,
            args.iters,
        )
    if provider == "triton_fixed":
        return do_bench_ms(
            lambda: matmul_fixed(
                a,
                b,
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
    if provider == "torch":
        return do_bench_ms(lambda: torch.matmul(a, b), args.warmup, args.iters)
    raise ValueError(f"unknown provider: {provider}")


def config_text(provider: str, args: argparse.Namespace) -> str:
    if provider == "persistent_autotune":
        return f"get_4090d_friendly_configs;NUM_SMS={args.num_sms}"
    if provider in ("persistent_fixed", "triton_fixed"):
        return (
            f"BLOCK_M={args.block_size_m};BLOCK_N={args.block_size_n};BLOCK_K={args.block_size_k};"
            f"GROUP_M={args.group_size_m};NUM_SMS={args.num_sms};"
            f"num_warps={args.num_warps};num_stages={args.num_stages}"
        )
    return "torch_matmul"


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
    notes = {
        "persistent_autotune": "non-TMA persistent scheduling; 4090D-friendly autotune configs",
        "persistent_fixed": "non-TMA persistent scheduling; fixed 4090D-friendly tile",
        "triton_fixed": "regular one-program-per-tile Triton matmul baseline",
        "torch": "torch.matmul baseline",
    }[provider]
    if args.is_4090d:
        notes += "; device detected as RTX 4090/4090D class"
    return {
        "stage": "W11",
        "op": "persistent_matmul",
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
        "reference": "torch_matmul",
        "device": torch.cuda.get_device_name(),
        "notes": notes,
    }


def run_single(args: argparse.Namespace, m: int, n: int, k: int) -> None:
    dtype = DTYPES[args.dtype]
    a, b = make_inputs(m, n, k, dtype, check_autotune=not args.skip_autotune_correctness, args=args)
    for provider in args.providers:
        median_ms, low_ms, high_ms = benchmark_provider(provider, a, b, args)
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
            styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "-")][: len(args.providers)],
            ylabel="TFLOP/s",
            plot_name="w11-persistent-matmul",
            args={"bench_args": args},
        )
    )
    def benchmark(size: int, provider: str, bench_args: argparse.Namespace):
        dtype = DTYPES[bench_args.dtype]
        a, b = make_inputs(size, size, bench_args.k, dtype, check_autotune=(provider == "persistent_autotune"), args=bench_args)
        median_ms, low_ms, high_ms = benchmark_provider(provider, a, b, bench_args)
        return (
            tflops(median_ms, size, size, bench_args.k),
            tflops(high_ms, size, size, bench_args.k),
            tflops(low_ms, size, size, bench_args.k),
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
    allowed = {"persistent_autotune", "persistent_fixed", "triton_fixed", "torch"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark W11 Triton persistent matmul with RTX 4090D-friendly configs.")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("persistent_fixed,persistent_autotune,triton_fixed,torch"))
    parser.add_argument("--block-size-m", type=int, default=128)
    parser.add_argument("--block-size-n", type=int, default=128)
    parser.add_argument("--block-size-k", type=int, default=32)
    parser.add_argument("--group-size-m", type=int, default=8)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w11_persistent_matmul.csv")
    parser.add_argument("--sweep", action="store_true", help="Run square M/N sweep with fixed K and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-power", type=int, default=10)
    parser.add_argument("--max-power", type=int, default=13)
    parser.add_argument("--skip-autotune-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("m, n, and k must be positive")
    if args.num_sms is None:
        args.num_sms = default_num_sms()
    args.is_4090d = is_likely_4090d()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.m, args.n, args.k)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
