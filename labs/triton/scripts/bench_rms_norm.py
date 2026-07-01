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

from triton_playground.kernels.rms_norm import default_group_size_m, default_num_warps, resolve_block_size
from triton_playground.ops import naive_rms_norm, rms_norm


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}

QUANTILES = [0.5, 0.2, 0.8]


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    weight_float = weight.float()
    rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_float * rstd * weight_float).to(dtype=x.dtype)


def assert_rms_norm_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def do_bench_ms(fn, warmup: int, iters: int, grad_to_none: list[torch.Tensor] | None = None) -> tuple[float, float, float]:
    kwargs = {
        "warmup": warmup,
        "rep": iters,
        "quantiles": QUANTILES,
    }
    if grad_to_none is not None:
        kwargs["grad_to_none"] = grad_to_none
    try:
        median_ms, low_ms, high_ms = triton.testing.do_bench(fn, **kwargs)
    except TypeError as exc:
        if grad_to_none is None or "grad_to_none" not in str(exc):
            raise

        def wrapped_fn():
            for tensor in grad_to_none:
                tensor.grad = None
            return fn()

        del kwargs["grad_to_none"]
        median_ms, low_ms, high_ms = triton.testing.do_bench(wrapped_fn, **kwargs)
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


def make_inputs(rows: int, cols: int, dtype: torch.dtype, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (-2.3 + 0.5 * torch.randn((rows, cols), device="cuda", dtype=dtype)).requires_grad_(True)
    weight = torch.rand((cols,), device="cuda", dtype=dtype, requires_grad=True)
    dy = 0.1 * torch.randn_like(x)

    y_prod = rms_norm(x, weight, eps=eps)
    y_naive = naive_rms_norm(x, weight, eps=eps)
    y_ref = torch_rms_norm(x, weight, eps=eps)
    assert_rms_norm_close(y_prod, y_ref)
    assert_rms_norm_close(y_naive, y_ref)

    y_prod.backward(dy, retain_graph=True)
    dx_prod = x.grad.detach().clone()
    dw_prod = weight.grad.detach().clone()
    x.grad = None
    weight.grad = None

    y_naive.backward(dy, retain_graph=True)
    dx_naive = x.grad.detach().clone()
    dw_naive = weight.grad.detach().clone()
    x.grad = None
    weight.grad = None

    y_ref.backward(dy, retain_graph=True)
    assert_rms_norm_close(dx_prod, x.grad)
    assert_rms_norm_close(dx_naive, x.grad)
    assert_rms_norm_close(dw_prod, weight.grad)
    assert_rms_norm_close(dw_naive, weight.grad)
    x.grad = None
    weight.grad = None
    return x, weight, dy


def forward_provider(provider: str, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if provider == "triton_prod":
        return rms_norm(x, weight, eps=eps)
    if provider == "triton_naive":
        return naive_rms_norm(x, weight, eps=eps)
    if provider == "torch":
        return torch_rms_norm(x, weight, eps=eps)
    raise ValueError(f"unknown provider: {provider}")


def benchmark_provider(
    provider: str,
    mode: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    eps: float,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    if mode == "forward":
        return do_bench_ms(lambda: forward_provider(provider, x, weight, eps), warmup, iters)
    if mode == "backward":
        y = forward_provider(provider, x, weight, eps)
        return do_bench_ms(
            lambda: y.backward(dy, retain_graph=True),
            warmup,
            iters,
            grad_to_none=[x, weight],
        )
    raise ValueError(f"unknown mode: {mode}")


def estimated_bytes(mode: str, x: torch.Tensor) -> int:
    factor = 3 if mode == "forward" else 5
    return factor * x.numel() * x.element_size()


def result_row(
    provider: str,
    mode: str,
    rows: int,
    cols: int,
    dtype_name: str,
    block_size: int,
    num_warps: int,
    group_size_m: int,
    warmup: int,
    iters: int,
    median_ms: float,
    low_ms: float,
    high_ms: float,
    bytes_moved: int,
) -> dict[str, object]:
    if provider == "triton_prod":
        config = f"mode={mode};BLOCK_SIZE={block_size};num_warps={num_warps};GROUP_SIZE_M={group_size_m}"
        notes = "production-style Triton RMSNorm; backward fuses dx and partial dweight"
    elif provider == "triton_naive":
        config = f"mode={mode};BLOCK_SIZE={block_size};num_warps={num_warps};naive_dw_reduce"
        notes = "naive Triton RMSNorm; backward uses separate dx and column-wise dweight reduction"
    else:
        config = f"mode={mode};torch_expression"
        notes = "PyTorch expression baseline: x * rsqrt(mean(x^2) + eps) * weight"

    return {
        "stage": "W12",
        "op": "rms_norm",
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
        "reference": "torch_expression",
        "device": torch.cuda.get_device_name(),
        "notes": notes,
    }


def run_single(args: argparse.Namespace, cols: int) -> None:
    dtype = DTYPES[args.dtype]
    block_size = resolve_block_size(cols, dtype, args.block_size)
    num_warps = args.num_warps if args.num_warps is not None else default_num_warps(block_size)
    group_size_m = args.group_size_m if args.group_size_m is not None else default_group_size_m(cols)
    x, weight, dy = make_inputs(args.rows, cols, dtype, args.eps)
    bytes_moved = estimated_bytes(args.mode, x)

    for provider in args.providers:
        median_ms, low_ms, high_ms = benchmark_provider(
            provider,
            args.mode,
            x,
            weight,
            dy,
            args.eps,
            args.warmup,
            args.iters,
        )
        row = result_row(
            provider,
            args.mode,
            args.rows,
            cols,
            args.dtype,
            block_size,
            num_warps,
            group_size_m,
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
    for cols in range(args.min_cols, args.max_cols + 1, args.cols_step):
        run_single(args, cols)


def run_perf_report(args: argparse.Namespace) -> None:
    x_vals = list(range(args.min_cols, args.max_cols + 1, args.cols_step))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["cols"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            styles=[("blue", "-"), ("green", "-"), ("red", "-")][: len(args.providers)],
            ylabel="GB/s_est",
            plot_name=f"w12-rms-norm-{args.mode}",
            args={
                "rows": args.rows,
                "dtype_name": args.dtype,
                "mode": args.mode,
                "eps": args.eps,
                "warmup": args.warmup,
                "iters": args.iters,
            },
        )
    )
    def benchmark(cols: int, provider: str, rows: int, dtype_name: str, mode: str, eps: float, warmup: int, iters: int):
        dtype = DTYPES[dtype_name]
        x, weight, dy = make_inputs(rows, cols, dtype, eps)
        median_ms, low_ms, high_ms = benchmark_provider(provider, mode, x, weight, dy, eps, warmup, iters)
        bytes_moved = estimated_bytes(mode, x)
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
    allowed = {"triton_prod", "triton_naive", "torch"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton RMSNorm forward/backward against PyTorch.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=8192)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--mode", choices=["forward", "backward"], default="backward")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--group-size-m", type=int, default=None)
    parser.add_argument("--providers", type=parse_providers, default=parse_providers("triton_prod,triton_naive,torch"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUT_BENCHMARK_ROOT / "w12_rms_norm.csv")
    parser.add_argument("--sweep", action="store_true", help="Run a column sweep and append CSV rows.")
    parser.add_argument("--plot", action="store_true", help="Generate a Triton perf_report plot for the sweep.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows when generating plots.")
    parser.add_argument("--plot-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--min-cols", type=int, default=1024)
    parser.add_argument("--max-cols", type=int, default=16384)
    parser.add_argument("--cols-step", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")
    if args.cols_step <= 0:
        raise ValueError("cols-step must be positive")

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args, args.cols)

    if args.plot:
        run_perf_report(args)


if __name__ == "__main__":
    main()
