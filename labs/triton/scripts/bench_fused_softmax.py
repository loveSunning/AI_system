from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

TRITON_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = TRITON_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import torch

from triton_playground.kernels.fused_softmax import default_num_warps, next_power_of_2
from triton_playground.ops import fused_softmax


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def time_cuda_ms(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times), min(times), max(times)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark W09 Triton fused softmax against PyTorch.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path, default=TRITON_ROOT / "benchmarks" / "w09_vector_add_softmax.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = DTYPES[args.dtype]
    block_size = args.block_size if args.block_size is not None else next_power_of_2(args.cols)
    num_warps = args.num_warps if args.num_warps is not None else default_num_warps(block_size)
    x = torch.randn((args.rows, args.cols), device="cuda", dtype=dtype)

    actual = fused_softmax(x, block_size=block_size, num_warps=num_warps)
    expected = torch.softmax(x, dim=-1)
    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    device = torch.cuda.get_device_name()
    bytes_moved = args.rows * args.cols * x.element_size() * 2
    shape = f"{args.rows}x{args.cols}"

    triton_avg, triton_min, triton_max = time_cuda_ms(
        lambda: fused_softmax(x, block_size=block_size, num_warps=num_warps),
        args.warmup,
        args.iters,
    )
    torch_avg, torch_min, torch_max = time_cuda_ms(lambda: torch.softmax(x, dim=-1), args.warmup, args.iters)

    rows = [
        {
            "stage": "W09",
            "op": "fused_softmax",
            "impl": "triton",
            "shape": shape,
            "dtype": args.dtype,
            "config": f"BLOCK_SIZE={block_size};num_warps={num_warps}",
            "warmup": args.warmup,
            "iters": args.iters,
            "avg_ms": f"{triton_avg:.6f}",
            "min_ms": f"{triton_min:.6f}",
            "max_ms": f"{triton_max:.6f}",
            "throughput": f"{bytes_moved / (triton_avg * 1e6):.3f}",
            "unit": "GB/s_est",
            "reference": "torch_softmax",
            "device": device,
            "notes": "one Triton program per row",
        },
        {
            "stage": "W09",
            "op": "fused_softmax",
            "impl": "torch",
            "shape": shape,
            "dtype": args.dtype,
            "config": "torch_softmax",
            "warmup": args.warmup,
            "iters": args.iters,
            "avg_ms": f"{torch_avg:.6f}",
            "min_ms": f"{torch_min:.6f}",
            "max_ms": f"{torch_max:.6f}",
            "throughput": f"{bytes_moved / (torch_avg * 1e6):.3f}",
            "unit": "GB/s_est",
            "reference": "torch_softmax",
            "device": device,
            "notes": "PyTorch softmax baseline",
        },
    ]

    for row in rows:
        append_result(args.output, row)
        print(row)


if __name__ == "__main__":
    main()
