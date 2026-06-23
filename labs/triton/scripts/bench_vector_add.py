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

from triton_playground.ops import vector_add


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
    parser = argparse.ArgumentParser(description="Benchmark W09 Triton vector add against PyTorch.")
    parser.add_argument("--n-elements", type=int, default=1 << 24)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path, default=TRITON_ROOT / "benchmarks" / "w09_vector_add_softmax.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = DTYPES[args.dtype]
    x = torch.randn(args.n_elements, device="cuda", dtype=dtype)
    y = torch.randn(args.n_elements, device="cuda", dtype=dtype)

    actual = vector_add(x, y, block_size=args.block_size)
    expected = x + y
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    device = torch.cuda.get_device_name()
    bytes_moved = args.n_elements * x.element_size() * 3

    triton_avg, triton_min, triton_max = time_cuda_ms(
        lambda: vector_add(x, y, block_size=args.block_size),
        args.warmup,
        args.iters,
    )
    torch_avg, torch_min, torch_max = time_cuda_ms(lambda: x + y, args.warmup, args.iters)

    rows = [
        {
            "stage": "W09",
            "op": "vector_add",
            "impl": "triton",
            "shape": str(args.n_elements),
            "dtype": args.dtype,
            "config": f"BLOCK_SIZE={args.block_size}",
            "warmup": args.warmup,
            "iters": args.iters,
            "avg_ms": f"{triton_avg:.6f}",
            "min_ms": f"{triton_min:.6f}",
            "max_ms": f"{triton_max:.6f}",
            "throughput": f"{bytes_moved / (triton_avg * 1e6):.3f}",
            "unit": "GB/s",
            "reference": "torch_add",
            "device": device,
            "notes": "minimal one-dimensional vector add",
        },
        {
            "stage": "W09",
            "op": "vector_add",
            "impl": "torch",
            "shape": str(args.n_elements),
            "dtype": args.dtype,
            "config": "torch_add",
            "warmup": args.warmup,
            "iters": args.iters,
            "avg_ms": f"{torch_avg:.6f}",
            "min_ms": f"{torch_min:.6f}",
            "max_ms": f"{torch_max:.6f}",
            "throughput": f"{bytes_moved / (torch_avg * 1e6):.3f}",
            "unit": "GB/s",
            "reference": "torch_add",
            "device": device,
            "notes": "PyTorch eager baseline",
        },
    ]

    for row in rows:
        append_result(args.output, row)
        print(row)


if __name__ == "__main__":
    main()
