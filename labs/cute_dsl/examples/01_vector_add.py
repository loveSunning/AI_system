"""Minimal CuTe DSL vector addition for NVIDIA Ada GPUs such as RTX 4090 D.

Computation:
    out[i] = x[i] + y[i]

The first run JIT-compiles the Python DSL to a CUDA kernel. Later runs can use
the CuTe DSL compilation cache.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
import time

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS_PER_BLOCK = 256


@cute.kernel
def vector_add_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    out: cute.Tensor,
    num_elements: cutlass.Int32,
):
    """Device code: one CUDA thread computes one output element."""
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    block_dim, _, _ = cute.arch.block_dim()

    idx = block_idx * block_dim + thread_idx
    if idx < num_elements:
        out[idx] = x[idx] + y[idx]


@cute.jit
def launch_vector_add(
    stream: cuda.CUstream,
    x: cute.Tensor,
    y: cute.Tensor,
    out: cute.Tensor,
    num_elements: cutlass.Int32,
):
    """Host-side JIT wrapper: calculate the grid and launch the kernel."""
    grid_size = (num_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    vector_add_kernel(x, y, out, num_elements).launch(
        grid=(grid_size, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        stream=stream,
    )


def run(num_elements: int, warmup_iterations: int, iterations: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Start the container with GPU access.")
    if num_elements <= 0:
        raise ValueError("--num-elements must be greater than zero")
    if warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be zero or greater")
    if iterations <= 0:
        raise ValueError("--iterations must be greater than zero")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name}")
    print(f"Compute capability: {capability}")
    print(f"PyTorch: {torch.__version__}, PyTorch CUDA: {torch.version.cuda}")
    try:
        dsl_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        dsl_version = "unknown"
    print(f"CUTLASS DSL: {dsl_version}")

    if capability != (8, 9):
        print("Warning: this tutorial targets RTX 4090 D / SM89, continuing anyway.")

    torch.manual_seed(0)
    x = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    # DLPack creates zero-copy CuTe tensor views of PyTorch CUDA allocations.
    x_cute = from_dlpack(x, assumed_align=16)
    y_cute = from_dlpack(y, assumed_align=16)
    out_cute = from_dlpack(out, assumed_align=16)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    print("Compiling CuTe DSL kernel (the first run can take a few seconds)...")
    compile_start = time.perf_counter()
    compiled = cute.compile(
        launch_vector_add,
        stream,
        x_cute,
        y_cute,
        out_cute,
        num_elements,
    )
    compile_time_s = time.perf_counter() - compile_start
    print(f"Compilation time: {compile_time_s:.3f} s")

    print("Launching kernel for correctness check...")
    compiled(stream, x_cute, y_cute, out_cute, num_elements)
    torch_stream.synchronize()

    reference = x + y
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-6)

    max_error = (out - reference).abs().max().item()
    print(f"First five outputs: {out[:5].tolist()}")
    print(f"Max absolute error: {max_error:.3e}")

    print(f"Warming up: {warmup_iterations} iterations...")
    for _ in range(warmup_iterations):
        compiled(stream, x_cute, y_cute, out_cute, num_elements)
    torch_stream.synchronize()

    # CUDA events measure GPU elapsed time on the same stream used by CuTe DSL.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(torch_stream)
    for _ in range(iterations):
        compiled(stream, x_cute, y_cute, out_cute, num_elements)
    end_event.record(torch_stream)
    end_event.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    average_time_ms = total_time_ms / iterations
    average_time_us = average_time_ms * 1_000.0

    # Each invocation reads x and y, then writes out: 3 float32 values/element.
    bytes_per_iteration = num_elements * 3 * out.element_size()
    throughput_gbps = bytes_per_iteration / (average_time_ms / 1_000.0) / 1e9

    print(f"Benchmark iterations: {iterations}")
    print(f"Total GPU time: {total_time_ms:.3f} ms")
    print(f"Average kernel time: {average_time_us:.3f} us ({average_time_ms:.6f} ms)")
    print(f"Effective memory throughput: {throughput_gbps:.2f} GB/s")
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal CuTe DSL vector addition")
    parser.add_argument(
        "--num-elements",
        type=int,
        default=1_000_003,
        help="Vector length; the non-multiple default also tests the boundary predicate",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Warmup launches excluded from timing",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed kernel launches",
    )
    args = parser.parse_args()
    run(args.num_elements, args.warmup_iterations, args.iterations)
