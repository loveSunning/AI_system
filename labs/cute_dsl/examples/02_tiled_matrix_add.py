"""Tiled 2D matrix addition in CuTe DSL for NVIDIA Ada GPUs such as RTX 4090 D.

Computation:
    C[i, j] = A[i, j] + B[i, j]

Unlike ``01_vector_add.py`` (one thread computes one element with scalar
indexing), this example demonstrates the core CuTe abstractions -- Layout,
Tile and Copy:

1. ``make_layout_tv`` derives a CTA tile shape (16, 128) from a thread layout
   (4, 32) and a value layout (4, 4): 128 threads, each handling 4x4 elements
   with 128-bit vectorized loads/stores.
2. ``zipped_divide`` tiles the (M, N) problem into
   ``((TileM, TileN), (RestM, RestN))``; in the kernel, ``((None, None), bidx)``
   selects the tile handled by the current CTA.
3. ``make_tiled_copy_tv`` + ``partition_S`` partition a tile among the threads
   of a CTA.
4. ``make_rmem_tensor_like`` allocates register fragments and ``cute.copy``
   moves data between global memory and registers.
5. An identity coordinate tensor (``make_identity_tensor``) plus
   ``cute.elem_less`` builds the boundary predicate, so shapes that are not
   multiples of the tile size (e.g. 1000 x 1200) are handled without
   out-of-bounds access.

The first run JIT-compiles the Python DSL to a CUDA kernel; later runs can use
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


# One CTA tile is (4 * VECTOR_SIZE) x ((THREADS // 4) * VECTOR_SIZE) = 16 x 128
# for fp32, derived below from the thread/value layouts inside the JIT launcher.
THREADS = 128
VECTOR_SIZE = 4  # fp32 x 4 = 128-bit vectorized copies
TILE_M = 4 * VECTOR_SIZE
TILE_N = (THREADS // 4) * VECTOR_SIZE


@cute.kernel
def tiled_add_kernel(
    gA: cute.Tensor,  # ((TileM, TileN), (RestM, RestN))
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # identity (coordinate) tensor with the same tiling
    shape: cute.Shape,  # global problem shape (M, N)
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    """Device code: one CTA loads one (TileM, TileN) tile, adds it, stores it."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Keep the two tile modes and index the (RestM, RestN) modes by the block
    # id: each CTA processes exactly one (TileM, TileN) tile.
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]  # (TileM, TileN)
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]

    # Copy atom used for both global->register loads and register->global stores.
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    # Each thread's share of the tile, shaped (CPY, CPY_M, CPY_N).
    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    # Register fragments holding the per-thread values.
    frgA = cute.make_rmem_tensor_like(thrA)
    frgB = cute.make_rmem_tensor_like(thrB)
    frgC = cute.make_rmem_tensor_like(thrC)

    # Predicate: element (m, n) is in bounds iff (m, n) < (M, N). The last
    # tile is partial whenever M or N is not a multiple of the tile size.
    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    # Load A and B into registers (predicated), compute C = A + B, store back.
    cute.copy(copy_atom, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom, thrB, frgB, pred=frgPred)

    frgC.store(frgA.load() + frgB.load())

    cute.copy(copy_atom, frgC, thrC, pred=frgPred)


@cute.jit
def launch_tiled_add(
    stream: cuda.CUstream,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    """Host-side JIT wrapper: tile the tensors, compute the grid, launch."""
    dtype = mA.element_type
    vector_size = 128 // dtype.width  # 4 for fp32: 128-bit vectorized copy

    # Thread layout: (4, 32) with 32 the fastest mode -> 128 threads, each
    # thread owns a 4x4 sub-block (value layout (4, vector_size)).
    thr_layout = cute.make_ordered_layout((4, THREADS // 4), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # ((TileM, TileN), (RestM, RestN)): tile modes plus remaining modes.
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    # Identity tensor carries the global coordinate of every element; slicing
    # it with the same tile coordinates yields per-element (m, n) coords.
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler_mn)

    tiled_add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )


def run(m: int, n: int, warmup_iterations: int, iterations: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Start the container with GPU access.")
    if m <= 0 or n <= 0:
        raise ValueError("--M and --N must be greater than zero")
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
    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    # DLPack creates zero-copy CuTe tensor views of PyTorch CUDA allocations.
    a_cute = from_dlpack(a, assumed_align=16)
    b_cute = from_dlpack(b, assumed_align=16)
    c_cute = from_dlpack(c, assumed_align=16)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    print(f"CTA tile: {TILE_M} x {TILE_N}, threads per CTA: {THREADS}")
    print("Compiling CuTe DSL kernel (the first run can take a few seconds)...")
    compile_start = time.perf_counter()
    compiled = cute.compile(launch_tiled_add, stream, a_cute, b_cute, c_cute)
    compile_time_s = time.perf_counter() - compile_start
    print(f"Compilation time: {compile_time_s:.3f} s")

    print("Launching kernel for correctness check...")
    compiled(stream, a_cute, b_cute, c_cute)
    torch_stream.synchronize()

    reference = a + b
    torch.testing.assert_close(c, reference, rtol=1e-5, atol=1e-6)

    max_error = (c - reference).abs().max().item()
    print(f"First 2x4 block of C: {c[:2, :4].tolist()}")
    print(f"Max absolute error: {max_error:.3e}")

    print(f"Warming up: {warmup_iterations} iterations...")
    for _ in range(warmup_iterations):
        compiled(stream, a_cute, b_cute, c_cute)
    torch_stream.synchronize()

    # CUDA events measure GPU elapsed time on the same stream used by CuTe DSL.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(torch_stream)
    for _ in range(iterations):
        compiled(stream, a_cute, b_cute, c_cute)
    end_event.record(torch_stream)
    end_event.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    average_time_ms = total_time_ms / iterations
    average_time_us = average_time_ms * 1_000.0

    # Each invocation reads A and B, then writes C: 3 float32 values/element.
    bytes_per_iteration = m * n * 3 * c.element_size()
    throughput_gbps = bytes_per_iteration / (average_time_ms / 1_000.0) / 1e9

    print(f"Benchmark iterations: {iterations}")
    print(f"Total GPU time: {total_time_ms:.3f} ms")
    print(f"Average kernel time: {average_time_us:.3f} us ({average_time_ms:.6f} ms)")
    print(f"Effective memory throughput: {throughput_gbps:.2f} GB/s")
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiled 2D matrix addition in CuTe DSL"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=1024,
        help="Number of rows; need not be a multiple of the tile size",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=1024,
        help="Number of columns; need not be a multiple of the tile size",
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
    run(args.M, args.N, args.warmup_iterations, args.iterations)
