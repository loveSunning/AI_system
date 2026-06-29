from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    group_lds,
    GROUP_SIZE: tl.constexpr,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(axis=0)
    last_problem_end = 0

    for group_id in range(GROUP_SIZE):
        m = tl.load(group_gemm_sizes + group_id * 3)
        n = tl.load(group_gemm_sizes + group_id * 3 + 1)
        k = tl.load(group_gemm_sizes + group_id * 3 + 2)
        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(n, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m = tile_idx_in_gemm // num_n_tiles
            tile_n = tile_idx_in_gemm % num_n_tiles

            lda = tl.load(group_lds + group_id * 3)
            ldb = tl.load(group_lds + group_id * 3 + 1)
            ldc = tl.load(group_lds + group_id * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + group_id).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + group_id).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + group_id).to(tl.pointer_type(tl.float16))

            offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k_tile in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                k_remaining = k - k_tile * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < n),
                    other=0.0,
                )
                accumulator = tl.dot(a, b, accumulator)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb

            c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
            tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))
            tile_idx += NUM_SM

        last_problem_end += num_tiles


def default_num_sms(device: torch.device | int | None = None) -> int:
    if not torch.cuda.is_available():
        return 1
    return torch.cuda.get_device_properties(device if device is not None else "cuda").multi_processor_count


def _validate_grouped_gemm_inputs(group_a: list[torch.Tensor], group_b: list[torch.Tensor]) -> None:
    if len(group_a) != len(group_b):
        raise ValueError(f"group_a and group_b must have the same length, got {len(group_a)} and {len(group_b)}")
    if not group_a:
        raise ValueError("grouped_gemm requires at least one GEMM problem")

    device = group_a[0].device
    dtype = group_a[0].dtype
    for index, (a, b) in enumerate(zip(group_a, group_b)):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"grouped_gemm expects 2D tensors, got {tuple(a.shape)} and {tuple(b.shape)} at index {index}")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"grouped_gemm shape mismatch at index {index}: {tuple(a.shape)} x {tuple(b.shape)}")
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("grouped_gemm requires CUDA tensors")
        if a.device != device or b.device != device:
            raise ValueError("all grouped_gemm tensors must be on the same CUDA device")
        if a.dtype != dtype or b.dtype != dtype:
            raise ValueError("all grouped_gemm tensors must have the same dtype")
        if a.dtype != torch.float16:
            raise ValueError(f"grouped_gemm currently supports float16 tensors, got {a.dtype}")
        if not a.is_contiguous() or not b.is_contiguous():
            raise ValueError("grouped_gemm expects contiguous input tensors")
        if a.shape[0] == 0 or a.shape[1] == 0 or b.shape[1] == 0:
            raise ValueError("grouped_gemm requires non-empty M, N, and K dimensions")


def _validate_config(block_size_m: int, block_size_n: int, block_size_k: int, num_sms: int) -> None:
    for name, value in {
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "block_size_k": block_size_k,
        "num_sms": num_sms,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in {
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "block_size_k": block_size_k,
    }.items():
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two, got {value}")


def _make_metadata(group_a: list[torch.Tensor], group_b: list[torch.Tensor], group_c: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = group_a[0].device
    a_addrs: list[int] = []
    b_addrs: list[int] = []
    c_addrs: list[int] = []
    sizes: list[int] = []
    lds: list[int] = []

    for a, b, c in zip(group_a, group_b, group_c):
        m, k = a.shape
        _, n = b.shape
        a_addrs.append(a.data_ptr())
        b_addrs.append(b.data_ptr())
        c_addrs.append(c.data_ptr())
        sizes.extend([m, n, k])
        lds.extend([a.stride(0), b.stride(0), c.stride(0)])

    d_a_ptrs = torch.tensor(a_addrs, device=device, dtype=torch.int64)
    d_b_ptrs = torch.tensor(b_addrs, device=device, dtype=torch.int64)
    d_c_ptrs = torch.tensor(c_addrs, device=device, dtype=torch.int64)
    d_sizes = torch.tensor(sizes, device=device, dtype=torch.int32)
    d_lds = torch.tensor(lds, device=device, dtype=torch.int32)
    return d_a_ptrs, d_b_ptrs, d_c_ptrs, d_sizes, d_lds


def launch_grouped_gemm(
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    num_sms: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> list[torch.Tensor]:
    _validate_grouped_gemm_inputs(group_a, group_b)
    if num_sms is None:
        num_sms = default_num_sms(group_a[0].device)
    _validate_config(block_size_m, block_size_n, block_size_k, num_sms)

    group_c = [torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype) for a, b in zip(group_a, group_b)]
    d_a_ptrs, d_b_ptrs, d_c_ptrs, d_sizes, d_lds = _make_metadata(group_a, group_b, group_c)

    _grouped_gemm_kernel[(num_sms,)](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_sizes,
        d_lds,
        GROUP_SIZE=len(group_a),
        NUM_SM=num_sms,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return group_c
