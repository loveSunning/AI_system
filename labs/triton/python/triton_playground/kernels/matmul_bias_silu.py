from __future__ import annotations

import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = (torch.float16, torch.float32)


@triton.jit
def _matmul_bias_silu_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    local_pid = pid % num_pid_in_group
    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    z = accumulator + bias[None, :]
    sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
    y = z * sigmoid_z

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def _validate_matmul_bias_silu_inputs(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul_bias_silu expects 2D a/b tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if bias.ndim != 1:
        raise ValueError(f"bias must be a 1D tensor, got shape {tuple(bias.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {tuple(a.shape)} x {tuple(b.shape)}")
    if bias.shape[0] != b.shape[1]:
        raise ValueError(f"bias must have shape ({b.shape[1]},), got {tuple(bias.shape)}")
    if not a.is_cuda or not b.is_cuda or not bias.is_cuda:
        raise ValueError("matmul_bias_silu requires CUDA tensors")
    if a.device != b.device or a.device != bias.device:
        raise ValueError("a, b, and bias must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype != bias.dtype:
        raise ValueError(f"a, b, and bias must have the same dtype, got {a.dtype}, {b.dtype}, {bias.dtype}")
    if a.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"matmul_bias_silu supports float16 and float32, got {a.dtype}")
    return a.shape[0], b.shape[1], a.shape[1]


def launch_matmul_bias_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    group_size_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    M, N, K = _validate_matmul_bias_silu_inputs(a, b, bias)
    if block_size_m <= 0 or block_size_n <= 0 or block_size_k <= 0:
        raise ValueError("block sizes must be positive")
    if group_size_m <= 0:
        raise ValueError("group_size_m must be positive")
    if num_warps <= 0 or num_stages <= 0:
        raise ValueError("num_warps and num_stages must be positive")

    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    y = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)
    _matmul_bias_silu_kernel[grid](
        a,
        b,
        bias,
        y,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y
