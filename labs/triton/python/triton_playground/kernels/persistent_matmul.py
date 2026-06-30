from __future__ import annotations

import torch
import triton
import triton.language as tl


def get_4090d_friendly_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
    ]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _persistent_matmul_kernel_body(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k_tile * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k_mask[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k_mask[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c = accumulator.to(tl.float32) if c_ptr.dtype.element_ty == tl.float32 else accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(configs=get_4090d_friendly_configs(), key=["M", "N", "K"])
@triton.jit
def _persistent_matmul_autotuned_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    _persistent_matmul_kernel_body(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        NUM_SMS,
    )


@triton.jit
def _persistent_matmul_fixed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    _persistent_matmul_kernel_body(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        NUM_SMS,
    )


def default_num_sms(device: torch.device | int | None = None) -> int:
    if not torch.cuda.is_available():
        return 1
    return torch.cuda.get_device_properties(device if device is not None else "cuda").multi_processor_count


def is_likely_4090d(device: torch.device | int | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(device if device is not None else "cuda")
    return props.major == 8 and props.minor == 9 and "4090" in props.name


def _validate_matmul_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"persistent_matmul expects 2D tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"persistent_matmul shape mismatch: {tuple(a.shape)} x {tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("persistent_matmul requires CUDA tensors")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    if a.dtype != b.dtype:
        raise ValueError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}")
    if a.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"persistent_matmul supports float16 and float32, got {a.dtype}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("persistent_matmul expects contiguous input tensors")
    if a.shape[0] == 0 or a.shape[1] == 0 or b.shape[1] == 0:
        raise ValueError("persistent_matmul requires non-empty M, N, and K dimensions")
    return a.shape[0], b.shape[1], a.shape[1]


def _validate_config(
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_sms: int,
    num_warps: int,
    num_stages: int,
) -> None:
    for name, value in {
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "block_size_k": block_size_k,
        "group_size_m": group_size_m,
        "num_sms": num_sms,
        "num_warps": num_warps,
        "num_stages": num_stages,
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


def _launch_kernel(
    kernel,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_sms: int,
    **kwargs,
) -> torch.Tensor:
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    grid = lambda META: (min(num_sms, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)
    kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
        **kwargs,
    )
    return c


def launch_persistent_matmul(a: torch.Tensor, b: torch.Tensor, num_sms: int | None = None) -> torch.Tensor:
    _validate_matmul_inputs(a, b)
    if num_sms is None:
        num_sms = default_num_sms(a.device)
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    return _launch_kernel(_persistent_matmul_autotuned_kernel, a, b, c, num_sms)


def launch_persistent_matmul_fixed(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    group_size_m: int = 8,
    num_sms: int | None = None,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    _validate_matmul_inputs(a, b)
    if num_sms is None:
        num_sms = default_num_sms(a.device)
    _validate_config(block_size_m, block_size_n, block_size_k, group_size_m, num_sms, num_warps, num_stages)
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    return _launch_kernel(
        _persistent_matmul_fixed_kernel,
        a,
        b,
        c,
        num_sms,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
