from __future__ import annotations

import torch
import triton
import triton.language as tl


MAX_FUSED_SIZE_BYTES = 65536
SUPPORTED_DTYPES = (torch.float16, torch.float32)


@triton.jit
def _layer_norm_fwd_fused(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x_row = x_ptr + row * stride
    y_row = y_ptr + row * stride

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias
    tl.store(y_row + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(
    dx_ptr,
    dy_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    lock_ptr,
    stride,
    n_cols,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < n_cols

    x_row = x_ptr + row * stride
    dy_row = dy_ptr + row * stride
    dx_row = dx_ptr + row * stride

    lock_id = row % GROUP_SIZE_M
    lock = lock_ptr + lock_id
    count = lock_ptr + GROUP_SIZE_M + lock_id
    partial_dw = partial_dw_ptr + lock_id * n_cols + cols
    partial_db = partial_db_ptr + lock_id * n_cols + cols

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = tl.where(mask, weight * dy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / n_cols
    c2 = tl.sum(wdy, axis=0) / n_cols
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(dx_row + cols, dx, mask=mask)

    row_dw = dy * xhat
    row_db = dy
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass

    is_first_writer = tl.load(count) == 0
    if is_first_writer:
        tl.atomic_xchg(count, 1)
    else:
        row_dw += tl.load(partial_dw, mask=mask, other=0.0)
        row_db += tl.load(partial_db, mask=mask, other=0.0)

    tl.store(partial_dw, row_dw, mask=mask)
    tl.store(partial_db, row_db, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(
    partial_dw_ptr,
    partial_db_ptr,
    final_dw_ptr,
    final_db_ptr,
    n_groups,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for offset in range(0, n_groups, BLOCK_SIZE_M):
        rows = offset + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < n_groups) & (cols[None, :] < n_cols)
        ptrs = rows[:, None] * n_cols + cols[None, :]
        dw += tl.load(partial_dw_ptr + ptrs, mask=mask, other=0.0)
        db += tl.load(partial_db_ptr + ptrs, mask=mask, other=0.0)

    tl.store(final_dw_ptr + cols, tl.sum(dw, axis=0), mask=cols < n_cols)
    tl.store(final_db_ptr + cols, tl.sum(db, axis=0), mask=cols < n_cols)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def max_fused_elements(dtype: torch.dtype) -> int:
    return MAX_FUSED_SIZE_BYTES // torch.empty((), dtype=dtype).element_size()


def default_num_warps(block_size: int) -> int:
    return min(max(block_size // 256, 1), 8)


def default_group_size_m(n_cols: int) -> int:
    if n_cols <= 1024:
        return 256
    if n_cols <= 4096:
        return 128
    if n_cols <= 8192:
        return 96
    return 64


def resolve_block_size(n_cols: int, dtype: torch.dtype, block_size: int | None) -> int:
    max_block_size = max_fused_elements(dtype)
    if block_size is None:
        block_size = min(max_block_size, next_power_of_2(n_cols))
    if block_size < n_cols:
        raise ValueError(f"block_size must be >= normalized dimension, got block_size={block_size}, n_cols={n_cols}")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    if block_size > max_block_size:
        raise ValueError(
            f"block_size={block_size} exceeds the 64KB fused-row limit for {dtype}; "
            f"max block_size is {max_block_size}"
        )
    return block_size


def _validate_layer_norm_inputs(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> int:
    if x.ndim < 1:
        raise ValueError("layer_norm expects at least one input dimension")
    if x.shape[-1] == 0:
        raise ValueError("layer_norm requires a non-empty normalized dimension")
    if x.numel() == 0:
        raise ValueError("layer_norm requires a non-empty input tensor")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("layer_norm requires CUDA tensors")
    if x.device != weight.device or x.device != bias.device:
        raise ValueError("x, weight, and bias must be on the same CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("layer_norm expects contiguous x, weight, and bias tensors")
    if x.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"layer_norm supports float16 and float32 inputs, got {x.dtype}")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise ValueError(f"x, weight, and bias must have the same dtype, got {x.dtype}, {weight.dtype}, {bias.dtype}")

    n_cols = x.shape[-1]
    expected_shape = (n_cols,)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(f"weight must have shape {expected_shape}, got {tuple(weight.shape)}")
    if tuple(bias.shape) != expected_shape:
        raise ValueError(f"bias must have shape {expected_shape}, got {tuple(bias.shape)}")
    if n_cols > max_fused_elements(x.dtype):
        raise ValueError(
            f"layer_norm supports normalized dimensions up to 64KB per row; "
            f"got n_cols={n_cols} for dtype={x.dtype}"
        )
    return n_cols


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        block_size: int | None,
        num_warps: int | None,
        group_size_m: int | None,
    ) -> torch.Tensor:
        n_cols = _validate_layer_norm_inputs(x, weight, bias)
        block_size = resolve_block_size(n_cols, x.dtype, block_size)
        if num_warps is None:
            num_warps = default_num_warps(block_size)
        elif num_warps <= 0:
            raise ValueError(f"num_warps must be positive, got {num_warps}")
        if group_size_m is None:
            group_size_m = default_group_size_m(n_cols)
        elif group_size_m <= 0:
            raise ValueError(f"group_size_m must be positive, got {group_size_m}")

        x_2d = x.reshape(-1, n_cols)
        y = torch.empty_like(x)
        y_2d = y.reshape(-1, n_cols)
        n_rows = x_2d.shape[0]
        mean = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

        _layer_norm_fwd_fused[(n_rows,)](
            x_2d,
            y_2d,
            weight,
            bias,
            mean,
            rstd,
            x_2d.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.group_size_m = group_size_m
        ctx.n_cols = n_cols
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        del bias
        n_cols = ctx.n_cols
        group_size_m = ctx.group_size_m

        dy = dy.contiguous()
        x_2d = x.reshape(-1, n_cols)
        dy_2d = dy.reshape(-1, n_cols)
        dx = torch.empty_like(x)
        dx_2d = dx.reshape(-1, n_cols)
        n_rows = x_2d.shape[0]

        locks = torch.zeros((2 * group_size_m,), device=x.device, dtype=torch.int32)
        partial_dw = torch.zeros((group_size_m, n_cols), device=x.device, dtype=torch.float32)
        partial_db = torch.zeros((group_size_m, n_cols), device=x.device, dtype=torch.float32)
        dw = torch.empty_like(weight)
        db = torch.empty_like(weight)

        _layer_norm_bwd_dx_fused[(n_rows,)](
            dx_2d,
            dy_2d,
            partial_dw,
            partial_db,
            x_2d,
            weight,
            mean,
            rstd,
            locks,
            x_2d.stride(0),
            n_cols,
            GROUP_SIZE_M=group_size_m,
            BLOCK_SIZE_N=ctx.block_size,
            num_warps=ctx.num_warps,
        )

        grid = (triton.cdiv(n_cols, 128),)
        _layer_norm_bwd_dwdb[grid](
            partial_dw,
            partial_db,
            dw,
            db,
            min(group_size_m, n_rows),
            n_cols,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
        )
        return dx, dw, db, None, None, None, None


def launch_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    block_size: int | None = None,
    num_warps: int | None = None,
    group_size_m: int | None = None,
) -> torch.Tensor:
    return _LayerNorm.apply(x, weight, bias, eps, block_size, num_warps, group_size_m)
