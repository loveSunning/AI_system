from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_playground.kernels.fused_softmax import default_num_warps


@triton.jit
def _online_softmax_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start = row_id * n_cols

    running_max = -float("inf")
    running_sum = 0.0

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols
        block = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(block, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        block_sum = tl.sum(tl.exp(block - new_max), axis=0)
        running_sum = running_sum * old_scale + block_sum
        running_max = new_max

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols
        block = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
        out = tl.exp(block - running_max) / running_sum
        tl.store(out_ptr + row_start + cols, out, mask=mask)


def _validate_online_softmax_input(x: torch.Tensor) -> tuple[int, int]:
    if x.ndim != 2:
        raise ValueError(f"online_softmax expects a 2D tensor, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("online_softmax requires a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("online_softmax expects a contiguous tensor")
    if x.shape[1] == 0:
        raise ValueError("online_softmax requires at least one column")
    return x.shape


def launch_online_softmax(
    x: torch.Tensor,
    block_size: int = 1024,
    num_warps: int | None = None,
) -> torch.Tensor:
    n_rows, n_cols = _validate_online_softmax_input(x)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    if num_warps is None:
        num_warps = default_num_warps(block_size)
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")

    out = torch.empty_like(x)
    _online_softmax_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
