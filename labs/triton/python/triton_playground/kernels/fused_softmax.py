from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_softmax_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row_id * n_cols

    row = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf"))
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax = numerator / denominator

    tl.store(out_ptr + row_start + offsets, softmax, mask=mask)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def default_num_warps(block_size: int) -> int:
    if block_size >= 4096:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def launch_fused_softmax(
    x: torch.Tensor,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"fused_softmax expects a 2D tensor, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("fused_softmax requires a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("fused_softmax expects a contiguous tensor")
    if x.shape[1] == 0:
        raise ValueError("fused_softmax requires at least one column")

    n_rows, n_cols = x.shape
    if block_size is None:
        block_size = next_power_of_2(n_cols)
    if block_size < n_cols:
        raise ValueError(f"block_size must be >= n_cols, got block_size={block_size}, n_cols={n_cols}")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    if num_warps is None:
        num_warps = default_num_warps(block_size)

    out = torch.empty_like(x)
    _fused_softmax_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
