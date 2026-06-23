from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def launch_vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise ValueError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("vector_add requires CUDA tensors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("vector_add expects contiguous tensors")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, block_size),)
    _vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out
