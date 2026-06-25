from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dropout_with_mask_kernel(x_ptr, keep_ptr, out_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0) != 0
    out = tl.where(keep, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _seeded_dropout_kernel(x_ptr, out_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    random = tl.rand(seed, offsets)
    keep = random > p
    out = tl.where(keep, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def _validate_p(p: float) -> None:
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must satisfy 0 <= p < 1, got {p}")


def _validate_x(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("dropout requires a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("dropout expects a contiguous tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"dropout supports float16 and float32, got {x.dtype}")


def launch_dropout_with_mask(x: torch.Tensor, keep_mask: torch.Tensor, p: float, block_size: int = 1024) -> torch.Tensor:
    _validate_p(p)
    _validate_x(x)
    if keep_mask.shape != x.shape:
        raise ValueError(f"keep_mask must have shape {tuple(x.shape)}, got {tuple(keep_mask.shape)}")
    if not keep_mask.is_cuda:
        raise ValueError("keep_mask must be a CUDA tensor")
    if not keep_mask.is_contiguous():
        raise ValueError("keep_mask must be contiguous")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, block_size),)
    _dropout_with_mask_kernel[grid](x, keep_mask, out, n_elements, p, BLOCK_SIZE=block_size)
    return out


def launch_seeded_dropout(x: torch.Tensor, p: float, seed: int, block_size: int = 1024) -> torch.Tensor:
    _validate_p(p)
    _validate_x(x)
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, block_size),)
    _seeded_dropout_kernel[grid](x, out, n_elements, p, seed, BLOCK_SIZE=block_size)
    return out
