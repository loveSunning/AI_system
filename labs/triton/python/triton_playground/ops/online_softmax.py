from __future__ import annotations

import torch

from triton_playground.kernels.online_softmax import launch_online_softmax


def online_softmax(
    x: torch.Tensor,
    block_size: int = 1024,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Apply row-wise softmax with a block-streaming Triton online algorithm."""
    return launch_online_softmax(x, block_size=block_size, num_warps=num_warps)


def torch_online_softmax(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """PyTorch implementation of the two-pass online softmax recurrence."""
    if x.ndim != 2:
        raise ValueError(f"torch_online_softmax expects a 2D tensor, got shape {tuple(x.shape)}")
    if x.shape[1] == 0:
        raise ValueError("torch_online_softmax requires at least one column")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_rows, n_cols = x.shape
    running_max = torch.full((n_rows,), -float("inf"), device=x.device, dtype=torch.float32)
    running_sum = torch.zeros((n_rows,), device=x.device, dtype=torch.float32)

    for col_start in range(0, n_cols, block_size):
        block = x[:, col_start : col_start + block_size].to(torch.float32)
        block_max = torch.max(block, dim=-1).values
        new_max = torch.maximum(running_max, block_max)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.sum(torch.exp(block - new_max[:, None]), dim=-1)
        running_max = new_max

    out = torch.empty_like(x)
    for col_start in range(0, n_cols, block_size):
        block = x[:, col_start : col_start + block_size].to(torch.float32)
        out[:, col_start : col_start + block_size] = torch.exp(block - running_max[:, None]) / running_sum[:, None]

    return out
