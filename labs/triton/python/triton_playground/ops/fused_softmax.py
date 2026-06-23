from __future__ import annotations

import torch

from triton_playground.kernels.fused_softmax import launch_fused_softmax


def fused_softmax(
    x: torch.Tensor,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Apply softmax along the last dimension with a row-wise Triton kernel."""
    return launch_fused_softmax(x, block_size=block_size, num_warps=num_warps)
