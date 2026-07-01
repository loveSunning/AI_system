from __future__ import annotations

import torch

from triton_playground.kernels.rms_norm import launch_naive_rms_norm, launch_rms_norm


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    block_size: int | None = None,
    num_warps: int | None = None,
    group_size_m: int | None = None,
) -> torch.Tensor:
    """Apply production-style affine RMSNorm over the last dimension."""
    return launch_rms_norm(
        x,
        weight,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
        group_size_m=group_size_m,
    )


def naive_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Apply RMSNorm with a simple Triton backward path for comparison."""
    return launch_naive_rms_norm(
        x,
        weight,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
    )
