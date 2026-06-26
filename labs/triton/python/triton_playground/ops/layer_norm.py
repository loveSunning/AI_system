from __future__ import annotations

import torch

from triton_playground.kernels.layer_norm import launch_layer_norm


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    block_size: int | None = None,
    num_warps: int | None = None,
    group_size_m: int | None = None,
) -> torch.Tensor:
    """Apply affine LayerNorm over the last dimension with Triton forward/backward kernels."""
    return launch_layer_norm(
        x,
        weight,
        bias,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
        group_size_m=group_size_m,
    )
