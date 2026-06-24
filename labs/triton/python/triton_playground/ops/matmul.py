from __future__ import annotations

import torch

from triton_playground.kernels.matmul import launch_matmul, launch_matmul_fixed


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with the autotuned Triton kernel."""
    return launch_matmul(a, b)


def matmul_fixed(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    group_size_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """Matrix multiplication with one fixed Triton tile configuration."""
    return launch_matmul_fixed(
        a,
        b,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        group_size_m=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
