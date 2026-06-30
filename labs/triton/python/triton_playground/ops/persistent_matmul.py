from __future__ import annotations

import torch

from triton_playground.kernels.persistent_matmul import launch_persistent_matmul, launch_persistent_matmul_fixed


def persistent_matmul(a: torch.Tensor, b: torch.Tensor, num_sms: int | None = None) -> torch.Tensor:
    """Matrix multiplication with an autotuned non-TMA persistent Triton kernel."""
    return launch_persistent_matmul(a, b, num_sms=num_sms)


def persistent_matmul_fixed(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    group_size_m: int = 8,
    num_sms: int | None = None,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """Matrix multiplication with one 4090D-friendly persistent tile configuration."""
    return launch_persistent_matmul_fixed(
        a,
        b,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        group_size_m=group_size_m,
        num_sms=num_sms,
        num_warps=num_warps,
        num_stages=num_stages,
    )
