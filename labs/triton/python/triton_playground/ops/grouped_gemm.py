from __future__ import annotations

import torch

from triton_playground.kernels.grouped_gemm import launch_grouped_gemm


def grouped_gemm(
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    num_sms: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> list[torch.Tensor]:
    """Run a group of independent float16 GEMMs with one Triton kernel launch."""
    return launch_grouped_gemm(
        group_a,
        group_b,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        num_sms=num_sms,
        num_warps=num_warps,
        num_stages=num_stages,
    )
