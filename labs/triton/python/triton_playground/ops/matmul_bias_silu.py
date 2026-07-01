from __future__ import annotations

import torch
import torch.nn.functional as F

from triton_playground.kernels.matmul_bias_silu import launch_matmul_bias_silu


def matmul_bias_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    group_size_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """Compute SiLU(A @ B + bias) with a fused Triton matmul epilogue."""
    return launch_matmul_bias_silu(
        a,
        b,
        bias,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        group_size_m=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def torch_matmul_bias_silu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """PyTorch expression baseline for SiLU(A @ B + bias)."""
    return F.silu(torch.matmul(a, b) + bias)
