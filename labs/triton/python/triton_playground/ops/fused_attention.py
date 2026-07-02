from __future__ import annotations

import torch

from triton_playground.kernels.fused_attention import launch_fused_attention
from triton_playground.ops.attention_forward import torch_attention


def triton_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
    block_d: int | None = None,
) -> torch.Tensor:
    """Fused attention forward with online softmax, without materializing scores/probs."""
    return launch_fused_attention(
        q,
        k,
        v,
        causal=causal,
        scale=scale,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
    )


def torch_fused_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """PyTorch materialized reference for fused attention forward."""
    return torch_attention(q, k, v, causal=causal, scale=scale)[0]
