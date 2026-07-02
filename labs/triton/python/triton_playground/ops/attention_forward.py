from __future__ import annotations

import math

import torch

from triton_playground.kernels.attention_forward import launch_stepwise_attention


def torch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference materialized attention: scores = QK^T, probs = softmax(scores), out = probs V."""
    if q.ndim != 4:
        raise ValueError(f"torch_attention expects q/k/v as [B, H, S, D], got q shape {tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v must have the same shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")

    _, _, S, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones((S, S), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None, None, :, :], -float("inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out, scores, probs


def triton_stepwise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
    block_d: int = 32,
    softmax_block: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stepwise Triton attention that materializes scores and probabilities."""
    return launch_stepwise_attention(
        q,
        k,
        v,
        causal=causal,
        scale=scale,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        softmax_block=softmax_block,
    )
