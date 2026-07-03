from __future__ import annotations

import torch

from triton_playground.kernels.fused_attention import (
    launch_fused_attention,
    launch_fused_attention_backward,
    launch_fused_attention_with_lse,
)
from triton_playground.ops.attention_forward import torch_attention


class _FlashAttention1Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        block_m: int,
        block_n: int,
        block_d: int | None,
        dropout_p: float,
        dropout_seed: int,
    ) -> torch.Tensor:
        out, lse = launch_fused_attention_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            block_m=block_m,
            block_n=block_n,
            block_d=block_d,
            dropout_p=dropout_p,
            dropout_seed=dropout_seed,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        ctx.scale = scale
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.block_d = block_d
        ctx.dropout_p = dropout_p
        ctx.dropout_seed = dropout_seed
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = launch_fused_attention_backward(
            q,
            k,
            v,
            out,
            lse,
            do,
            causal=ctx.causal,
            scale=ctx.scale,
            block_m=ctx.block_m,
            block_n=ctx.block_n,
            block_d=ctx.block_d,
            dropout_p=ctx.dropout_p,
            dropout_seed=ctx.dropout_seed,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def triton_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
    block_d: int | None = None,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
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
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
    )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
    block_d: int | None = None,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> torch.Tensor:
    """FlashAttention-1 style exact attention with Triton forward and backward."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _FlashAttention1Function.apply(
        q,
        k,
        v,
        causal,
        float(scale),
        block_m,
        block_n,
        block_d,
        float(dropout_p),
        int(dropout_seed),
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
