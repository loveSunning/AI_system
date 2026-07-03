from __future__ import annotations

import torch

from triton_playground.kernels.flash_attention_v2 import (
    flash_attention_v2_feature_reason,
    flash_attention_v2_is_available,
    launch_flash_attention_v2,
    launch_flash_attention_v2_backward,
    launch_flash_attention_v2_with_lse,
)
from triton_playground.ops.attention_forward import torch_attention


class _FlashAttentionV2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        warp_specialize: bool,
    ) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out, lse = launch_flash_attention_v2_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            warp_specialize=warp_specialize,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = launch_flash_attention_v2_backward(
            q,
            k,
            v,
            out,
            lse,
            do,
            causal=ctx.causal,
            scale=ctx.scale,
        )
        return dq, dk, dv, None, None, None


def triton_flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    warp_specialize: bool = False,
) -> torch.Tensor:
    """Forward-only FlashAttention v2 path adapted from the Triton tutorial."""
    return launch_flash_attention_v2(
        q,
        k,
        v,
        causal=causal,
        scale=scale,
        warp_specialize=warp_specialize,
    )


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    warp_specialize: bool = False,
) -> torch.Tensor:
    """Autograd FlashAttention v2 path with Triton forward and backward kernels."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _FlashAttentionV2Function.apply(
        q,
        k,
        v,
        causal,
        float(scale),
        bool(warp_specialize),
    )


def torch_flash_attention_v2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """PyTorch materialized reference for FlashAttention v2 correctness checks."""
    return torch_attention(q, k, v, causal=causal, scale=scale)[0]


__all__ = [
    "flash_attention_v2",
    "flash_attention_v2_feature_reason",
    "flash_attention_v2_is_available",
    "torch_flash_attention_v2_reference",
    "triton_flash_attention_v2",
]
