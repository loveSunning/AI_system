from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + bh * S * D
    k_base = k_ptr + bh * S * D
    v_base = v_ptr + bh * S * D
    out_base = out_ptr + bh * S * D

    q = tl.load(
        q_base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for kv_start in tl.range(0, S, BLOCK_N):
        offs_kv = kv_start + offs_n
        k = tl.load(
            k_base + offs_kv[None, :] * D + offs_d[:, None],
            mask=(offs_kv[None, :] < S) & (offs_d[:, None] < D),
            other=0.0,
        )
        v = tl.load(
            v_base + offs_kv[:, None] * D + offs_d[None, :],
            mask=(offs_kv[:, None] < S) & (offs_d[None, :] < D),
            other=0.0,
        )

        qk = tl.dot(q, k) * SCALE
        qk = tl.where(offs_kv[None, :] < S, qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_kv[None, :] <= offs_m[:, None], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]
    tl.store(
        out_base + offs_m[:, None] * D + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
    )


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _validate_fused_attention_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"fused_attention expects q/k/v as [B, H, S, D], got q shape {tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v must have the same shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("fused_attention requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"fused_attention supports float16 and float32, got {q.dtype}")
    if q.shape[2] <= 0 or q.shape[3] <= 0:
        raise ValueError("fused_attention requires non-empty S and D dimensions")
    return q.shape


def launch_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
    block_d: int | None = None,
) -> torch.Tensor:
    B, H, S, D = _validate_fused_attention_inputs(q, k, v)
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")
    if block_d is None:
        block_d = next_power_of_2(D)
    if block_d < D:
        raise ValueError(f"block_d must be >= D, got block_d={block_d}, D={D}")
    if block_d & (block_d - 1):
        raise ValueError("block_d must be a power of two")
    if scale is None:
        scale = D ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    grid = (triton.cdiv(S, block_m), B * H)
    _fused_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        S,
        D,
        float(scale),
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=4,
    )
    return out
