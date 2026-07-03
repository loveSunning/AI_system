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
    lse_ptr,
    dropout_p,
    dropout_seed,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
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

        p_for_v = p
        if ENABLE_DROPOUT:
            dropout_offsets = bh * S * S + offs_m[:, None] * S + offs_kv[None, :]
            keep = tl.rand(dropout_seed, dropout_offsets) > dropout_p
            keep = keep & (offs_kv[None, :] < S)
            if CAUSAL:
                keep = keep & (offs_kv[None, :] <= offs_m[:, None])
            p_for_v = tl.where(keep, p / (1.0 - dropout_p), 0.0)

        acc = acc * alpha[:, None] + tl.dot(p_for_v.to(tl.float32), v.to(tl.float32))
        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]
    tl.store(
        out_base + offs_m[:, None] * D + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
    )
    tl.store(lse_ptr + bh * S + offs_m, m_i + tl.log(l_i), mask=offs_m < S)


@triton.jit
def _fused_attention_bwd_preprocess_kernel(
    out_ptr,
    do_ptr,
    delta_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    base = bh * S * D

    out = tl.load(
        out_ptr + base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(out * do, axis=1)
    tl.store(delta_ptr + bh * S + offs_m, delta, mask=offs_m < S)


@triton.jit
def _fused_attention_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dropout_p,
    dropout_seed,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    start_m = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + bh * S * D
    k_base = k_ptr + bh * S * D
    v_base = v_ptr + bh * S * D
    do_base = do_ptr + bh * S * D
    dq_base = dq_ptr + bh * S * D

    q = tl.load(
        q_base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
        other=0.0,
    )
    do = tl.load(
        do_base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    lse = tl.load(lse_ptr + bh * S + offs_m, mask=offs_m < S, other=float("inf")).to(tl.float32)
    delta = tl.load(delta_ptr + bh * S + offs_m, mask=offs_m < S, other=0.0).to(tl.float32)

    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for kv_start in tl.range(0, S, BLOCK_N):
        offs_kv = kv_start + offs_n
        k_t = tl.load(
            k_base + offs_kv[None, :] * D + offs_d[:, None],
            mask=(offs_kv[None, :] < S) & (offs_d[:, None] < D),
            other=0.0,
        )
        k = tl.load(
            k_base + offs_kv[:, None] * D + offs_d[None, :],
            mask=(offs_kv[:, None] < S) & (offs_d[None, :] < D),
            other=0.0,
        )
        v_t = tl.load(
            v_base + offs_kv[None, :] * D + offs_d[:, None],
            mask=(offs_kv[None, :] < S) & (offs_d[:, None] < D),
            other=0.0,
        )

        qk = tl.dot(q, k_t) * SCALE
        valid = (offs_m[:, None] < S) & (offs_kv[None, :] < S)
        if CAUSAL:
            valid = valid & (offs_kv[None, :] <= offs_m[:, None])
        qk = tl.where(valid, qk, -float("inf"))

        p = tl.exp(qk - lse[:, None])
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(do, v_t)
        if ENABLE_DROPOUT:
            dropout_offsets = bh * S * S + offs_m[:, None] * S + offs_kv[None, :]
            keep = (tl.rand(dropout_seed, dropout_offsets) > dropout_p) & valid
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.float32), k.to(tl.float32)) * SCALE

    tl.store(
        dq_base + offs_m[:, None] * D + offs_d[None, :],
        dq,
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
    )


@triton.jit
def _fused_attention_bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    dropout_p,
    dropout_seed,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    start_n = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + bh * S * D
    k_base = k_ptr + bh * S * D
    v_base = v_ptr + bh * S * D
    do_base = do_ptr + bh * S * D
    dk_base = dk_ptr + bh * S * D
    dv_base = dv_ptr + bh * S * D

    k_t = tl.load(
        k_base + offs_n[None, :] * D + offs_d[:, None],
        mask=(offs_n[None, :] < S) & (offs_d[:, None] < D),
        other=0.0,
    )
    v_t = tl.load(
        v_base + offs_n[None, :] * D + offs_d[:, None],
        mask=(offs_n[None, :] < S) & (offs_d[:, None] < D),
        other=0.0,
    )

    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for q_start in tl.range(0, S, BLOCK_M):
        q_rows = q_start + offs_m
        q = tl.load(
            q_base + q_rows[:, None] * D + offs_d[None, :],
            mask=(q_rows[:, None] < S) & (offs_d[None, :] < D),
            other=0.0,
        )
        do = tl.load(
            do_base + q_rows[:, None] * D + offs_d[None, :],
            mask=(q_rows[:, None] < S) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(lse_ptr + bh * S + q_rows, mask=q_rows < S, other=float("inf")).to(tl.float32)
        delta = tl.load(delta_ptr + bh * S + q_rows, mask=q_rows < S, other=0.0).to(tl.float32)

        qk = tl.dot(q, k_t) * SCALE
        valid = (q_rows[:, None] < S) & (offs_n[None, :] < S)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_rows[:, None])
        qk = tl.where(valid, qk, -float("inf"))

        p = tl.exp(qk - lse[:, None])
        p = tl.where(valid, p, 0.0)
        p_for_v = p
        if ENABLE_DROPOUT:
            dropout_offsets = bh * S * S + q_rows[:, None] * S + offs_n[None, :]
            keep = (tl.rand(dropout_seed, dropout_offsets) > dropout_p) & valid
            p_for_v = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        dv += tl.dot(tl.trans(p_for_v).to(tl.float32), do)

        dp = tl.dot(do, v_t)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(tl.float32), q.to(tl.float32)) * SCALE

    tl.store(
        dk_base + offs_n[:, None] * D + offs_d[None, :],
        dk,
        mask=(offs_n[:, None] < S) & (offs_d[None, :] < D),
    )
    tl.store(
        dv_base + offs_n[:, None] * D + offs_d[None, :],
        dv,
        mask=(offs_n[:, None] < S) & (offs_d[None, :] < D),
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


def _validate_dropout(dropout_p: float) -> None:
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must satisfy 0 <= dropout_p < 1, got {dropout_p}")


def launch_fused_attention(
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
    out, _ = launch_fused_attention_with_lse(
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
    return out


def launch_fused_attention_with_lse(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = _validate_fused_attention_inputs(q, k, v)
    _validate_dropout(dropout_p)
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
    lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(S, block_m), B * H)
    _fused_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        float(dropout_p),
        int(dropout_seed),
        S,
        D,
        float(scale),
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        ENABLE_DROPOUT=dropout_p > 0.0,
        num_warps=4,
        num_stages=4,
    )
    return out, lse


def launch_fused_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
    block_d: int | None = None,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, D = _validate_fused_attention_inputs(q, k, v)
    _validate_dropout(dropout_p)
    if out.shape != q.shape or do.shape != q.shape:
        raise ValueError("out and do must have the same shape as q")
    if lse.shape != (B, H, S):
        raise ValueError(f"lse must have shape {(B, H, S)}, got {tuple(lse.shape)}")
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
    out = out.contiguous()
    lse = lse.contiguous()
    do = do.contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    bh = B * H

    q_grid = (triton.cdiv(S, block_m), bh)
    kv_grid = (triton.cdiv(S, block_n), bh)
    _fused_attention_bwd_preprocess_kernel[q_grid](
        out,
        do,
        delta,
        S,
        D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    _fused_attention_bwd_dq_kernel[q_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        float(dropout_p),
        int(dropout_seed),
        S,
        D,
        float(scale),
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        ENABLE_DROPOUT=dropout_p > 0.0,
        num_warps=4,
        num_stages=4,
    )
    _fused_attention_bwd_dkdv_kernel[kv_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        float(dropout_p),
        int(dropout_seed),
        S,
        D,
        float(scale),
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        ENABLE_DROPOUT=dropout_p > 0.0,
        num_warps=4,
        num_stages=4,
    )
    return dq, dk, dv
