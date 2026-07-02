from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tile = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)

    num_tiles_n = tl.cdiv(S, BLOCK_N)
    pid_m = pid_tile // num_tiles_n
    pid_n = pid_tile % num_tiles_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + bh * S * D
    k_base = k_ptr + bh * S * D
    scores_base = scores_ptr + bh * S * S

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d_start in tl.range(0, D, BLOCK_D):
        d = d_start + offs_d
        q_ptrs = q_base + offs_m[:, None] * D + d[None, :]
        k_ptrs = k_base + offs_n[None, :] * D + d[:, None]

        q = tl.load(q_ptrs, mask=(offs_m[:, None] < S) & (d[None, :] < D), other=0.0)
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < S) & (d[:, None] < D), other=0.0)
        acc += tl.dot(q, k)

    acc = acc * SCALE
    if CAUSAL:
        causal_mask = offs_n[None, :] <= offs_m[:, None]
        acc = tl.where(causal_mask, acc, -float("inf"))

    score_ptrs = scores_base + offs_m[:, None] * S + offs_n[None, :]
    score_mask = (offs_m[:, None] < S) & (offs_n[None, :] < S)
    tl.store(score_ptrs, acc, mask=score_mask)


@triton.jit
def _attention_softmax_kernel(scores_ptr, probs_ptr, S: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    bh = row_id // S
    row = row_id - bh * S

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < S

    scores_base = scores_ptr + bh * S * S + row * S
    probs_base = probs_ptr + bh * S * S + row * S

    x = tl.load(scores_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    p = tl.exp(x)
    denom = tl.sum(p, axis=0)
    y = p / denom

    tl.store(probs_base + cols, y, mask=mask)


@triton.jit
def _pv_out_kernel(
    probs_ptr,
    v_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_tile = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)

    num_tiles_n = tl.cdiv(D, BLOCK_N)
    pid_m = pid_tile // num_tiles_n
    pid_n = pid_tile % num_tiles_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    probs_base = probs_ptr + bh * S * S
    v_base = v_ptr + bh * S * D
    out_base = out_ptr + bh * S * D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, S, BLOCK_K):
        k = k_start + offs_k
        probs_ptrs = probs_base + offs_m[:, None] * S + k[None, :]
        v_ptrs = v_base + k[:, None] * D + offs_n[None, :]

        probs = tl.load(probs_ptrs, mask=(offs_m[:, None] < S) & (k[None, :] < S), other=0.0)
        v = tl.load(v_ptrs, mask=(k[:, None] < S) & (offs_n[None, :] < D), other=0.0)
        acc += tl.dot(probs, v)

    out_ptrs = out_base + offs_m[:, None] * D + offs_n[None, :]
    out_mask = (offs_m[:, None] < S) & (offs_n[None, :] < D)
    tl.store(out_ptrs, acc, mask=out_mask)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def default_softmax_num_warps(block_size: int) -> int:
    if block_size >= 4096:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _validate_attention_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"attention expects q/k/v as [B, H, S, D], got q shape {tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v must have the same shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("attention requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"attention supports float16 and float32, got {q.dtype}")
    if q.shape[2] <= 0 or q.shape[3] <= 0:
        raise ValueError("attention requires non-empty S and D dimensions")
    return q.shape


def launch_stepwise_attention(
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
    B, H, S, D = _validate_attention_inputs(q, k, v)
    if block_m <= 0 or block_n <= 0 or block_d <= 0:
        raise ValueError("block_m, block_n, and block_d must be positive")

    if scale is None:
        scale = D ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    scores = torch.empty((B, H, S, S), device=q.device, dtype=torch.float32)
    probs = torch.empty((B, H, S, S), device=q.device, dtype=q.dtype)
    out = torch.empty((B, H, S, D), device=q.device, dtype=q.dtype)

    bh = B * H
    grid_scores = (triton.cdiv(S, block_m) * triton.cdiv(S, block_n), bh)
    _qk_scores_kernel[grid_scores](
        q,
        k,
        scores,
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

    if softmax_block is None:
        softmax_block = next_power_of_2(S)
    if softmax_block < S:
        raise ValueError(f"softmax_block must be >= S, got softmax_block={softmax_block}, S={S}")
    if softmax_block & (softmax_block - 1):
        raise ValueError("softmax_block must be a power of two")

    _attention_softmax_kernel[(bh * S,)](
        scores,
        probs,
        S,
        BLOCK_SIZE=softmax_block,
        num_warps=default_softmax_num_warps(softmax_block),
    )

    grid_out = (triton.cdiv(S, block_m) * triton.cdiv(D, block_n), bh)
    _pv_out_kernel[grid_out](
        probs,
        v,
        out,
        S,
        D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_d,
        num_warps=4,
        num_stages=4,
    )

    return out, scores, probs
