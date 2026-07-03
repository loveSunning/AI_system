from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:  # pragma: no cover - depends on Triton version.
    TensorDescriptor = None


_CONFIG_PRE_HOOK_SUPPORTED = True


def _make_config(kwargs: dict[str, int], num_stages: int, num_warps: int) -> triton.Config:
    global _CONFIG_PRE_HOOK_SUPPORTED
    try:
        return triton.Config(kwargs, num_stages=num_stages, num_warps=num_warps, pre_hook=_host_descriptor_pre_hook)
    except TypeError:
        _CONFIG_PRE_HOOK_SUPPORTED = False
        return triton.Config(kwargs, num_stages=num_stages, num_warps=num_warps)


def _current_backend() -> str | None:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        return None


def _is_cuda() -> bool:
    return _current_backend() == "cuda"


def _is_hip() -> bool:
    return _current_backend() == "hip"


def _is_blackwell() -> bool:
    return _is_cuda() and torch.cuda.get_device_capability()[0] == 10


def _is_hopper() -> bool:
    return _is_cuda() and torch.cuda.get_device_capability()[0] == 9


def _supports_host_descriptor() -> bool:
    return (
        TensorDescriptor is not None
        and _CONFIG_PRE_HOOK_SUPPORTED
        and _is_cuda()
        and torch.cuda.get_device_capability()[0] >= 9
    )


def flash_attention_v2_feature_reason() -> str | None:
    if not hasattr(tl, "make_tensor_descriptor") or not hasattr(tl, "tensor_descriptor"):
        return "this FlashAttention v2 path requires Triton tensor descriptor APIs"
    return None


def flash_attention_v2_is_available() -> bool:
    return flash_attention_v2_feature_reason() is None


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            qk = qk - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)

        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            bm: tl.constexpr = acc.shape[0]
            bn: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([bm, 2, bn // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([bm, bn])
        else:
            acc = acc * alpha[:, None]

        v = desc_v.load([offsetv_y, 0])
        acc = tl.dot(p.to(dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    if TensorDescriptor is None or not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    head_dim = nargs["HEAD_DIM"]
    nargs["desc_q"].block_shape = [block_m, head_dim]
    nargs["desc_k"].block_shape = [block_n, head_dim]
    nargs["desc_v"].block_shape = [block_n, head_dim]
    nargs["desc_o"].block_shape = [block_m, head_dim]


_NUM_STAGES_OPTIONS = [1] if _is_hip() else [2, 3, 4]
_CONFIGS = [
    _make_config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=s, num_warps=w)
    for bm in [64, 128]
    for bn in [16, 32, 64, 128]
    for s in _NUM_STAGES_OPTIONS
    for w in [4, 8]
]
if "PYTEST_VERSION" in os.environ:
    _CONFIGS = [_make_config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4)]


def _keep_config(conf: triton.Config) -> bool:
    block_m = conf.kwargs["BLOCK_M"]
    block_n = conf.kwargs["BLOCK_N"]
    if _is_cuda() and torch.cuda.get_device_capability()[0] == 9 and block_m * block_n < 128 * 128:
        return conf.num_warps != 8
    return True


def _prune_invalid_configs(configs, named_args, **kwargs):
    n_ctx = kwargs["N_CTX"]
    head_dim = kwargs["HEAD_DIM"]
    stage = kwargs["STAGE"]
    pruned = []
    for conf in configs:
        block_m = conf.kwargs["BLOCK_M"]
        block_n = conf.kwargs["BLOCK_N"]
        if block_m > n_ctx or block_n > head_dim:
            continue
        if n_ctx % block_m != 0 or n_ctx % block_n != 0:
            continue
        if stage != 1 and block_m < block_n:
            continue
        pruned.append(conf)
    return pruned


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=list(filter(_keep_config, _CONFIGS)),
    key=["N_CTX", "HEAD_DIM", "warp_specialize"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, [y_dim, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, [y_dim, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, [y_dim, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, [y_dim, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.4426950408889634
    q = desc_q.load([qo_offset_y, 0])

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX + offs_m, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    k,
    v,
    sm_scale,
    DO,
    M,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

    curr_m = start_m
    step_m = BLOCK_M1
    for _ in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs)
        dv += tl.dot(pT.to(tl.float16), do)

        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk += tl.dot(dsT.to(tl.float16), tl.trans(qT))

        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,
    do,
    m,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

    curr_n = start_n
    step_n = BLOCK_N2
    for _ in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(tl.float16), tl.trans(kT))

        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    start_m = 0
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,
            Q,
            k,
            v,
            sm_scale,
            DO,
            M,
            D,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            MASK_BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,
            start_n,
            start_m,
            num_steps,
            MASK=True,
        )
        start_m += num_steps * MASK_BLOCK_M1

    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        start_m,
        num_steps,
        MASK=False,
    )

    tl.store(DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, dv)
    dk *= sm_scale
    tl.store(DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, dk)

    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    m = tl.load(M + offs_m)[:, None]

    if CAUSAL:
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq,
            q,
            K,
            V,
            do,
            m,
            D,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            BLOCK_M2,
            MASK_BLOCK_N2,
            HEAD_DIM,
            start_m,
            end_n - num_steps * MASK_BLOCK_N2,
            num_steps,
            MASK=True,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,
        start_m,
        start_n,
        num_steps,
        MASK=False,
    )

    dq *= LN2
    tl.store(DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, dq)


def _validate_flash_attention_v2_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int]:
    reason = flash_attention_v2_feature_reason()
    if reason is not None:
        raise RuntimeError(reason)
    if q.ndim != 4:
        raise ValueError(f"flash_attention_v2 expects q/k/v as [B, H, S, D], got q shape {tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v must have the same shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("flash_attention_v2 requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("flash_attention_v2 currently supports float16 q/k/v")
    B, H, S, D = q.shape
    if D not in {16, 32, 64, 128, 256}:
        raise ValueError(f"head dim must be one of 16, 32, 64, 128, 256, got {D}")
    if S < 128 or S % 128 != 0:
        raise ValueError(f"sequence length must be a multiple of 128 for this v2 backward path, got {S}")
    return B, H, S, D


def launch_flash_attention_v2_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    warp_specialize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = _validate_flash_attention_v2_inputs(q, k, v)
    if scale is None:
        scale = D ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    stage = 3 if causal else 1

    desc_q = q
    desc_k = k
    desc_v = v
    desc_o = out
    if _supports_host_descriptor() and not (_is_hopper() and warp_specialize):
        y_dim = B * H * S
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, D], strides=[D, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, D], strides=[D, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[y_dim, D], strides=[D, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(out, shape=[y_dim, D], strides=[D, 1], block_shape=dummy_block)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device=q.device)

    triton.set_allocator(alloc_fn)

    def grid(meta):
        return (triton.cdiv(S, meta["BLOCK_M"]), B * H, 1)

    extra_kernel_args = {}
    if _is_hip():
        extra_kernel_args = {"waves_per_eu": 3 if D <= 64 else 2, "allow_flush_denorm": True}
    if _is_blackwell() and warp_specialize:
        extra_kernel_args["maxnreg"] = 168 if D == 128 else 80

    _attn_fwd[grid](
        float(scale),
        lse,
        B,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=S,
        HEAD_DIM=D,
        STAGE=stage,
        warp_specialize=warp_specialize,
        IS_HOPPER=_is_hopper(),
        **extra_kernel_args,
    )
    return out, lse


def launch_flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    warp_specialize: bool = False,
) -> torch.Tensor:
    out, _ = launch_flash_attention_v2_with_lse(q, k, v, causal=causal, scale=scale, warp_specialize=warp_specialize)
    return out


def launch_flash_attention_v2_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, D = _validate_flash_attention_v2_inputs(q, k, v)
    if out.shape != q.shape or do.shape != q.shape:
        raise ValueError("out and do must have the same shape as q")
    if lse.shape != (B, H, S):
        raise ValueError(f"lse must have shape {(B, H, S)}, got {tuple(lse.shape)}")
    if scale is None:
        scale = D ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    do = do.contiguous()
    lse = lse.contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty_like(lse)

    pre_block = 128
    pre_grid = (S // pre_block, B * H)
    _attn_bwd_preprocess[pre_grid](
        out,
        do,
        delta,
        B,
        H,
        S,
        BLOCK_M=pre_block,
        HEAD_DIM=D,
    )

    block_m1, block_n1, block_m2, block_n2 = 32, 128, 128, 32
    blk_slice_factor = 2
    rcp_ln2 = 1.4426950408889634
    scaled_k = k * (float(scale) * rcp_ln2)
    grid = (S // block_n1, 1, B * H)
    _attn_bwd[grid](
        q,
        scaled_k,
        v,
        float(scale),
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        H,
        S,
        BLOCK_M1=block_m1,
        BLOCK_N1=block_n1,
        BLOCK_M2=block_m2,
        BLOCK_N2=block_n2,
        BLK_SLICE_FACTOR=blk_slice_factor,
        HEAD_DIM=D,
        CAUSAL=causal,
        num_warps=4,
        num_stages=5,
    )
    return dq, dk, dv
