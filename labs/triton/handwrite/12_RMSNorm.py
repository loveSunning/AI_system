import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_kernel(
    X,
    W,
    Y,
    RSTD,
    stride_xm: tl.constexpr,
    stride_ym: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X + row * stride_xm + cols
    y_ptrs = Y + row * stride_ym + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    mean_square = tl.sum(x *x, axis=0) / N
    rstd = tl.rsqrt(mean_square + eps)

    y = x * rstd * w

    tl.store(y_ptrs, y, mask=mask)
    tl.store(RSTD + row, rstd)


def rmsnorm_triton_forward(x, weight, eps=1e-6):
    assert x.is_cuda and weight.is_cuda
    assert x.shape[-1] == weight.shape[0]

    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N

    # 学习版先要求 contiguous，生产版再处理 stride
    x_2d = x.contiguous().view(M, N)
    y = torch.empty_like(x_2d)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(N)

    # hidden size 不能无限大，否则一个 program 的 block 太大
    assert BLOCK_SIZE <= 131072

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16

    _rmsnorm_fwd_kernel[(M,)](
        x_2d,
        weight,
        y,
        rstd,
        x_2d.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y.view(orig_shape), rstd


torch.manual_seed(0)

B, T, N = 2, 16, 1024
x = torch.randn(B, T, N, device="cuda", dtype=torch.float16)
weight = torch.randn(N, device="cuda", dtype=torch.float16)
eps = 1e-6

y_tri, rstd = rmsnorm_triton_forward(x, weight, eps)

y_ref = torch.nn.functional.rms_norm(
    x,
    normalized_shape=(N,),
    weight=weight,
    eps=eps,
)

print(torch.max(torch.abs(y_tri - y_ref)))
print(torch.allclose(y_tri, y_ref, atol=1e-2, rtol=1e-2))


@triton.jit
def _rmsnorm_bwd_dx_kernel(
    X,
    W,
    DY,
    DX,
    PARTIAL_DX,
    RSTD,
    stride_xm: tl.constexpr,
    stride_dym: tl.constexpr,
    stride_dxm: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride_dym + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row).to(tl.float32)

    x_hat = x * rstd
    p = dy * w

    c = tl.sum(p * x_hat, axis=0) / N
    dx = rstd(p - x_hat * c)
    partial_dw = dy * x_hat

    tl.store(DX + row * stride_dxm + cols, dx, mask=mask)
    tl.store(PARTIAL_DX + row * N +cols, partial_dw, mask=mask)


def rmsnorm_triton_backward_leanring(x, weight, dy, eps = 1e-6):
    orig_shape = x.shape

    N = x.shape[-1]
    M = x.numel() // N

    x_2d = x.contiguous().view(M, N)
    dy_2d = dy.contiguous().view(M, N)

    y, rstd = rmsnorm_triton_forward(x, weight, eps)

    dx = torch.empty_like(x_2d)
    partial_dw = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16

    
    _rmsnorm_bwd_dx_kernel[(M,)](
        x_2d,
        weight,
        dy_2d,
        dx,
        partial_dw,
        rstd,
        x_2d.stride(0),
        dy_2d.stride(0),
        dx.stride(0),
        N,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    dweight = partial_dw.sum(dim=0).to(weight.dtype)
    return dx.view(orig_shape), dweight

