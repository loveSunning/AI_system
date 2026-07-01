from __future__ import annotations

import pytest


def require_triton_matmul_bias_silu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import matmul_bias_silu, torch_matmul_bias_silu

    return torch, matmul_bias_silu, torch_matmul_bias_silu


def assert_matmul_bias_silu_close(torch, actual, expected) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M,N,K",
    [
        (128, 128, 64),
        (256, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
        (513, 257, 129),
    ],
)
@pytest.mark.parametrize("dtype_name", ["float16"])
def test_matmul_bias_silu_matches_torch(M: int, N: int, K: int, dtype_name: str) -> None:
    torch, matmul_bias_silu, torch_matmul_bias_silu = require_triton_matmul_bias_silu()
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)

    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    bias = torch.randn((N,), device="cuda", dtype=dtype)

    actual = matmul_bias_silu(a, b, bias)
    expected = torch_matmul_bias_silu(a, b, bias)

    assert_matmul_bias_silu_close(torch, actual, expected)


def test_matmul_bias_silu_supports_explicit_config() -> None:
    torch, matmul_bias_silu, torch_matmul_bias_silu = require_triton_matmul_bias_silu()
    a = torch.randn((257, 129), device="cuda", dtype=torch.float16)
    b = torch.randn((129, 263), device="cuda", dtype=torch.float16)
    bias = torch.randn((263,), device="cuda", dtype=torch.float16)

    actual = matmul_bias_silu(
        a,
        b,
        bias,
        block_size_m=64,
        block_size_n=128,
        block_size_k=32,
        group_size_m=4,
        num_warps=4,
        num_stages=4,
    )
    expected = torch_matmul_bias_silu(a, b, bias)

    assert_matmul_bias_silu_close(torch, actual, expected)


def test_matmul_bias_silu_rejects_bad_bias_shape() -> None:
    torch, matmul_bias_silu, _ = require_triton_matmul_bias_silu()
    a = torch.empty((8, 16), device="cuda", dtype=torch.float16)
    b = torch.empty((16, 32), device="cuda", dtype=torch.float16)
    bias = torch.empty((31,), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="bias must have shape"):
        matmul_bias_silu(a, b, bias)


def test_matmul_bias_silu_rejects_shape_mismatch() -> None:
    torch, matmul_bias_silu, _ = require_triton_matmul_bias_silu()
    a = torch.empty((8, 15), device="cuda", dtype=torch.float16)
    b = torch.empty((16, 32), device="cuda", dtype=torch.float16)
    bias = torch.empty((32,), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="shape mismatch"):
        matmul_bias_silu(a, b, bias)
