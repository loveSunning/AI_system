from __future__ import annotations

import pytest


def require_triton_fused_softmax():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import fused_softmax

    return torch, fused_softmax


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (4, 16), (17, 129), (32, 1000)])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_fused_softmax_matches_torch(shape: tuple[int, int], dtype_name: str) -> None:
    torch, fused_softmax = require_triton_fused_softmax()
    dtype = getattr(torch, dtype_name)
    x = torch.randn(shape, device="cuda", dtype=dtype)

    actual = fused_softmax(x)
    expected = torch.softmax(x, dim=-1)

    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fused_softmax_supports_explicit_block_size() -> None:
    torch, fused_softmax = require_triton_fused_softmax()
    x = torch.randn((8, 257), device="cuda", dtype=torch.float32)

    actual = fused_softmax(x, block_size=512, num_warps=4)
    expected = torch.softmax(x, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fused_softmax_rejects_non_2d_input() -> None:
    torch, fused_softmax = require_triton_fused_softmax()
    x = torch.randn(16, device="cuda")

    with pytest.raises(ValueError, match="2D tensor"):
        fused_softmax(x)


def test_fused_softmax_rejects_too_small_block_size() -> None:
    torch, fused_softmax = require_triton_fused_softmax()
    x = torch.randn((4, 129), device="cuda")

    with pytest.raises(ValueError, match="block_size must be >= n_cols"):
        fused_softmax(x, block_size=128)
