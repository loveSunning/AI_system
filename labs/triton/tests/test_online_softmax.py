from __future__ import annotations

import pytest


def require_triton_online_softmax():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import fused_softmax, online_softmax, torch_online_softmax

    return torch, online_softmax, torch_online_softmax, fused_softmax


def assert_softmax_close(torch, actual, expected) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (4, 16), (17, 129), (32, 1000), (8, 2049)])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_online_softmax_matches_torch(shape: tuple[int, int], dtype_name: str) -> None:
    torch, online_softmax, torch_online_softmax, _ = require_triton_online_softmax()
    dtype = getattr(torch, dtype_name)
    x = torch.randn(shape, device="cuda", dtype=dtype)

    actual_triton = online_softmax(x, block_size=256, num_warps=4)
    actual_torch_online = torch_online_softmax(x, block_size=256)
    expected = torch.softmax(x, dim=-1)

    assert_softmax_close(torch, actual_triton, expected)
    assert_softmax_close(torch, actual_torch_online, expected)


def test_online_softmax_matches_fused_softmax() -> None:
    torch, online_softmax, _, fused_softmax = require_triton_online_softmax()
    x = torch.randn((16, 257), device="cuda", dtype=torch.float32)

    actual = online_softmax(x, block_size=128, num_warps=4)
    expected = fused_softmax(x, block_size=512, num_warps=4)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_online_softmax_rejects_non_power_of_two_block_size() -> None:
    torch, online_softmax, _, _ = require_triton_online_softmax()
    x = torch.randn((4, 129), device="cuda")

    with pytest.raises(ValueError, match="power of two"):
        online_softmax(x, block_size=192)


def test_torch_online_softmax_rejects_non_2d_input() -> None:
    torch, _, torch_online_softmax, _ = require_triton_online_softmax()
    x = torch.randn(16, device="cuda")

    with pytest.raises(ValueError, match="2D tensor"):
        torch_online_softmax(x)
