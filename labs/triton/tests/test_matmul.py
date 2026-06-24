from __future__ import annotations

import pytest


def require_triton_matmul():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import matmul, matmul_fixed

    return torch, matmul, matmul_fixed


def assert_matmul_close(torch, actual, expected) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-1)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 19, 23), (64, 64, 64), (128, 96, 64)])
def test_matmul_fixed_matches_torch(shape: tuple[int, int, int]) -> None:
    torch, _, matmul_fixed = require_triton_matmul()
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    actual = matmul_fixed(a, b)
    expected = torch.matmul(a, b)

    assert_matmul_close(torch, actual, expected)


def test_matmul_autotune_matches_torch_smoke() -> None:
    torch, matmul, _ = require_triton_matmul()
    a = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 128), device="cuda", dtype=torch.float16)

    actual = matmul(a, b)
    expected = torch.matmul(a, b)

    assert_matmul_close(torch, actual, expected)


def test_matmul_rejects_shape_mismatch() -> None:
    torch, matmul, _ = require_triton_matmul()
    a = torch.empty((8, 7), device="cuda", dtype=torch.float16)
    b = torch.empty((8, 9), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="shape mismatch"):
        matmul(a, b)
