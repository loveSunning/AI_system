from __future__ import annotations

import pytest


def require_triton_persistent_matmul():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.kernels.persistent_matmul import get_4090d_friendly_configs, is_likely_4090d
    from triton_playground.ops import persistent_matmul, persistent_matmul_fixed

    return torch, persistent_matmul, persistent_matmul_fixed, get_4090d_friendly_configs, is_likely_4090d


def assert_matmul_close(torch, actual, expected) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-1)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 19, 23), (128, 96, 64), (257, 129, 65)])
def test_persistent_matmul_fixed_matches_torch(shape: tuple[int, int, int]) -> None:
    torch, _, persistent_matmul_fixed, _, _ = require_triton_persistent_matmul()
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    actual = persistent_matmul_fixed(a, b, num_sms=8)
    expected = torch.matmul(a, b)

    assert_matmul_close(torch, actual, expected)


def test_persistent_matmul_autotune_matches_torch_smoke() -> None:
    torch, persistent_matmul, _, _, _ = require_triton_persistent_matmul()
    a = torch.randn((256, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 256), device="cuda", dtype=torch.float16)

    actual = persistent_matmul(a, b, num_sms=8)
    expected = torch.matmul(a, b)

    assert_matmul_close(torch, actual, expected)


def test_persistent_matmul_supports_float32() -> None:
    torch, _, persistent_matmul_fixed, _, _ = require_triton_persistent_matmul()
    a = torch.randn((64, 32), device="cuda", dtype=torch.float32)
    b = torch.randn((32, 48), device="cuda", dtype=torch.float32)

    actual = persistent_matmul_fixed(a, b, block_size_m=64, block_size_n=64, block_size_k=32, num_sms=4)
    expected = torch.matmul(a, b)

    assert_matmul_close(torch, actual, expected)


def test_persistent_matmul_rejects_shape_mismatch() -> None:
    torch, persistent_matmul, _, _, _ = require_triton_persistent_matmul()
    a = torch.empty((8, 7), device="cuda", dtype=torch.float16)
    b = torch.empty((8, 9), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="shape mismatch"):
        persistent_matmul(a, b)


def test_persistent_matmul_rejects_non_contiguous_input() -> None:
    torch, _, persistent_matmul_fixed, _, _ = require_triton_persistent_matmul()
    a = torch.randn((8, 16), device="cuda", dtype=torch.float16).t()
    b = torch.randn((8, 9), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="contiguous"):
        persistent_matmul_fixed(a, b)


def test_4090d_friendly_configs_are_bounded() -> None:
    _, _, _, get_4090d_friendly_configs, _ = require_triton_persistent_matmul()

    for config in get_4090d_friendly_configs():
        block_m = config.kwargs["BLOCK_SIZE_M"]
        block_n = config.kwargs["BLOCK_SIZE_N"]
        block_k = config.kwargs["BLOCK_SIZE_K"]
        shared_bytes = (block_m * block_k + block_k * block_n) * 2 * config.num_stages
        assert shared_bytes <= 100 * 1024
