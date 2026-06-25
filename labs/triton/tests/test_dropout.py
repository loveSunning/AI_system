from __future__ import annotations

import pytest


def require_triton_dropout():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import dropout_with_mask, low_memory_dropout, seeded_dropout

    return torch, dropout_with_mask, seeded_dropout, low_memory_dropout


def assert_dropout_close(torch, actual, expected) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_elements", [1, 7, 1024, 1024 + 3, 98432])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_dropout_with_mask_matches_pytorch_formula(n_elements: int, dtype_name: str) -> None:
    torch, dropout_with_mask, _, _ = require_triton_dropout()
    dtype = getattr(torch, dtype_name)
    p = 0.35
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    keep_mask = torch.rand(n_elements, device="cuda") > p

    actual = dropout_with_mask(x, keep_mask, p=p, block_size=1024)
    expected = torch.where(keep_mask, x / (1.0 - p), torch.zeros_like(x))

    assert_dropout_close(torch, actual, expected)


def test_seeded_dropout_is_reproducible_for_same_seed() -> None:
    torch, _, seeded_dropout, _ = require_triton_dropout()
    x = torch.randn(4096, device="cuda", dtype=torch.float32) + 1.0

    actual_1 = seeded_dropout(x, p=0.5, seed=123, block_size=1024)
    actual_2 = seeded_dropout(x, p=0.5, seed=123, block_size=1024)
    actual_3 = seeded_dropout(x, p=0.5, seed=512, block_size=1024)

    torch.testing.assert_close(actual_1, actual_2, rtol=0.0, atol=0.0)
    assert not torch.equal(actual_1, actual_3)


def test_low_memory_dropout_alias_matches_seeded_dropout() -> None:
    torch, _, seeded_dropout, low_memory_dropout = require_triton_dropout()
    x = torch.randn(4096, device="cuda", dtype=torch.float32)

    actual = low_memory_dropout(x, p=0.25, seed=777, block_size=1024)
    expected = seeded_dropout(x, p=0.25, seed=777, block_size=1024)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_seeded_dropout_zero_fraction_is_reasonable() -> None:
    torch, _, seeded_dropout, _ = require_triton_dropout()
    p = 0.4
    x = torch.ones(1 << 20, device="cuda", dtype=torch.float32)

    actual = seeded_dropout(x, p=p, seed=123, block_size=1024)
    zero_fraction = (actual == 0).float().mean().item()

    assert abs(zero_fraction - p) < 0.02


def test_dropout_rejects_invalid_probability() -> None:
    torch, dropout_with_mask, _, _ = require_triton_dropout()
    x = torch.randn(16, device="cuda", dtype=torch.float32)
    keep_mask = torch.ones(16, device="cuda", dtype=torch.bool)

    with pytest.raises(ValueError, match="0 <= p < 1"):
        dropout_with_mask(x, keep_mask, p=1.0)
