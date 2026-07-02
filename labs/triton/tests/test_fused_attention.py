from __future__ import annotations

import pytest


def require_triton_fused_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import torch_fused_attention_reference, triton_fused_attention, triton_stepwise_attention

    return torch, torch_fused_attention_reference, triton_fused_attention, triton_stepwise_attention


def assert_attention_close(torch, actual, expected) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "B,H,S,D",
    [
        (1, 1, 16, 32),
        (1, 2, 64, 32),
        (2, 4, 128, 64),
        (1, 8, 256, 64),
        (1, 2, 65, 32),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype_name", ["float16"])
def test_fused_attention_matches_torch(B: int, H: int, S: int, D: int, causal: bool, dtype_name: str) -> None:
    torch, torch_ref, triton_fused_attention, _ = require_triton_fused_attention()
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)

    q = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_fused_attention(q, k, v, causal=causal, block_m=16, block_n=32)
    expected = torch_ref(q, k, v, causal=causal)

    assert_attention_close(torch, actual, expected)


def test_fused_attention_matches_stepwise_attention() -> None:
    torch, _, triton_fused_attention, triton_stepwise_attention = require_triton_fused_attention()
    q = torch.randn((1, 2, 96, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_fused_attention(q, k, v, causal=True, block_m=16, block_n=32, block_d=64)
    expected = triton_stepwise_attention(q, k, v, causal=True, block_m=16, block_n=32, block_d=32)[0]

    assert_attention_close(torch, actual, expected)


def test_fused_attention_rejects_shape_mismatch() -> None:
    torch, _, triton_fused_attention, _ = require_triton_fused_attention()
    q = torch.empty((1, 2, 16, 32), device="cuda", dtype=torch.float16)
    k = torch.empty((1, 2, 17, 32), device="cuda", dtype=torch.float16)
    v = torch.empty((1, 2, 16, 32), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="same shape"):
        triton_fused_attention(q, k, v)


def test_fused_attention_rejects_too_small_block_d() -> None:
    torch, _, triton_fused_attention, _ = require_triton_fused_attention()
    q = torch.empty((1, 1, 16, 64), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="block_d must be >= D"):
        triton_fused_attention(q, q, q, block_d=32)
