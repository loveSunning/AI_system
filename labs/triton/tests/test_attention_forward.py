from __future__ import annotations

import pytest


def require_triton_attention_forward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import torch_attention, triton_stepwise_attention

    return torch, torch_attention, triton_stepwise_attention


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
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype_name", ["float16"])
def test_attention_forward_matches_torch(B: int, H: int, S: int, D: int, causal: bool, dtype_name: str) -> None:
    torch, torch_attention, triton_stepwise_attention = require_triton_attention_forward()
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)

    q = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
    k = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
    v = torch.randn((B, H, S, D), device="cuda", dtype=dtype)

    out_ref, _, _ = torch_attention(q, k, v, causal=causal)
    out_tri, _, probs_tri = triton_stepwise_attention(q, k, v, causal=causal)

    assert_attention_close(torch, out_tri, out_ref)

    row_sum = probs_tri.float().sum(dim=-1)
    torch.testing.assert_close(row_sum, torch.ones_like(row_sum), rtol=2e-3, atol=2e-3)

    if causal:
        mask = torch.triu(torch.ones((S, S), device=q.device, dtype=torch.bool), diagonal=1)
        masked_probs = probs_tri[:, :, mask]
        assert masked_probs.abs().max().item() < 1e-4


def test_attention_forward_supports_explicit_blocks() -> None:
    torch, torch_attention, triton_stepwise_attention = require_triton_attention_forward()
    q = torch.randn((1, 2, 65, 32), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual, _, _ = triton_stepwise_attention(
        q,
        k,
        v,
        causal=True,
        block_m=16,
        block_n=32,
        block_d=32,
        softmax_block=128,
    )
    expected, _, _ = torch_attention(q, k, v, causal=True)

    assert_attention_close(torch, actual, expected)


def test_attention_forward_rejects_shape_mismatch() -> None:
    torch, _, triton_stepwise_attention = require_triton_attention_forward()
    q = torch.empty((1, 2, 16, 32), device="cuda", dtype=torch.float16)
    k = torch.empty((1, 2, 17, 32), device="cuda", dtype=torch.float16)
    v = torch.empty((1, 2, 16, 32), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="same shape"):
        triton_stepwise_attention(q, k, v)


def test_attention_forward_rejects_too_small_softmax_block() -> None:
    torch, _, triton_stepwise_attention = require_triton_attention_forward()
    q = torch.empty((1, 1, 129, 32), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="softmax_block must be >= S"):
        triton_stepwise_attention(q, q, q, softmax_block=128)
