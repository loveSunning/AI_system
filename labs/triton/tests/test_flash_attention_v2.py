from __future__ import annotations

import os

import pytest


def require_flash_attention_v2():
    os.environ.setdefault("PYTEST_VERSION", "1")
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import (
        flash_attention,
        flash_attention_v2,
        flash_attention_v2_feature_reason,
        flash_attention_v2_is_available,
        torch_flash_attention_v2_reference,
        triton_flash_attention_v2,
    )

    if not flash_attention_v2_is_available():
        pytest.skip(flash_attention_v2_feature_reason())
    return torch, flash_attention, flash_attention_v2, torch_flash_attention_v2_reference, triton_flash_attention_v2


def assert_attention_close(torch, actual, expected, rtol: float = 3e-2, atol: float = 3e-2) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (1, 4, 256, 64)])
def test_flash_attention_v2_forward_matches_torch(causal: bool, shape: tuple[int, int, int, int]) -> None:
    torch, _, _, torch_ref, triton_flash_attention_v2 = require_flash_attention_v2()
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_flash_attention_v2(q, k, v, causal=causal, warp_specialize=False)
    expected = torch_ref(q, k, v, causal=causal)

    assert_attention_close(torch, actual, expected)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_v2_forward_matches_v1(causal: bool) -> None:
    torch, flash_attention, _, _, triton_flash_attention_v2 = require_flash_attention_v2()
    torch.manual_seed(1)
    q = torch.randn((1, 2, 128, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_flash_attention_v2(q, k, v, causal=causal, warp_specialize=False)
    expected = flash_attention(q, k, v, causal=causal, block_m=16, block_n=32, block_d=64)

    assert_attention_close(torch, actual, expected)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_v2_backward_matches_torch(causal: bool) -> None:
    torch, _, flash_attention_v2, torch_ref, _ = require_flash_attention_v2()
    torch.manual_seed(2)
    shape = (1, 2, 128, 64)
    q_ref = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    k_ref = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    v_ref = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)
    dout = torch.randn_like(q_ref)

    out_ref = torch_ref(q_ref, k_ref, v_ref, causal=causal)
    out_tri = flash_attention_v2(q_tri, k_tri, v_tri, causal=causal, warp_specialize=False)
    out_ref.backward(dout)
    out_tri.backward(dout)

    assert_attention_close(torch, out_tri, out_ref)
    torch.testing.assert_close(q_tri.grad.float(), q_ref.grad.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(k_tri.grad.float(), k_ref.grad.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(v_tri.grad.float(), v_ref.grad.float(), rtol=6e-2, atol=6e-2)


def test_flash_attention_v2_rejects_non_multiple_of_128_sequence() -> None:
    torch, _, flash_attention_v2, _, _ = require_flash_attention_v2()
    q = torch.empty((1, 2, 96, 64), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="multiple of 128"):
        flash_attention_v2(q, q, q)
