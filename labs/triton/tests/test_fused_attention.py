from __future__ import annotations

import pytest


def require_triton_fused_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import flash_attention, torch_fused_attention_reference, triton_fused_attention, triton_stepwise_attention

    return torch, flash_attention, torch_fused_attention_reference, triton_fused_attention, triton_stepwise_attention


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
    torch, _, torch_ref, triton_fused_attention, _ = require_triton_fused_attention()
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)

    q = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_fused_attention(q, k, v, causal=causal, block_m=16, block_n=32)
    expected = torch_ref(q, k, v, causal=causal)

    assert_attention_close(torch, actual, expected)


def test_fused_attention_matches_stepwise_attention() -> None:
    torch, _, _, triton_fused_attention, triton_stepwise_attention = require_triton_fused_attention()
    q = torch.randn((1, 2, 96, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_fused_attention(q, k, v, causal=True, block_m=16, block_n=32, block_d=64)
    expected = triton_stepwise_attention(q, k, v, causal=True, block_m=16, block_n=32, block_d=32)[0]

    assert_attention_close(torch, actual, expected)


def test_fused_attention_rejects_shape_mismatch() -> None:
    torch, _, _, triton_fused_attention, _ = require_triton_fused_attention()
    q = torch.empty((1, 2, 16, 32), device="cuda", dtype=torch.float16)
    k = torch.empty((1, 2, 17, 32), device="cuda", dtype=torch.float16)
    v = torch.empty((1, 2, 16, 32), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="same shape"):
        triton_fused_attention(q, k, v)


def test_fused_attention_rejects_too_small_block_d() -> None:
    torch, _, _, triton_fused_attention, _ = require_triton_fused_attention()
    q = torch.empty((1, 1, 16, 64), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="block_d must be >= D"):
        triton_fused_attention(q, q, q, block_d=32)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 16, 32), (1, 2, 32, 32), (1, 2, 33, 64)])
def test_flash_attention_backward_matches_torch(causal: bool, shape: tuple[int, int, int, int]) -> None:
    torch, flash_attention, torch_ref, _, _ = require_triton_fused_attention()
    torch.manual_seed(0)
    B, H, S, D = shape

    q_ref = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16, requires_grad=True)
    k_ref = torch.randn_like(q_ref, requires_grad=True)
    v_ref = torch.randn_like(q_ref, requires_grad=True)
    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)
    dout = torch.randn_like(q_ref)

    out_ref = torch_ref(q_ref, k_ref, v_ref, causal=causal)
    out_tri = flash_attention(q_tri, k_tri, v_tri, causal=causal, block_m=16, block_n=32)
    out_ref.backward(dout)
    out_tri.backward(dout)

    assert_attention_close(torch, out_tri, out_ref)
    torch.testing.assert_close(q_tri.grad.float(), q_ref.grad.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(k_tri.grad.float(), k_ref.grad.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_tri.grad.float(), v_ref.grad.float(), rtol=5e-2, atol=5e-2)
