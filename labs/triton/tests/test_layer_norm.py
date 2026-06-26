from __future__ import annotations

import pytest


def require_triton_layer_norm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import layer_norm

    return torch, layer_norm


def assert_layer_norm_close(torch, actual, expected) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def run_layer_norm_case(
    shape: tuple[int, ...],
    dtype_name: str,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> None:
    torch, layer_norm = require_triton_layer_norm()
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)

    x = (-2.3 + 0.5 * torch.randn(shape, device="cuda", dtype=dtype)).requires_grad_(True)
    weight = torch.rand((shape[-1],), device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.rand((shape[-1],), device="cuda", dtype=dtype, requires_grad=True)
    dy = 0.1 * torch.randn_like(x)

    y_tri = layer_norm(x, weight, bias, eps=1e-5, block_size=block_size, num_warps=num_warps)
    y_ref = torch.nn.functional.layer_norm(x, (shape[-1],), weight, bias, eps=1e-5)

    y_tri.backward(dy, retain_graph=True)
    dx_tri = x.grad.detach().clone()
    dw_tri = weight.grad.detach().clone()
    db_tri = bias.grad.detach().clone()

    x.grad = None
    weight.grad = None
    bias.grad = None

    y_ref.backward(dy, retain_graph=True)
    dx_ref = x.grad.detach().clone()
    dw_ref = weight.grad.detach().clone()
    db_ref = bias.grad.detach().clone()

    assert_layer_norm_close(torch, y_tri, y_ref)
    assert_layer_norm_close(torch, dx_tri, dx_ref)
    assert_layer_norm_close(torch, dw_tri, dw_ref)
    assert_layer_norm_close(torch, db_tri, db_ref)


@pytest.mark.parametrize("shape", [(1, 1), (2, 7), (17, 129), (4, 3, 257)])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_layer_norm_forward_backward_matches_torch(shape: tuple[int, ...], dtype_name: str) -> None:
    run_layer_norm_case(shape, dtype_name)


def test_layer_norm_supports_explicit_block_size() -> None:
    run_layer_norm_case((8, 257), "float32", block_size=512, num_warps=4)


def test_layer_norm_rejects_bad_weight_shape() -> None:
    torch, layer_norm = require_triton_layer_norm()
    x = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    weight = torch.ones((7,), device="cuda", dtype=torch.float32)
    bias = torch.zeros((8,), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="weight must have shape"):
        layer_norm(x, weight, bias)


def test_layer_norm_rejects_non_contiguous_input() -> None:
    torch, layer_norm = require_triton_layer_norm()
    x = torch.randn((8, 16), device="cuda", dtype=torch.float32).t()
    weight = torch.ones((8,), device="cuda", dtype=torch.float32)
    bias = torch.zeros((8,), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="contiguous"):
        layer_norm(x, weight, bias)


def test_layer_norm_rejects_feature_dim_over_64kb() -> None:
    torch, layer_norm = require_triton_layer_norm()
    n_cols = 16385
    x = torch.empty((1, n_cols), device="cuda", dtype=torch.float32)
    weight = torch.empty((n_cols,), device="cuda", dtype=torch.float32)
    bias = torch.empty((n_cols,), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="64KB"):
        layer_norm(x, weight, bias)
