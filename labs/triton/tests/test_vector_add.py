from __future__ import annotations

import pytest


def require_triton_vector_add():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import vector_add

    return torch, vector_add


@pytest.mark.parametrize("n_elements", [1, 7, 1024, 1024 + 7, 98432])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_vector_add_matches_torch(n_elements: int, dtype_name: str) -> None:
    torch, vector_add = require_triton_vector_add()
    dtype = getattr(torch, dtype_name)
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    y = torch.randn(n_elements, device="cuda", dtype=dtype)

    actual = vector_add(x, y, block_size=1024)
    expected = x + y

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_vector_add_rejects_shape_mismatch() -> None:
    torch, vector_add = require_triton_vector_add()
    x = torch.empty(4, device="cuda")
    y = torch.empty(5, device="cuda")

    with pytest.raises(ValueError, match="same shape"):
        vector_add(x, y)
