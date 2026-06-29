from __future__ import annotations

import pytest


def require_triton_grouped_gemm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tests")

    from triton_playground.ops import grouped_gemm

    return torch, grouped_gemm


def assert_grouped_gemm_close(torch, actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_i, expected_i in zip(actual, expected):
        torch.testing.assert_close(actual_i, expected_i, rtol=1e-2, atol=1e-1)


@pytest.mark.parametrize(
    "shapes",
    [
        [(16, 16, 16)],
        [(64, 64, 64), (32, 128, 64), (128, 32, 96)],
        [(17, 19, 23), (33, 65, 31), (129, 71, 37)],
    ],
)
def test_grouped_gemm_matches_torch(shapes: list[tuple[int, int, int]]) -> None:
    torch, grouped_gemm = require_triton_grouped_gemm()
    torch.manual_seed(0)
    group_a = []
    group_b = []
    for m, n, k in shapes:
        group_a.append(torch.randn((m, k), device="cuda", dtype=torch.float16))
        group_b.append(torch.randn((k, n), device="cuda", dtype=torch.float16))

    actual = grouped_gemm(group_a, group_b)
    expected = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]

    assert_grouped_gemm_close(torch, actual, expected)


def test_grouped_gemm_supports_explicit_config() -> None:
    torch, grouped_gemm = require_triton_grouped_gemm()
    group_a = [
        torch.randn((96, 80), device="cuda", dtype=torch.float16),
        torch.randn((48, 40), device="cuda", dtype=torch.float16),
    ]
    group_b = [
        torch.randn((80, 64), device="cuda", dtype=torch.float16),
        torch.randn((40, 112), device="cuda", dtype=torch.float16),
    ]

    actual = grouped_gemm(
        group_a,
        group_b,
        block_size_m=64,
        block_size_n=64,
        block_size_k=32,
        num_sms=8,
        num_warps=4,
        num_stages=3,
    )
    expected = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]

    assert_grouped_gemm_close(torch, actual, expected)


def test_grouped_gemm_rejects_empty_group() -> None:
    _, grouped_gemm = require_triton_grouped_gemm()

    with pytest.raises(ValueError, match="at least one"):
        grouped_gemm([], [])


def test_grouped_gemm_rejects_shape_mismatch() -> None:
    torch, grouped_gemm = require_triton_grouped_gemm()
    group_a = [torch.empty((8, 7), device="cuda", dtype=torch.float16)]
    group_b = [torch.empty((8, 9), device="cuda", dtype=torch.float16)]

    with pytest.raises(ValueError, match="shape mismatch"):
        grouped_gemm(group_a, group_b)


def test_grouped_gemm_rejects_non_contiguous_input() -> None:
    torch, grouped_gemm = require_triton_grouped_gemm()
    group_a = [torch.randn((8, 16), device="cuda", dtype=torch.float16).t()]
    group_b = [torch.randn((8, 9), device="cuda", dtype=torch.float16)]

    with pytest.raises(ValueError, match="contiguous"):
        grouped_gemm(group_a, group_b)
