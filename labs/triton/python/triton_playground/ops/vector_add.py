from __future__ import annotations

import torch

from triton_playground.kernels.vector_add import launch_vector_add


def vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Add two CUDA tensors with a minimal Triton vector-add kernel."""
    return launch_vector_add(x, y, block_size=block_size)
