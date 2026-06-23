from __future__ import annotations

import torch


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """PyTorch baseline that materializes softmax's intermediate tensors."""
    x_max = torch.max(x, dim=-1, keepdim=True).values
    numerator = torch.exp(x - x_max)
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator
