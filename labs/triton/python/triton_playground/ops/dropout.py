from __future__ import annotations

import torch

from triton_playground.kernels.dropout import launch_dropout_with_mask, launch_seeded_dropout


def dropout_with_mask(x: torch.Tensor, keep_mask: torch.Tensor, p: float, block_size: int = 1024) -> torch.Tensor:
    """Apply dropout with an explicit keep mask, matching the Triton tutorial baseline."""
    return launch_dropout_with_mask(x, keep_mask, p=p, block_size=block_size)


def seeded_dropout(x: torch.Tensor, p: float, seed: int, block_size: int = 1024) -> torch.Tensor:
    """Apply low-memory dropout by regenerating the mask from a Philox seed."""
    return launch_seeded_dropout(x, p=p, seed=seed, block_size=block_size)


def low_memory_dropout(x: torch.Tensor, p: float, seed: int, block_size: int = 1024) -> torch.Tensor:
    """Alias for seeded dropout, emphasizing that the mask tensor is not stored."""
    return seeded_dropout(x, p=p, seed=seed, block_size=block_size)
