"""User-facing experimental ops built on top of Triton kernels."""

from triton_playground.ops.attention_forward import torch_attention, triton_stepwise_attention
from triton_playground.ops.dropout import dropout_with_mask, low_memory_dropout, seeded_dropout
from triton_playground.ops.fused_softmax import fused_softmax
from triton_playground.ops.grouped_gemm import grouped_gemm
from triton_playground.ops.layer_norm import layer_norm
from triton_playground.ops.matmul import matmul, matmul_fixed
from triton_playground.ops.matmul_bias_silu import matmul_bias_silu, torch_matmul_bias_silu
from triton_playground.ops.online_softmax import online_softmax, torch_online_softmax
from triton_playground.ops.persistent_matmul import persistent_matmul, persistent_matmul_fixed
from triton_playground.ops.rms_norm import naive_rms_norm, rms_norm
from triton_playground.ops.softmax_baselines import naive_softmax
from triton_playground.ops.vector_add import vector_add

__all__ = [
    "dropout_with_mask",
    "fused_softmax",
    "grouped_gemm",
    "layer_norm",
    "low_memory_dropout",
    "matmul",
    "matmul_bias_silu",
    "matmul_fixed",
    "naive_softmax",
    "naive_rms_norm",
    "online_softmax",
    "persistent_matmul",
    "persistent_matmul_fixed",
    "rms_norm",
    "seeded_dropout",
    "torch_online_softmax",
    "torch_matmul_bias_silu",
    "torch_attention",
    "triton_stepwise_attention",
    "vector_add",
]
