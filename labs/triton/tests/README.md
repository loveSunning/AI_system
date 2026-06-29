# Tests

这里放 `pytest` 测试，优先覆盖 correctness，再覆盖配置边界。

建议命名：

- `test_vector_add.py`
- `test_fused_softmax.py`
- `test_matmul.py`
- `test_grouped_gemm.py`
- `test_dropout.py`
- `test_layer_norm.py`
- `test_persistent_matmul.py`
- `test_fused_ops.py`
- `test_online_softmax.py`
- `test_attention_forward.py`

每个测试至少记录 shape、dtype、reference、`atol/rtol`。Attention 相关测试要覆盖 causal mask 和非 2 的幂 shape。

当前已落地：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_vector_add.py
PYTHONPATH=python pytest tests/test_fused_softmax.py
PYTHONPATH=python pytest tests/test_matmul.py
PYTHONPATH=python pytest tests/test_grouped_gemm.py
PYTHONPATH=python pytest tests/test_dropout.py
PYTHONPATH=python pytest tests/test_layer_norm.py
```
