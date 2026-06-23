# Tests

这里放 `pytest` 测试，优先覆盖 correctness，再覆盖配置边界。

建议命名：

- `test_vector_add.py`
- `test_fused_softmax.py`
- `test_matmul.py`
- `test_persistent_matmul.py`
- `test_fused_ops.py`
- `test_online_softmax.py`
- `test_attention_forward.py`

每个测试至少记录 shape、dtype、reference、`atol/rtol`。Attention 相关测试要覆盖 causal mask 和非 2 的幂 shape。
