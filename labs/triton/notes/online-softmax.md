# Online Softmax

本笔记用于沉淀 W13 的 online softmax 推导和实验结果。

## 普通 Softmax

对一行输入 `x`：

```text
m = max(x)
y_i = exp(x_i - m) / sum_j exp(x_j - m)
```

减去行最大值可以避免指数溢出。

## Streaming 形式

分块扫描时维护两个状态：

```text
m_old: 已处理元素的最大值
l_old: sum(exp(x - m_old))
```

读入新 block 后：

```text
m_new = max(m_old, max(block))
l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))
```

最终输出仍然除以全局归一化项。这个形式是后续 FlashAttention online softmax 的基础。

## 实验记录模板

| shape | dtype | mask | reference | atol | rtol | max error | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | `torch.softmax` | TODO | TODO | TODO | TODO |
