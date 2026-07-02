# Fused Attention Forward

本笔记对应 W14 fused attention forward。目标是把前面 W13 online softmax 和 W14 stepwise attention 连接起来：

```text
stepwise:
QK^T -> materialized scores -> softmax -> materialized probs -> PV

fused:
QK^T block -> online softmax state -> accumulate PV -> O
```

当前实现是教学版 forward-only kernel，参考 Triton 官方 fused attention 教程的核心思路，但暂不实现 backward、FP8、TensorDescriptor、warp-specialization、Hopper/Blackwell 专用路径和 autotune。

## 输入输出

输入：

```text
Q: [B, H, S, D]
K: [B, H, S, D]
V: [B, H, S, D]
```

逻辑上把 batch 和 head flatten：

```text
BH = B * H
Q/K/V: [BH, S, D]
O:     [BH, S, D]
```

Python API 保持 `[B, H, S, D]`，kernel 内用 `bh = tl.program_id(1)` 选中一个 batch-head。

## Grid 映射

当前 grid：

```text
grid = (ceil(S / BLOCK_M), B * H)
```

program id：

```text
pid_m = tl.program_id(0)   # 第几个 Q block
bh    = tl.program_id(1)   # 第几个 batch-head
```

每个 Triton program 负责：

```text
Q block: [BLOCK_M, D]
O block: [BLOCK_M, D]
```

然后沿 sequence 维扫描所有 K/V block：

```text
for kv_start in range(0, S, BLOCK_N):
    K block: [BLOCK_N, D]
    V block: [BLOCK_N, D]
```

## Kernel 主流程

```text
输入 Q/K/V: [B, H, S, D]
        |
        v
逻辑 flatten: [BH, S, D]
        |
        v
grid = (ceil(S / BLOCK_M), BH)
        |
        v
每个 program 负责一个 batch-head + 一个 Q block
        |
        v
加载 Q_block: [BLOCK_M, D]
        |
        v
初始化 online softmax 状态
m   = [-inf]       shape: [BLOCK_M]
l   = [0]          shape: [BLOCK_M]
acc = zeros        shape: [BLOCK_M, D]
        |
        v
循环扫描 K/V block
        |
        v
qk = Q_block @ K_block.T
qk shape: [BLOCK_M, BLOCK_N]
        |
        v
加 boundary / causal mask
        |
        v
online softmax 更新 m/l/acc
        |
        v
所有 K/V block 扫完
        |
        v
O_block = acc / l[:, None]
        |
        v
写回 O: [BH, S, D]
        |
        v
恢复逻辑视图: [B, H, S, D]
```

## Online Softmax 状态

对每个 Q row 维护三个状态：

```text
m_i:   已扫描 K/V block 的 row-wise max
l_i:   sum(exp(score - m_i))
acc:   sum(exp(score - m_i) * V)
```

形状：

```text
m_i: [BLOCK_M]
l_i: [BLOCK_M]
acc: [BLOCK_M, D]
```

扫描新 block 时：

```text
qk = Q_block @ K_block.T * scale
m_new = max(m_i, rowmax(qk))
p = exp(qk - m_new[:, None])
alpha = exp(m_i - m_new)
l_new = l_i * alpha + rowsum(p)
acc_new = acc * alpha[:, None] + p @ V_block
```

最后：

```text
O_block = acc / l_i[:, None]
```

这个公式的关键是 `alpha`。当新的 block 里出现更大的 row max 时，历史的 `l_i` 和 `acc` 都是在旧 max 下累积的，需要乘：

```text
exp(m_old - m_new)
```

把历史项重新缩放到新的 max 坐标系。

## Mask 语义

当前有两类 mask：

### Boundary Mask

当 `S` 或 `D` 不是 block size 整除时，越界位置不能参与计算。

K/V block 边界：

```text
offs_kv < S
```

D 维边界：

```text
offs_d < D
```

越界 qk 填为 `-inf`，这样它在 rowmax 和 exp/sum 中都没有贡献。

### Causal Mask

causal attention 要求第 `i` 个 query 只能看 `j <= i` 的 key：

```text
valid = key_index <= query_index
```

kernel 中：

```text
qk = where(offs_kv[None, :] <= offs_m[:, None], qk, -inf)
```

这和 stepwise attention 中对完整 `scores[S, S]` 上三角填 `-inf` 等价。

## 与 Stepwise Attention 的区别

Stepwise 版本：

```text
scores = Q @ K.T      # [B, H, S, S], fp32
probs = softmax(scores)
out = probs @ V
```

优点：

- 容易 debug。
- 可以直接检查 scores/probs。
- 和 PyTorch reference 一一对应。

缺点：

- 显式物化 `scores` 和 `probs`。
- 需要多个 kernel launch。
- `S` 大时显存按 `O(B * H * S^2)` 增长。

Fused 版本：

```text
for each Q block:
    scan K/V blocks
    update online softmax state
    accumulate PV
```

优点：

- 不物化完整 `scores/probs`。
- softmax 和 `PV` 在一个 kernel 中融合。
- 训练的是 FlashAttention 的核心状态更新方式。

限制：

- 当前实现只做 forward。
- 当前实现是教学版，未做 autotune、官方教程中的 descriptor 路径和架构特化。
- 当前 `tl.dot(p, V)` 为了清晰使用 fp32 路径，性能不是最终目标。

## 运行

Correctness：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_fused_attention.py
```

Benchmark：

```bash
PYTHONPATH=python python3 scripts/bench_fused_attention.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_fused_attention.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16 --causal
PYTHONPATH=python python3 scripts/bench_fused_attention.py --sweep --plot --batch 1 --heads 8 --dim 64 --dtype float16
```

默认输出：

```text
out/triton/benchmarks/w14_fused_attention.csv
out/triton/benchmarks/plots/
```

## 结果如何解读

benchmark 默认对比：

- `triton_fused`: online softmax fused attention，不物化 scores/probs。
- `triton_stepwise`: 三段式 Triton attention，会物化 scores/probs。
- `torch_attention`: PyTorch materialized baseline。

CSV 中 `TFLOP/s_est` 使用近似 dense attention FLOPs：

```text
4 * B * H * S * S * D
```

其中：

- `QK^T`: `2 * B * H * S * S * D`
- `PV`: `2 * B * H * S * S * D`

causal 模式实际有效计算约少一半，但当前为了横向对比，仍使用同一个 dense 估算。
