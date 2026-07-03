# FlashAttention v2 Tutorial Path

本笔记对应 `flash_attention_v2`，目标是把 Triton 官方教程 06 中偏 FlashAttention-2 和性能工程的实现路线，单独放到本项目中，和已有的 `fused_attention` 教学版并存。

参考来源：
- Triton official tutorial 06: <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- FlashAttention-2 paper: <https://tridao.me/publications/flash2/flash2.pdf>

## 文件入口

```text
kernel:    python/triton_playground/kernels/flash_attention_v2.py
op:        python/triton_playground/ops/flash_attention_v2.py
test:      tests/test_flash_attention_v2.py
benchmark: scripts/bench_flash_attention_v2.py
```

Python API:

```python
from triton_playground.ops import flash_attention_v2, triton_flash_attention_v2

out = flash_attention_v2(q, k, v, causal=True)
```

输入输出：

```text
Q/K/V: [B, H, S, D], contiguous, float16
O:     [B, H, S, D]
LSE:   [B, H, S], float32, internal saved tensor
```

当前学习版约束：

- `D` 必须在 `{16, 32, 64, 128, 256}` 中。
- `S` 必须是 128 的倍数，因为 backward 复用了官方教程的固定 block 拆分。
- 只支持 `float16`。
- 当前没有 dropout、attention bias、varlen、GQA/MQA、paged KV cache。
- FP8 路径没有接入；官方教程中的 FP8 output 是更靠近 Hopper/Blackwell 的扩展点。

## 和当前 v1 学习版的区别

`fused_attention.py` 是 FlashAttention-1 风格教学实现，重点是清楚地展示：

```text
one program = one batch-head + one Q block
scan all K/V blocks
maintain m/l/acc
write O
```

`flash_attention_v2.py` 更接近官方教程 06，重点不是把公式写得最短，而是加入性能工程结构：

- 用 `tl.math.exp2` 和 `tl.math.log2`，把 softmax scale 转成 base-2。
- 用 `STAGE` 把 causal attention 拆成 off-band 和 on-band。
- forward 用 `@triton.autotune` 搜索 `BLOCK_M/BLOCK_N/num_warps/num_stages`。
- 支持 Triton TensorDescriptor API，并保留 host descriptor pre-hook。
- backward 采用官方教程的 block 组织方式，分别计算 preprocess、`dK/dV`、`dQ`。

## Forward Grid

逻辑上把 batch 和 head flatten：

```text
BH = B * H
Q/K/V: [BH, S, D]
```

Triton grid：

```text
grid = (ceil(S / BLOCK_M), B * H, 1)
```

program id：

```text
start_m = tl.program_id(0)  # Q block id
off_hz  = tl.program_id(1)  # batch-head id
```

每个 program 负责：

```text
Q block: [BLOCK_M, D]
O block: [BLOCK_M, D]
```

## Base-2 Softmax

普通 softmax 使用：

```text
exp(score * sm_scale - m)
```

官方教程里为了使用 `exp2`，先把 scale 转成 base-2：

```text
qk_scale = sm_scale * 1.4426950408889634
```

其中：

```text
1.4426950408889634 = 1 / ln(2)
```

然后 kernel 内使用：

```text
p = exp2(qk * qk_scale - m)
```

最后保存的 `lse` 也是 base-2 坐标：

```text
M = m + log2(l)
```

Backward 重建概率块时：

```text
P_block = exp2(QK_block * sm_scale / ln(2) - M)
```

## Forward STAGE

非 causal：

```text
STAGE = 1
inner stage = 3
scan K/V range: [0, S)
```

Causal：

```text
STAGE = 3
stage 1/off-band: scan [0, start_m * BLOCK_M)
stage 2/on-band:  scan [start_m * BLOCK_M, (start_m + 1) * BLOCK_M)
```

这样做的原因是：causal attention 中，远离对角线的块完全合法，不需要逐元素 causal mask；只有对角线附近的 on-band 块需要 mask。

## Forward Flow

```text
load Q block
init:
  m_i = -inf
  l_i = 1
  acc = 0

for each selected K/V block:
  qk = Q_block @ K_block.T
  apply stage-specific causal mask if needed
  m_new = max(m_i, rowmax(qk))
  p = exp2(qk - m_new)
  alpha = exp2(m_i - m_new)
  acc = acc * alpha + p @ V_block
  l_i = l_i * alpha + rowsum(p)
  m_i = m_new

lse = m_i + log2(l_i)
O = acc / l_i
```

这里的 `l_i` 初始值沿用了官方教程写法，初始化为 `1.0`。第一轮由于 `m_i = -inf`，`alpha = 0`，历史项不会贡献。

## Autotune

当前配置搜索：

```text
BLOCK_M in [64, 128]
BLOCK_N in [16, 32, 64, 128]
num_stages in [2, 3, 4] on CUDA
num_warps in [4, 8]
```

剪枝规则：

- `BLOCK_M <= N_CTX`
- `BLOCK_N <= HEAD_DIM`
- `N_CTX % BLOCK_M == 0`
- `N_CTX % BLOCK_N == 0`
- causal on-band 路径要求 `BLOCK_M >= BLOCK_N`

测试中会使用单一配置，避免 pytest 第一次运行时 autotune 时间过长。

## Backward Flow

Forward 保存：

```text
O:   output
M:   base-2 lse
Q/K/V
```

Backward 输入：

```text
dO
```

第一步 preprocess：

```text
Delta_i = sum_d O[i, d] * dO[i, d]
```

然后重建局部概率块：

```text
P = exp2(QK^T * sm_scale / ln(2) - M)
```

核心公式：

```text
dV = P^T @ dO
dP = dO @ V^T
dS = P * (dP - Delta)
dQ = dS @ K * sm_scale
dK = dS^T @ Q * sm_scale
```

实现细节：

- `_attn_bwd_preprocess` 计算 `Delta`。
- `_attn_bwd_dkdv` 对一个 K/V block 累积 `dK/dV`。
- `_attn_bwd_dq` 对一个 Q block 累积 `dQ`。
- `_attn_bwd` 是 wrapper kernel，调度 `dK/dV` 和 `dQ` 两条路径。

官方教程中 backward 会把 `K` 预先乘上：

```text
sm_scale * (1 / ln(2))
```

这样 `Q @ K.T` 直接得到 base-2 坐标下的 score，最后对 `dQ` 乘回 `ln(2)`，对 `dK` 乘回 `sm_scale`。

## Benchmark

单点：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16 --causal
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --mode backward --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16
```

画图：

```bash
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --sweep --plot --batch 1 --heads 8 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --sweep --plot --mode backward --batch 1 --heads 8 --dim 64 --dtype float16
```

默认 CSV：

```text
out/triton/benchmarks/w14_flash_attention_v2.csv
```

默认对比：

```text
flash_attention_v2  # official tutorial 06 style
flash_attention_v1  # project learning implementation
torch_attention     # materialized PyTorch baseline
```

## 学习建议

建议先读 `fused-attention.md`，确认自己已经能手推 `m/l/acc` 的 online softmax 更新。然后再看本文件里的 v2 差异：

1. 先看 base-2 softmax：为什么 `scale` 乘 `1 / ln(2)`。
2. 再看 causal `STAGE`：为什么对角线块和非对角线块要拆开。
3. 接着看 autotune：哪些配置被剪枝，为什么 `BLOCK_N <= HEAD_DIM`。
4. 最后看 backward：为什么只保存 `lse`，却能重建 `P_block`。
