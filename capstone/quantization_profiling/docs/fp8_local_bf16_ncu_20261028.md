# FP8-local 与 BF16 的核心 kernel 对比及 Nsight Compute 采集方案

**命令参数：** [通用模型选择、kernel 名称/正则和 basic/detail](offline_nsight_parameters.md)。

**非 server 入口已统一：** [LLM.generate 的 NSYS / NCU 测试方案](offline_nsight.md)。两个工具共用 profile_kernels.py、冻结快照校验和 ProfilingWorker；原 server 方案与历史数据保留供对照。

日期：2026-10-28。模型：Qwen3-8B；原始实验 GPU：RTX 4090D；vLLM 0.14.1；PyTorch 2.9.1+cu128。输入/输出各 128 token，并发与 batch size 均为 1。

## 1. 结论与当前完成状态

FP8-local 在本组 **单 token decode** 中，kernel 持续时间累计比 BF16 高约 **8.99%**：每步 18.315824 ms 对 16.805448 ms。最需要分析的是 **down_proj、o_proj 与 gate_up_proj 对应的矩阵 kernel**，以及 FP8 路径中的量化、归一化融合与转换操作。

- **down_proj 是最大的退化来源**：每步增加 1.265437 ms，约慢 32.17%。
- **o_proj 是第二个退化来源**：每步增加 0.543201 ms，约慢 39.64%。
- qkv_proj 增加 0.272156 ms。
- **gate_up_proj 是 FP8 的主要收益来源**：每步节省 0.974208 ms，约快 12.79%；它的绝对耗时仍然最大，因此也值得优化。
- 上述四个投影抵消后的净增加为 1.106586 ms，约占总差异的 73.27%；其余非矩阵操作净增加 0.403633 ms。
- lm_head 两边都走 BF16 GEMV，耗时基本一致，不是 FP8 退化的主要原因。

在 **128 token prefill** 中，方向相反：FP8-local 的 kernel 累计时间为 16.781297 ms，BF16 为 20.278850 ms，减少约 17.25%。主要收益仍来自 gate_up_proj，节省 3.743372 ms。

这里已经完成的是：使用原始 Nsight Systems 报告重新导出的数据做投影级归因，核对执行形状与资源元数据，并提供可执行的 NCU 采集脚本和分析步骤。**尚无原 RTX 4090D 环境产生的 NCU 报告，因此下文的带宽、占用率、Tensor Core 利用率与 stall 原因均列为待验证假设，没有把它们写成实测结论。** 当前可访问的本机是 RTX 5060、驱动 580.88、NCU 2025.3.1，且没有原模型所需的 torch/vLLM/Triton 环境，无法用它替代原机采集。

## 2. 比较范围、证据和计时口径

主要证据是原 8 份报告中以下 4 份：

```text
no_eager_fp8-local_w128_o128_20261027T030619Z.nsys-rep
no_eager_bf16_w128_o128_20261027T030619Z.nsys-rep
eager_fp8-local_w128_o128_20261027T030619Z.nsys-rep
eager_bf16_w128_o128_20261027T030619Z.nsys-rep
```

原文件位于 `D:/Downloads/profiling/`。本次使用同一批原报告重新导出的第 5 次完整预热请求：1 次 prefill、127 次 decode。正式测量阶段有些报告的尾部采集不完整，不能把被截断的尾部当成 GPU 空闲，因此这里使用四份报告都完整的这一请求。

可复核数据和分析代码位于仓库根目录下：

- `out/profiling_analysis_20261027T030619Z/fp8_local_bf16_projection_details.json`
- `out/profiling_analysis_20261027T030619Z/fp8_local_bf16_projection_analysis.py`
- `out/profiling_analysis_20261027T030619Z/eight_*_warmup5_from_rep.sqlite`
- `out/profiling_analysis_20261027T030619Z/eight_prefill_decode_analysis.md`

原 8 份报告的完整性与重新导出的数据一致性已经在此前工作中审计。本篇直接复用这些证据，无需重新采集 nsys。
在包含上述 JSON 与 SQLite 输入的当前工作区，可复现已有数据分析与图；这不会启动 GPU 采集：

```powershell
Set-Location D:/workspace/learing/AI_system
python out/profiling_analysis_20261027T030619Z/fp8_local_bf16_projection_analysis.py
python out/profiling_analysis_20261027T030619Z/plot_fp8_local_bf16_projections.py
```

分析脚本只需 Python 标准库；绘图另需 matplotlib、numpy。现有 SQLite 路径记录指向当前工作区，换机复现时需保留相应输入并调整路径。

本篇主体使用 no_eager 的稳定 decode，以尽量减小逐 kernel host 提交间隙对比较的干扰；eager 数据用于交叉检查。

### 2.1 本篇时间表示什么

本文的 kernel 时间使用 **duration 求和**，decode 数值再除以 127，得到每个 decode step 的平均值；一个投影的数据是该 step 中 36 层同类投影的合计，**不是单层单次调用时间**。独立 sampling kernel 不在范围内。

kernel duration 包含设备内的访存、同步和等待，不是纯 FLOP 计算时间。它与整个 prefill/decode 时间段、CPU scope、HTTP 延迟均不同。

此前报告中的 FP8-local decode **kernel 区间并集**为 18.310614 ms；本篇为了做逐类别可相加分解，使用 **duration 累计** 18.315824 ms。两者差约 0.005210 ms，来自少量执行区间重叠。BF16 的累计和并集均为 16.805448 ms。不要把两种口径混在一张差异分解表中。

### 2.2 投影名称的对应证据与边界

原 trace 没有 qkv/o/gate_up/down 各自的 NVTX 标记，也没有直接记录 tensor 形状。投影名称依据 Qwen3 的执行顺序、36 层的重复模式与 grid 几何结构推断：

```text
每层：qkv_proj → o_proj → gate_up_proj → down_proj
共 36 层；最后一个矩阵操作：lm_head
```

分析程序检查了全部 127 个 decode step，每步均有 `36 × 4 + 1 = 145` 个矩阵 kernel。这个一致性支持投影对应关系，但不能把它描述为原 profiler 直接提供的投影标签。

## 3. Decode：哪些 kernel 决定 FP8-local 比 BF16 慢

### 3.1 按投影对齐后的结果

单位为 ms/decode step；正差值表示 FP8-local 更慢。

| 操作 | BF16 | FP8-local | FP8−BF16 | 相对 BF16 变化 |
|---|---:|---:|---:|---:|
| qkv_proj，36 次 | 1.961553 | 2.233709 | +0.272156 | +13.87% |
| o_proj，36 次 | 1.370345 | 1.913546 | +0.543201 | +39.64% |
| gate_up_proj，36 次 | 7.619741 | 6.645533 | −0.974208 | −12.79% |
| down_proj，36 次 | 3.933206 | 5.198643 | +1.265437 | +32.17% |
| lm_head，1 次 | 1.298737 | 1.298894 | +0.000157 | +0.01% |
| 其余非矩阵 kernel | 0.621866 | 1.025499 | +0.403633 | — |
| **全部 kernel 累计** | **16.805448** | **18.315824** | **+1.510377** | **+8.99%** |

差异可以相加还原：

```text
四个投影净差异：
  +0.272156 + 0.543201 − 0.974208 + 1.265437 = +1.106586 ms

完整差异：
  +1.106586 + 0.000157 + 0.403633 = +1.510377 ms
```

因此，优化优先级应同时看两件事：**谁造成精度方案之间的退化，以及谁占用 FP8 自身最多时间。**

- down_proj：最大的净退化，是第一个 NCU 详细采集目标。
- o_proj：相对退化最大，且同为较少 CTA，适合与 down_proj 比较。
- gate_up_proj：FP8 自身耗时最大，占本 step 累计时间约 36.28%，但已经比 BF16 快；用它解释“为什么相同 FP8 kernel 名字，有些形状受益、有些退化”。
- qkv_proj：后续补采，重要性低于上面三者。
- 量化/归一化融合：解释剩余约 26.72% 的净差异。

![同一 decode step 内 BF16 与 FP8-local 各投影累计时间](../../../out/profiling_analysis_20261027T030619Z/fp8_local_bf16_decode_projections.png)

可点击打开[对比图](../../../out/profiling_analysis_20261027T030619Z/fp8_local_bf16_decode_projections.png)。图由 `out/profiling_analysis_20261027T030619Z/plot_fp8_local_bf16_projections.py` 生成，数据源为上述 JSON；`out/` 被 gitignore 忽略，是本地分析产物目录。

### 3.2 按 kernel 名字直接汇总为什么会误判

FP8 的四个层内投影都叫：

```text
_w8a8_triton_block_scaled_mm
```

BF16 的 decode 矩阵 kernel 主要叫：

```text
internal::gemvx::kernel<...>
```

其中 BF16 的模板变体会被多个投影复用，而且最终 lm_head 也使用同类 GEMV。只看到 FP8 矩阵 kernel 约 15.991431 ms、BF16 所有 GEMV 约 16.183582 ms，就得出“FP8 线性计算更快”，是不公平的比较：**BF16 这个合计含约 1.298737 ms 的共同 lm_head，FP8 的 Triton 合计不含它。**

公平对齐 144 个层内投影后：

```text
BF16 四投影：      14.884845 ms/step
FP8-local 四投影： 15.991431 ms/step
净差异：          +1.106586 ms/step
```

因此 NCU 也不能只选第一个同名 kernel 就把它当成四种形状的代表；需要分开采 qkv、o、gate_up、down。

### 3.3 额外的非矩阵开销

no_eager 下非矩阵累计从 0.621866 增至 1.025499 ms，净增加 0.403633 ms/step。主要项目如下。

| 对应工作 | BF16 ms/step | FP8-local ms/step | 差异解释 |
|---|---:|---:|---|
| 两组主要归一化及其 FP8 融合路径 | 0.123143 | 0.306439 | 净增约 0.183296；FP8 是 `rms_norm_per_block_quant_kernel`，BF16 是两组归约融合 kernel |
| 独立分组量化 | 0 | 0.105166 | `per_token_group_quant_8bit_kernel`，72 次/step |
| 转换/复制融合路径 | 0 | 0.060779 | `triton_poi_fused__to_copy_empty_like_to_1`，71 次/step |
| SiLU/乘法及其 FP8 融合路径 | 0.036286 | 0.040639 | 净增约 0.004353；不同融合实现 |
| flash attention 主 kernel | 0.225752 | 0.237858 | 增约 0.012106；不是主要差异来源 |

第一行的 0.306439 ms **不能整体记成“新增加的纯量化时间”**，因为它取代了 BF16 的归一化路径。按语义做净比较后，增加约 0.183296 ms。其余细小差异和融合边界变化见 JSON，不能把上述短表当成完整残差分解。

no_eager 每步共 617 个 kernel，BF16 为 437 个；多出的操作既增加设备执行开销，也可能增加 host 启动成本。CUDA Graph 可以压缩后者，仍然需要 GPU 执行这些量化、转换与矩阵操作。

### 3.4 eager 交叉检查

同一个完整请求中，eager 的 kernel 累计为 BF16 17.013856 ms/step、FP8-local 18.525964 ms/step，差约 +1.512107 ms/step，方向与 no_eager 的 +1.510377 ms 一致。

因此本次“FP8 在单 token decode 中 kernel 累计更慢”并非仅由 CUDA Graph 回放造成。eager 额外存在较大的提交间隙；这部分已在此前 Nsight Systems 分析中说明，不应重新算到某个 GPU kernel 的执行瓶颈里。

## 4. Prefill：为什么 FP8 反而更快

下表单位是一次 128-token prefill 的 ms，同样为 36 层合计。

| 操作 | BF16 | FP8-local | FP8−BF16 |
|---|---:|---:|---:|
| qkv_proj | 2.424162 | 1.981785 | −0.442377 |
| o_proj | 1.617812 | 1.720913 | +0.103101 |
| gate_up_proj | 9.025440 | 5.282068 | −3.743372 |
| down_proj | 5.051176 | 5.017359 | −0.033817 |
| lm_head | 1.302739 | 1.328302 | +0.025563 |
| 其余非矩阵 kernel | 0.857521 | 1.450870 | +0.593349 |
| **全部 kernel 累计** | **20.278850** | **16.781297** | **−3.497553** |

prefill 的 gate_up_proj 节省约 41.48%，足以覆盖其他投影与量化路径的开销。down_proj 基本持平，o_proj 仍有轻微退化。由此可知，FP8 的结果依赖 **M/N/K 形状、kernel 实现和并行度**，不能用“精度减半所以一定快一倍”解释。

本组 no_eager prefill 没有实际回放 CUDA Graph。其 GPU 首尾跨度 FP8 约 81.65 ms、BF16 约 59.25 ms，与上面的 kernel 累计排序不同，原因涉及此前已经确认的 host 提交间隙。本篇 NCU 的目标是解释 kernel 自身，不用 NCU 端到端运行时间替代已完成的 nsys 分析。

## 5. 源码与资源元数据给出的候选原因

### 5.1 四种矩阵形状与 launch 配置

以下 M/N/K 来自 Qwen3-8B 结构及投影对应推断，含义为 `A[M,K] × B[K,N]`。decode 的 M=1，prefill 的 M=128。

| 投影 | N | K | FP8 decode CTA 数 | BF16 decode CTA 数 |
|---|---:|---:|---:|---:|
| qkv_proj | 6144 | 4096 | 48 | 1536 |
| o_proj | 4096 | 4096 | 32 | 1024 |
| gate_up_proj | 24576 | 4096 | 192 | 6144 |
| down_proj | 4096 | 12288 | 32 | 1024 |

FP8 decode 四种形状的其他元数据一致：128 threads/block、238 registers/thread、25,088 bytes shared memory/block。BF16 的 qkv/gate_up/down 是 128 threads/block、162 registers/thread、528 bytes shared memory/block；o_proj 是 **64 threads/block**、168 registers/thread、272 bytes shared memory/block。

这些是 launch 元数据，不是 achieved occupancy。寄存器数高不直接等价于 spilling，trace 的 localMemoryPerThread=0 也不能代替 NCU 对实际 local load/store 的检查。

### 5.2 down/o 并行度偏少，是最有针对性的假设

固定版本实现见 [vLLM v0.14.1 fp8_utils.py](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1039-L1222)。该路径建立 M/N tile 的 grid，K 主要在 CTA 内循环推进，没有从这里看到独立 K grid 维度的 split-K 分解。

在本次实际 launch 中，o 和 down 只有 32 个 CTA。一次执行中，即使每个 CTA 分配到不同 SM，也至多有 32 个 SM 获得这些 CTA；CTA 内更高 occupancy 无法弥补完全没有分配到工作的 SM。down 的 K 又是 o 的 3 倍，需要在少量 CTA 内处理更长的归约路径。

gate_up 的 N 大得多，形成 192 个 CTA，具备更大的跨 SM 并行机会。它与 o/down 的收益差异，支持把“较窄 N＋单 token M 导致并行度不足”作为首要待验证假设。

但 **不能仅凭 CTA 个数断言瓶颈已经确定**。还要查看 grid waves、活跃 SM、设备带宽、L2 命中、scheduler issue、Tensor Core 管线以及寄存器/共享内存约束。BF16 GEMV 的 CTA 数更多，也不自动意味着它读取的总字节更少或每个 CTA 更高效。

### 5.3 小 M 与分块/缩放开销

源码中存在按 block 处理、缩放因子读取及 `tl.dot` 的矩阵计算。默认配置包含 BM=64、BN=128、BK=128，但可能被调优配置覆盖；**当前 trace 没有直接证明此次所有 kernel 的实际 BM/BK 调优值**。

M=1 时通用 GEMM tile 的组织方式可能带来额外的数据布局、控制或计算成本，block scaling 也需要处理缩放因子。源码里的 `% M` 说明 M 方向索引的组织方式，不能直接推出“硬件实际做了 64 倍有效 FLOP”；编译器、掩码、布局与指令生成都影响最终成本，需要看生成代码和 NCU。

FP8 权重理论字节更少，但如果 CTA 数过少，GPU 可能无法发出足够多的并发访存，实际带宽达不到 BF16 GEMV。反过来，如果 NCU 显示带宽已接近峰值，则应优先检查实际读取流量、缓存行为和冗余访存，而不是继续归因于“没有喂满”。

### 5.4 哪些说法目前没有证据

不能直接写成结论的包括：“FP8 没有使用 Tensor Core”“238 个寄存器已经导致溢出”“FP8 down 已经达到显存带宽上限”“所有时间差都是动态量化造成”。这些都需要下一节的计数器证实，或被计数器结果否定。

## 6. 在原 Linux GPU 环境执行 NCU

提供两个实际文件：

- `capstone/quantization_profiling/scripts/profile_offline_ncu.sh`：按阶段、投影和采集详细度运行 NCU，并自动导出结果。
- `capstone/quantization_profiling/scripts/profile_kernels.py`：复用原冻结请求，完成模型初始化、5 次 warmup，在实际 vLLM CUDA worker 内打开 profiler gate。

本篇命令适用于原 Linux 环境。不要在当前 Windows RTX 5060 上跑完后把结果当作 RTX 4090D 的结论。

脚本已通过 10 项本地验证：Python AST、Git Bash 语法、三个模拟 worker 控制流程和五组 wrapper 参数展开检查；本机 NCU 2025.3.1 也核验了所用 section ID。证据为 `out/ncu_fp8_bf16_validation/validation.json`，其中明确记录 `gpu_collection=false`。这些检查证明脚本语法、参数和控制顺序，不能代替原机 GPU 采集。
可点击查看[验证记录](../../../out/ncu_fp8_bf16_validation/validation.json)和[验证脚本](../../../out/ncu_fp8_bf16_validation/validate_delivery.py)。在当前 Windows 工作区、已安装 Git Bash 的条件下可复跑：

```powershell
python -m unittest discover -s capstone/quantization_profiling/tests -p test_offline_profile_cli.py
```

### 6.1 准备与版本记录

在原机运行：

```bash
cd /workspace/AI_system
source /workspace/venvs/vllm-0112/bin/activate

export MODEL_ROOT=/workspace/weights_pub
export REQUESTS=/workspace/AI_system/capstone/quantization_profiling/data/frozen/requests.jsonl
export NCU_OUT=/workspace/AI_system/capstone/quantization_profiling/log/ncu_fp8_bf16
export NCU_GRAPH=1
export NCU_CLOCK_CONTROL=none
export NCU_CACHE_CONTROL=all

mkdir -p "$NCU_OUT"
nvidia-smi > "$NCU_OUT/gpu_environment.txt"
ncu --version > "$NCU_OUT/ncu_version.txt"
ncu --list-sections > "$NCU_OUT/ncu_sections.txt"
python -m pip show vllm torch triton > "$NCU_OUT/python_packages.txt"
```

若安装路径不同，修改仓库、venv、权重和冻结请求路径。`NCU_BIN` 可指定 NCU 可执行文件；`PYTHON_BIN` 可指定原 venv 的 Python。脚本要求 vLLM **0.14.1**，因为本方案依赖这一版本的 worker profiler iteration gate。

原路径约定中，`MODEL_ROOT` 下应存在 `Qwen3-8B` 和 `Qwen3-8B-FP8-local`。脚本会验证模型格式；它不会通过把 dtype 改成 BF16 来伪装量化权重，也不会改写模型文件。

### 6.2 第一步：两份轻量结果确认四个投影的真实 launch

```bash
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode first4 basic
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode first4 basic
```

`first4` 采匹配矩阵 kernel 的前 4 次，预期对应第一层的 qkv、o、gate_up、down。先在报告里核对顺序、grid 和 block，预期如第 5.1 节。**这 4 次是四种不同形状的筛选，不是每种形状的重复统计。**

新环境如果换了 attention/linear backend、Triton 调优配置或融合方式，kernel 名称和启动结构可能变化。若 `first4` 与本篇几何结构不符，先确认映射再运行下面的 skip 定位命令，不能强行按旧顺序解释。

### 6.3 第二步：优先详细采 down、o、gate_up

```bash
# 最大退化来源：down_proj。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode down detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode down detail

# 相对退化最大：o_proj。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode o detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode o detail

# 绝对耗时最大、且 FP8 已有收益：gate_up_proj。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode gate_up detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode gate_up detail
```

每条命令只详细分析一个匹配的矩阵 launch。它代表第一层该投影；若要解释跨层平均，还需挑选中间层/末层做相同形状复核，不能把一个 launch 当成 36 层耗时分布。

### 6.4 第三步：补采 qkv 和额外量化开销

```bash
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode qkv detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode qkv detail

bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode quant detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode norm_quant detail
```

quant 与 norm_quant 属于 FP8 路径，wrapper 不接受把这两项指定给 BF16。归一化融合的公平比较，需要连同被替代的 BF16 归一化工作一起看，不能直接比较单个名字。

### 6.5 第四步：prefill 对照，确认小 M 影响

```bash
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local prefill first4 basic
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 prefill first4 basic

bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local prefill gate_up detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 prefill gate_up detail

# 需要区分 down 的 M=1 与 M=128 时，再补这两份。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local prefill down detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 prefill down detail
```

prefill 下 BF16 使用 CUTLASS BF16 GEMM，decode 下使用 GEMV，wrapper 已分别选择 regex，不能复用 decode 的 GEMV 名字筛 prefill。

### 6.6 wrapper 为什么能排除初始化和 profile_memory

核心组合是：

```text
ncu --target-processes all --profile-from-start off
    --replay-mode kernel --graph-profiling node
```

应用先完成模型加载、profile_memory、编译/capture 及 5 次完整 warmup，然后调用 vLLM `start_profile()`。vLLM 0.14.1 的 worker gate 在真正持有 CUDA context 的执行进程内开启 CUDA profiler：

| 目标阶段 | delay_iterations | max_iterations | 目的 |
|---|---:|---:|---|
| decode | 8 | 2 | 跳过正式请求的 prefill 和前几个 decode，采稳定 decode 中的少量 launch |
| prefill | 0 | 1 | 采正式请求的第一次模型迭代，即该请求的 prefill |

实际打开/关闭边界应以原机日志为准。检查 NCU 报告是否产生、worker 的 profiler start/stop 信息，以及采中的 grid，确认阶段没有错位。如果版本变化、gate 失效或日志显示并未进入 CUDA profiler，则不能沿用结果。

`--target-processes all` 使 NCU 跟踪 vLLM 子进程。父 Python 进程中单独调用 CUDA profiler start，或只在父进程 `generate()` 周围写 NVTX，并不足以控制另一个 CUDA worker 的采集。

`--launch-skip` 在本脚本中数的是 **匹配矩阵名字的 launch**：qkv=0、o=1、gate_up=2、down=3。它用于稳定区间内选择第一层投影，**不承担跳过模型加载或 warmup 的任务**。这些阶段靠 worker profiler gate 排除。

### 6.7 实际展开的 NCU 选项

wrapper 会选择对应模型的 kernel regex，并以如下形式调用 NCU；路径、prefix 及 gate 由脚本填入：

```bash
ncu \
  --target-processes all --profile-from-start off \
  --replay-mode kernel --graph-profiling node \
  --nvtx --nvtx-include 'quantization_profiling@worker\/decode/' \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*_w8a8_triton_block_scaled_mm.*' \
  --filter-mode global --launch-skip 3 --launch-count 1 \
  --cache-control all --clock-control none \
  --section LaunchStats --section SpeedOfLight --section Occupancy \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats --section SourceCounters \
  --export REPORT_PREFIX \
  python capstone/quantization_profiling/scripts/profile_kernels.py \
  --profile fp8-local --requests-file "$REQUESTS" --case w128_o128 \
  --count 1 --batch-size 1 --warmup 5 --batch-tokens 2048 \
  --profiler cuda --delay-iters 8 --max-iters 2 --graph \
  --nvtx --event-dir REPORT_PREFIX.events \
  --output WORKLOAD_JSON
```

这是 decode down/detail 的展开示意；实际使用前面的 wrapper 命令，会自动建立唯一输出名，不需要手填 `REPORT_PREFIX` 和 `WORKLOAD_JSON`。

不要把 `--graph-profiling graph` 或 `--replay-mode range` 替换进这条命令来做单 kernel 归因：前者将整图作为一个 workload，后者聚合整个范围，且 range replay 对支持的 CUDA API 有限制。`--graph-profiling node` 与 kernel replay 用于当前的逐节点分析。

如果旧版本 NCU 在 Graph node 采集上有兼容问题，可先保留错误日志，再用 `NCU_GRAPH=0` 跑同一投影核验 kernel。此时会切为 eager，并改变编译/融合及量化路径，应单独标记，不能称为 no_eager 的等价测量。

### 6.8 输出文件、导出与打开

每次 wrapper 调用自动保存：

```text
<prefix>.ncu-rep          NCU 原始报告
<prefix>.metrics.csv      全部已采指标
<prefix>.details.txt      各 section 与规则说明
<prefix>.log              运行与采集日志
<prefix>.settings.txt     kernel regex、skip/count、Graph/cache/clock 配置
<prefix>.version.txt      NCU 版本
<prefix>.workload.json    工作负载配置、请求 SHA256、Python 包版本及诊断耗时
```

手工导出/打开同样可以使用：

```bash
ncu --import /path/to/report.ncu-rep --page raw --csv > report.metrics.csv
ncu --import /path/to/report.ncu-rep --page details > report.details.txt
ncu-ui /path/to/report.ncu-rep
```

将 `.ncu-rep` 拷回 Windows 后可直接用 Nsight Compute 打开。建议使用与采集端相同或能够读取该报告格式的更新版 UI；不要用 Nsight Systems 打开 `.ncu-rep`。

## 7. 怎样用 NCU 证实或否定候选原因

basic 阶段采 LaunchStats、SpeedOfLight、Occupancy；detail 在此基础上增加 MemoryWorkloadAnalysis、ComputeWorkloadAnalysis、SchedulerStats、WarpStateStats、SourceCounters。先看总体资源与工作量，再看 stall，避免只根据最高的一条 stall 百分比归因。

| 待验证问题 | 重点 section / 指标 | 支持该解释的现象 | 否定或需要改判的现象 |
|---|---|---|---|
| FP8 down/o 的 grid 是否过小 | LaunchStats 的 grid、waves；Occupancy；SM 利用情况 | 32 CTA、waves 很少，整个设备的活跃程度偏低；gate_up 明显改善 | 实际 kernel 配置已改变，或设备侧吞吐已被另一资源限制 |
| 是否受显存访问限制 | MemoryWorkloadAnalysis 的 DRAM 读写字节、带宽、L2 命中与事务 | DRAM 带宽接近限制，时间随实际读写流量变化 | DRAM 带宽很低且 issue/活跃 SM 也低，此时更像并发不足或延迟未隐藏 |
| FP8 权重减半是否兑现成流量减少 | DRAM bytes、L2 bytes/transactions，按同一投影比较 | 总流量显著减少，并带来时间下降 | 额外重复读、缩放或布局开销抵消，或缓存状态不同导致比较不可用 |
| 高寄存器是否限制 resident blocks | LaunchStats registers；Occupancy 的寄存器/共享内存限制；local load/store | 资源限制与低 active warps 同时出现；确有 local 访存 | 没有 spilling，或 grid 过小已是更早约束；不能仅凭 238 registers 归因 |
| Tensor Core 是否实际参与、是否有效 | ComputeWorkloadAnalysis 的 tensor 管线、InstructionStats/Source/SASS 按需 | 可见相应矩阵指令，且管线活跃；不同 M 的利用明显变化 | 未出现预期指令或 tensor 活跃很低，需结合指令与占用确定原因 |
| 长 K 是否导致依赖与访存等待 | SchedulerStats 的 eligible/issued；WarpStateStats；SourceCounters | down 的 issue 较差、等待更多，且与 source 中 K 循环对应 | issue 已充分、主要资源已满，则长 K 本身不能解释退化 |
| 量化是否成为主要附加成本 | quant/norm_quant 的时长、DRAM 流量、launch 与 issue | 同类操作跨层累计后与 nsys 差异量级一致 | 单次虽慢但次数少，或它同时替代了 BF16 工作，不能算全部新增 |

注意两层 occupancy 的区别：一个获得 CTA 的 SM 内 resident warps 可能并不少，但只有少量 SM 分到工作，整卡仍然无法充分利用。反过来，achieved occupancy 低也不是独立的性能结论，需要结合 issue、资源限制和实际时长。

NCU 版本与架构会影响可用 metric 名字。优先使用上述 section，若要精简为单项指标，先在原机查询：

```bash
ncu --query-metrics-mode suffix \
  --metrics gpu__time_duration,dram__bytes_read,dram__bytes_write,sm__warps_active,smsp__issue_active
```

这些是查询命令，不能保证所有目标机都有相同的 metric 变体。采集日志若出现 metric 不支持，应按本机 query 输出调整；不要将 unavailable 当作数值 0。

## 8. 缓存、频率与重放对结果的影响

NCU 为采集硬件计数器可能多次重放同一个 launch，并保存/恢复相关显存。kernel replay 会改变工作之间的原始并发关系；默认 cache-control 会清空缓存。它不能还原 vLLM 每个请求的 CPU 提交间隙与原始请求吞吐。

当前 wrapper 默认 `NCU_CACHE_CONTROL=all`、`NCU_CLOCK_CONTROL=none`，并把选择写入 settings：

- `cache-control all` 让不同 replay pass 使用更一致的冷缓存条件，适合首先解释实际流量与带宽，但与正常执行的缓存状态有差异。
- `clock-control none` 不由 NCU 改变时钟；它本身不保证两次测量频率相同。需要保存实际频率，并在比较时使用一致条件。
- 若使用固定基准频率，可把两种精度都设为 `NCU_CLOCK_CONTROL=base`，与动态频率实验分开保存。
- `cache-control none` 仅仅表示不主动清缓存，并不保证多 pass 时每次保留真实应用的缓存状态。不能因为数值更接近 nsys 就默认采用。
- 如果专门研究真实缓存继承，可考虑 deterministic application replay；它会反复启动整个模型，成本高，且需要验证跨进程、stream、launch 匹配稳定。本篇默认方案不使用它。

Nsight Compute 与 Nsight Systems 的 duration 可能因锁频、缓存、序列化与采集机制不同而不一致；NCU 的总命令运行时间包含初始化、重放和采集，不能当作模型推理延迟。官方说明见 [Workload Durations / Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

## 9. 优化之后怎样验证

首先用相同 NCU 配置复采目标 kernel，确认变化确实减少了对应开销，例如 down_proj 增加了并行工作、减少了等待或降低了实际流量。随后关闭 profiler，在原冻结请求上复测模型，确认改善没有被量化/调度等其他开销抵消。

以下是无 profiler 的离线诊断，不需要重做 nsys。每次使用新的输出文件名：

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
python capstone/quantization_profiling/scripts/profile_kernels.py \
  --profile bf16 --requests-file "$REQUESTS" --case w128_o128 \
  --count 5 --batch-size 1 --warmup 5 --batch-tokens 2048 \
  --profiler none --graph \
  --output "$NCU_OUT/bf16_no_profiler_${stamp}.json"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
python capstone/quantization_profiling/scripts/profile_kernels.py \
  --profile fp8-local --requests-file "$REQUESTS" --case w128_o128 \
  --count 5 --batch-size 1 --warmup 5 --batch-tokens 2048 \
  --profiler none --graph \
  --output "$NCU_OUT/fp8_local_no_profiler_${stamp}.json"
```

这里记录的是固定请求的离线 wall time，含模型循环与调度，不是单 kernel duration，也不是正式 HTTP 服务指标。改变 tile、split-K、量化融合或后端后，除了速度，还应完成输出数值/模型质量验证；只凭计数器变好不能判断量化方案可替换。

## 10. 收到 NCU 报告后的填写模板

当前表内待填项表示没有实测结果，不表示 0。

| 指标 | BF16 down | FP8 down | FP8 o | FP8 gate_up |
|---|---:|---:|---:|---:|
| NCU 版本 / GPU / cache / clock | 待填 | 待填 | 待填 | 待填 |
| M/N/K，层号，Graph/eager | 待填 | 待填 | 待填 | 待填 |
| grid / threads / registers / shared | 待填 | 待填 | 待填 | 待填 |
| GPU duration | 待填 | 待填 | 待填 | 待填 |
| DRAM read/write bytes 与带宽 | 待填 | 待填 | 待填 | 待填 |
| L2 命中及事务 | 待填 | 待填 | 待填 | 待填 |
| achieved occupancy / waves | 待填 | 待填 | 待填 | 待填 |
| SM / tensor 管线利用 | 待填 | 待填 | 待填 | 待填 |
| eligible / issued warps | 待填 | 待填 | 待填 | 待填 |
| 主要 stall 与 source 位置 | 待填 | 待填 | 待填 | 待填 |
| local load/store / spilling 证据 | 待填 | 待填 | 待填 | 待填 |

每条最终归因应按“nsys 量级 → NCU 证据 → 源码位置 → 改动 → 无 profiler 复测”组织。例如：

```text
观察：down_proj 每步多 1.265437 ms，且只有 32 个 CTA。
假设：单 token、小 N、长 K 的工作分配使并行度不足。
NCU 证据：待填，包含 grid waves、issue、带宽和资源占用。
反证检查：是否实际已达带宽限制，是否有大量重复流量或 spilling。
改动与结果：待填；同时报告目标 kernel 和完整请求的变化。
```

## 11. 官方参考

- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)：kernel regex/name base、launch skip/count、子进程、profiler start/stop、Graph node、section 和结果导出。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)：replay、Graph、指标含义及跨工具时长差异。
- [Nsight Compute Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/)：版本支持与行为变更。不要把 Nsight Systems 版本当作 Nsight Compute 版本。
- [vLLM v0.14.1 FP8 Triton block-scaled 实现](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1039-L1222)：grid、K 循环、分块缩放及默认配置。实际生效的调优参数仍应从目标环境确认。
