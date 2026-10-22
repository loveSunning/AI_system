# vLLM 0.14.1：FP8 LinearOp 调用路径、硬件与后端

分析日期：2026-10-22。固定版本：v0.14.1；固定提交：**d7de043d55d1dd629554467e23874097e1c48993**。

依据为该提交源码，未在所有设备上实测。判断支持须同时考虑：模型格式、scheme、硬件、编译产物、dtype、shape、依赖、环境变量。

## 1. 缓存与完整性边界

- [离线交互调用图](reference/vllm_v0_14_1/callgraph.html)：搜索、缩放、七个视图、点击节点查看条件和源码。
- [完整 Mermaid](reference/vllm_v0_14_1/callgraph.mmd)、[结构化 JSON](reference/vllm_v0_14_1/callgraph.json)。
- 分图：[入口](reference/vllm_v0_14_1/entry.mmd)、[分块](reference/vllm_v0_14_1/block.mmd)、[普通](reference/vllm_v0_14_1/ordinary.mmd)、[激活量化](reference/vllm_v0_14_1/quant.mmd)、[CUDA GEMM](reference/vllm_v0_14_1/cuda.mmd)、[特殊路径和融合](reference/vllm_v0_14_1/special.mmd)。
- [来源与 SHA256](reference/vllm_v0_14_1/source_manifest.json)、[校验结果](reference/vllm_v0_14_1/validation.json)、[生成器](reference/vllm_v0_14_1/build_callgraph.py)。
- 源码快照在 reference/vllm_v0_14_1/source/，保留上游 LICENSE；仅供阅读，不作为运行依赖。

**完整性边界：该提交内 FP8 E4M3 稠密 Linear 的配置入口、权重后处理、普通/分块 LinearOp 所有控制分支、激活量化及内置 GEMM 后端，追踪到外部库 API 或内置 CUDA/Triton kernel。** PyTorch、DeepGEMM、FlashInfer、AITER、IPEX 的内部 kernel 分派不属于 vLLM 仓库，不能仅凭 vLLM 版本固定其内部实现。调用图人工追踪，节点由 AST/文本核对；边表示条件调用/选择关系，不表示所有下游在一次推理中都会执行。

2:4 sparse FP8 单列邻接路径。MoE、KV cache、attention、MXFP8、W4A8/FP4 为独立实现，本图不冒充其完整调用图。缓存含部分共用/上下文源码，不能据此将 INT8 ScaledMM 注册表混入 FP8 LinearOp。

## 2. 总体关系：两个 LinearOp，另外有绕行路径

~~~mermaid
flowchart TD
  A["Qwen3 Attention / MLP"] --> B["QKV / MergedColumn / RowParallelLinear.forward"]
  B --> C["quant_method.apply"]
  C --> D["Fp8LinearMethod：原生 fp8"]
  C --> E["CompressedTensorsLinearMethod"]
  E --> F["layer.scheme.apply_weights"]
  D --> G["Fp8LinearOp：tensor / channel"]
  D --> H["W8A8BlockFp8LinearOp：BLOCK"]
  F --> G
  F --> H
  D --> M["Marlin：W8A16"]
  D --> I["batch-invariant 非 BLOCK：BF16 DQ + F.linear"]
  G --> Q["QuantFP8 + scaled GEMM"]
  H --> K["FlashInfer / DeepGEMM / CUTLASS / AITER / Triton"]
  X["Intel XPU 原生 fp8 入口"] --> Y["XPUFp8LinearMethod → IPEX W8A16"]
~~~

核心源码：[普通 helper][S1]、[分块 helper][S2]、[原生方法][S3]、[compressed-tensors 入口][S4]、[W8A8 scheme][S5]。

### 2.1 模型实际调用位置

| 源模型逻辑权重 | vLLM 模块 | forward |
|---|---|---|
| q_proj、k_proj、v_proj | QKVParallelLinear：qkv_proj | 继承 ColumnParallelLinear.forward |
| o_proj | RowParallelLinear：o_proj | RowParallelLinear.forward |
| gate_proj、up_proj | MergedColumnParallelLinear：gate_up_proj | 继承 ColumnParallelLinear.forward |
| down_proj | RowParallelLinear：down_proj | RowParallelLinear.forward |

Qwen3MLP 是 Qwen2MLP 的别名。每层七个逻辑权重矩阵，常规未额外拆分路径对应四个线性模块调用；252 份投影权重不代表每次前向一定运行 252 个独立 GEMM kernel。编译融合、TP 和 backend 会改变 profiler 观察结果。

~~~text
Qwen3Attention.forward
  ├─ qkv_proj → ColumnParallelLinear.forward → quant_method.apply
  └─ attention 后 o_proj → RowParallelLinear.forward → quant_method.apply

Qwen3MLP / Qwen2MLP.forward
  ├─ gate_up_proj → ColumnParallelLinear.forward → quant_method.apply
  └─ SiLU+Mul 后 down_proj → RowParallelLinear.forward → quant_method.apply
~~~

### 2.2 汇入这两个 LinearOp 的 FP8 前端

| 方法 | 计算路径 |
|---|---|
| Fp8Config → Fp8LinearMethod | 普通或 BLOCK；另有 Marlin、batch-invariant 绕行 |
| CompressedTensorsW8A8Fp8 | tensor/channel → 普通；block → BLOCK |
| PTPCFp8LinearMethod | 普通，per-channel W + per-token A |
| FBGEMMFp8LinearMethod | 普通，带 input_scale_ub；名称不代表前向必须调用 FBGEMM 库 |
| ModelOptFp8LinearMethod | 普通，static per-tensor A |
| ModelOptFp8PcPtLinearMethod | 普通，dynamic per-token A |
| ModelOptFp8PbWoLinearMethod | BLOCK [128,128] + dynamic group128 A；调用方禁用 AITER |
| QuarkW8A8Fp8 | 普通，按配置 static/dynamic |
| CompressedTensorsW8A16Fp8 | 直接 Marlin |
| XPUFp8LinearMethod | 直接 IPEX W8A16 |

ModelOpt 的 FP8_PB_WO 名称容易误导：此版本前向实际通过 W8A8BlockFp8LinearOp 动态量化输入，并非 Marlin W8A16。

## 3. 三个不同时间点的分派

### 3.1 初始化

LinearBase.__init__ 调用 quant_config.get_quant_method。

- 原生 fp8：非 XPU 选择 Fp8LinearMethod；XPU 选择 XPUFp8LinearMethod。
- compressed-tensors：解析每层 scheme，由 CompressedTensorsLinearMethod 委托执行。
- 普通 LinearOp 预先决定 preferred_backend。
- BLOCK 预先决定 CUTLASS/AITER/Triton fallback；此时没有最终权重形状，尚不能决定 DeepGEMM。

### 3.2 权重加载后处理

~~~text
process_weights_after_loading
  ├─ 浮点 checkpoint：原生 fp8 可先量化权重
  ├─ 普通 FP8：合并/调整 logical shard scales，转置权重
  ├─ BLOCK：process_fp8_weight_block_strategy
  │    ├─ 适配 AMD FNUZ
  │    └─ DeepGEMM eligible：后处理 weight 与 scale
  └─ Marlin：重排/repack 权重与 scale，激活不量化
~~~

DeepGEMM 后处理可能将 scale 调整为 UE8M0 约束，并调整布局/打包。因此磁盘 checkpoint 的 scale 布局不一定等于运行时布局。不能手动修改 quant_method 或仅凭 scale 名称判断是否取倒数。[加载后处理源码][S2]

### 3.3 每次 forward

普通 FP8 根据实际 activation/weight scale 粒度选 GEMM。BLOCK 优先检查 FlashInfer、DeepGEMM 的 dtype/shape 条件，再用初始化 fallback。torch.compile 可融合量化；具体 kernel 名称和调用次数需要 profiler 确认。

## 4. 普通 FP8：Fp8LinearOp

源码位于 **utils/w8a8_utils.py**，不是 utils/fp8_utils.py。[S1]

~~~text
Fp8LinearOp.apply
  ├─ input 非 FP8 → QuantFP8(input, input_scale, input_scale_ub)
  ├─ input 已为平台 FP8 → 复用 input 与 input_scale
  ├─ 判断 per_tensor_weights / per_tensor_activations
  └─ dispatch_w8a8_scaled_mm
       ├─ rocm_per_tensor_w8a8_scaled_mm
       ├─ flashinfer_w8a8_scaled_mm
       ├─ cutlass_w8a8_scaled_mm
       ├─ torch_per_tensor_w8a8_scaled_mm
       ├─ torch_per_token_w8a8_scaled_mm
       └─ torch_channelwise_w8a8_scaled_mm
~~~

### 4.1 preferred_backend

按源码顺序：

1. ROCm → rocm。
2. CUDA 且 cutlass_fp8_supported：
   - has_flashinfer 且 has_device_capability(100) → flashinfer。
   - 否则 → cutlass。
3. 其余 → torch。

has_device_capability(100) 是 **>=100**，不是仅 SM100；它只是候选选择，仍要求 FlashInfer 的实际 API/kernel 支持。

torch fallback 且未进入 VLLM_COMPILE 时可将输入补到至少 17 行；与 Hopper BLOCK 补到 4 的倍数不同。

### 4.2 最终 dispatch 矩阵

| W scale / A scale | preferred_backend | 最终函数 |
|---|---|---|
| 都为 per-tensor | rocm | rocm_per_tensor_w8a8_scaled_mm |
| 都为 per-tensor | flashinfer | flashinfer_w8a8_scaled_mm |
| 都为 per-tensor | cutlass | cutlass_w8a8_scaled_mm |
| 都为 per-tensor | torch | torch_per_tensor_w8a8_scaled_mm |
| 至少一个非 per-tensor | cutlass 或 flashinfer | **cutlass_w8a8_scaled_mm** |
| 双方非 per-tensor，rowwise 条件满足 | rocm/torch | torch_per_token_w8a8_scaled_mm |
| 其余组合 | rocm/torch | torch_channelwise_w8a8_scaled_mm |

rowwise 条件：**ROCm、torch>=2.7、capability>=94**。

A per-tensor 判断同时检查 numel 和 dim：单 token 的 per-token scale 形状可能为 [1,1]，虽然只有一个元素，仍不是 scalar per-tensor。

### 4.3 各 backend 的末端

- **CUTLASS**：helper → _custom_ops.cutlass_scaled_mm → 通常 _C.cutlass_scaled_mm → CUDA 架构分派。B 的 K/N 非 16 对齐，或平台为 ROCm，则 Python wrapper 转 generic triton_scaled_mm。函数名不保证实际 kernel。
- **FlashInfer**：helper → flashinfer_scaled_fp8_mm → bmm_fp8(backend="auto")。仅 scalar A/W scale；外部 auto 分派取决于实际包版本。
- **ROCm per-tensor**：skinny 条件满足 → ops.wvSplitKQ，否则 torch._scaled_mm。
- **PyTorch per-tensor / rowwise**：torch._scaled_mm，真实 scale 在 GEMM 中应用。
- **PyTorch channelwise fallback**：单位 scale 的 FP32 GEMM，之后单独乘 A/W scale、加 bias、cast。

skinny 全部条件：VLLM_ROCM_USE_SKINNY_GEMM、MI3xx（gfx942/gfx950）、M=1、K%16=0、bias 为空或 dtype 与输出一致。

## 5. BLOCK FP8：W8A8BlockFp8LinearOp

本项目 quantize_fp8.py 的 FP8_BLOCK 主要走这里。[S2]

~~~text
W8A8BlockFp8LinearOp.apply
  input_scale 必须为 None；展平 input 为 [M,K]
    ├─ FlashInfer predicate AND DeepGEMM predicate
    │    └─ _run_flashinfer → vllm.flashinfer_fp8_blockscale_gemm
    │         └─ torch.cond(M<32)
    │              ├─ 真：FlashInfer SM90 swapAB，内部量化 BF16 input
    │              └─ 假：group quant → deep_gemm.fp8_gemm_nt
    ├─ 否则 DeepGEMM predicate
    │    └─ _run_deepgemm
    │         ├─ packed UE8M0 quant 或 colmajor QuantFP8
    │         └─ vllm.fp8_gemm_nt_op → wrapper → deep_gemm.fp8_gemm_nt
    └─ 否则初始化 fallback
         ├─ CUTLASS block supported → _run_cutlass
         ├─ 否则 AITER enabled/supported → _run_aiter
         └─ 否则 → _run_triton
~~~

结束时处理 bias、转回 input dtype，恢复输出形状。

### 5.1 DeepGEMM

全部条件：

- CUDA；**SM90 exact 或 SM100 family（10.x）**；
- deep_gemm 包存在；VLLM_USE_DEEP_GEMM=1（默认）；
- 输出 BF16；
- W=[N,K]，N%64=0，K%128=0。

SM11x、SM12x 不满足 SM100 family。不能将注释中的 Blackwell 泛化到所有 Blackwell 产品。[判定源码][S6]

激活 scale 的 oracle：

| 格式 | 含义 |
|---|---|
| FLOAT32 | 普通浮点 scale |
| FLOAT32_CEIL_UE8M0 | scale 向上调整到 UE8M0 约束，仍以 float32 保存 |
| UE8M0 | 每 4 个 scale 打包到 int32，使用 DeepGEMM 布局 |

VLLM_USE_DEEP_GEMM_E8M0 默认 1。包/API 满足时，10.x 采用 packed UE8M0；Hopper 为 FLOAT32_CEIL_UE8M0。开关可改变 scale 路径和数值。

### 5.2 FlashInfer BLOCK

全部条件：

- VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=1，**默认 0**；
- FlashInfer 及 fp8_blockscale_gemm_sm90 API 存在；
- **SM90 exact**；
- input/output BF16、W 为 float8_e4m3fn；
- N%64=0、K%128=0；
- apply 还要求 **DeepGEMM predicate 成立**。

所以关闭 DeepGEMM 可能同时阻止这条 FlashInfer BLOCK 路径。[判定源码][S7]

M 是展平后 **token 行数**，不是请求数。M<32 使用 FlashInfer swapAB；M>=32 在同一外层注册 op 内改走官方 DeepGEMM，不能只凭外层 op 名称判断实际库。

### 5.3 CUTLASS BLOCK

~~~text
_run_cutlass
  → QuantFP8(dynamic group, column_major_scales=True, use_ue8m0=False)
  → SM90：padded_cutlass，M 补到 4 的倍数
    其他：直接 helper
  → fp8_utils.cutlass_scaled_mm，转置 W/scales
  → _custom_ops.cutlass_scaled_mm → _C.cutlass_scaled_mm
  → SM90 / SM100 / SM120 blockwise FP8
~~~

block predicate：90<=CC<100 要求 CUDA 编译版本>=12.0；CC>=100 要求>=12.8。还必须实际编译对应 arch kernel，并满足 dtype/shape/layout。predicate 不检查全部编译宏。

### 5.4 AMD AITER BLOCK

~~~text
_run_aiter
  ├─ 非 FNUZ + (N,K) 有 tuned Triton 配置
  │    ├─ vLLM group quant wrapper，ROCm 实际走 Triton
  │    └─ aiter.ops.triton.gemm_a8w8_blockscale
  └─ 否则
       ├─ AITER get_hip_quant(QuantType.per_1x128)
       └─ aiter.gemm_a8w8_blockscale，CK API
~~~

要求 AITER 安装、ROCm/gfx9 检查通过、主开关和 LINEAR 开关开启；activation group=(1,128)。gfx942 为 FNUZ，不能进入非 FNUZ tuned 分支；gfx950 仍需 shape 有 tuned 配置。

AITER 的 gfx9 package gate 包含 gfx90a，**不代表 MI200 原生支持 FP8**；还要与 supports_fp8 的硬件判断求交集。

### 5.5 Triton BLOCK

~~~text
_run_triton
  → QuantFP8(dynamic group, row-major scales, use_ue8m0=False)
  → vllm.w8a8_triton_block_scaled_mm_func
  → _w8a8_triton_block_scaled_mm_func
  → w8a8_triton_block_scaled_mm
  → _w8a8_triton_block_scaled_mm[grid]：tl.dot + block scales
~~~

这个分块 GEMM 与普通 fallback 的 compressed_tensors.triton_scaled_mm 不同。

**Ada SM89 没有 CUTLASS BLOCK、DeepGEMM、FlashInfer SM90 路线，本项目 FP8_BLOCK 在常规 CUDA 配置下走 Triton GEMM；激活量化仍可走 CUDA kernel。**

## 6. 激活量化：所有常用入口

源码：[QuantFP8][S8]、[CustomOp 平台分派][S9]、[底层 wrapper][S10]。

~~~text
QuantFP8.__call__ → CustomOp.forward → 预先选定的 forward 方法
  ├─ forward_cuda
  │    ├─ dynamic group → per_token_group_quant_fp8
  │    │    ├─ CUDA + contiguous → _C.per_token_group_fp8_quant
  │    │    └─ 否则 → Triton rowmajor / colmajor kernel
  │    └─ static / dynamic tensor / dynamic token → ops.scaled_fp8_quant
  │         ├─ scale 已提供 → _C.static_scaled_fp8_quant
  │         ├─ dynamic tensor → _C.dynamic_scaled_fp8_quant
  │         └─ dynamic token → _C.dynamic_per_token_scaled_fp8_quant
  ├─ forward_hip
  │    ├─ AITER + non-group + contiguous + 无 scale_ub
  │    │    └─ AITER per_tensor_quant / per_token_quant
  │    └─ 否则 → forward_cuda 的兼容代码路径，底层仍为 HIP/Triton
  └─ forward_native
       └─ PyTorch 算子实现，可由 torch.compile 生成 kernel
~~~

DeepGEMM packed 分支绕过普通 QuantFP8：

~~~text
per_token_group_quant_fp8_packed_for_deepgemm
  → _C.per_token_group_fp8_quant_packed
  → C++ per_token_group_quant_8bit_packed
  → CUDA group quant + packed UE8M0 scale
~~~

关键区别：

1. 普通 group quant 要求 K 可被 group_size 整除，最后一维连续；CUDA 全连续输入优先 C++ kernel，否则 Triton。
2. packed 版本仅用于 CUDA/DeepGEMM，要求 UE8M0；每个 int32 保存四个 scale，含 TMA 布局对齐。
3. AITER BLOCK 和 FlashInfer 小 M 路线可使用各自的量化实现，不经过普通 QuantFP8。
4. QuantFP8 有 CPU/TPU native fallback 机制，不等于这些平台支持整条 FP8 GEMM。
5. FP8 权重不意味着激活一定 FP8：Marlin、IPEX、batch-invariant BF16 分支有不同语义。

## 7. CUDA C++ 的实际分派与编译条件

共同入口：[scaled_mm_entry.cu][S11]；编译控制：[CMakeLists.txt][S12]。

~~~text
_custom_ops.cutlass_scaled_mm
  → torch.ops._C.cutlass_scaled_mm
  → C++ cutlass_scaled_mm
       ├─ ENABLE_SM120 且 actual SM>=120
       │    → cutlass_scaled_mm_sm120
       ├─ ENABLE_SM100 且 100<=actual SM<120
       │    → cutlass_scaled_mm_sm100
       ├─ ENABLE_SM90 且 90<=actual SM<100
       │    → cutlass_scaled_mm_sm90
       ├─ C2X 且 SM89
       │    → cutlass_scaled_mm_sm89（普通 FP8）
       └─ SM80/SM75 路径也在同入口内，但相关函数只处理 INT8，不等于 FP8 支持
~~~

SM90/100/120 wrapper 各自区分普通 FP8 与 blockwise FP8，然后进入同目录 c3x 下的具体实现和 dispatch.cuh，按输出 dtype、M/N/K 等选择 template/tile。最终通过 cutlass_gemm_caller 构造 GemmUniversalAdapter，并调用 gemm_op.run。缓存保留了这些 dispatcher 和 launcher。

| 路径 | capability predicate / 编译条件 | 备注 |
|---|---|---|
| 普通 FP8 CUTLASS SM89 | CUDA 编译版本>=12.4；有 C2X arch 产物 | 不是 CUDA 12.0 即可 |
| 普通 FP8 SM90 | CUDA>=12.0；编译 SM90 arch | tensor/channel scale |
| BLOCK FP8 SM90 | CUDA>=12.0；编译 SM90 arch | 输入 group128，权重 block128x128 |
| 普通/BLOCK SM100 | 编译 SM100 组，CUDA>=12.8 | 常规 arch 包括 10.0a/10.1a/10.3a |
| 普通/BLOCK SM120 | 编译 SM120 组，CUDA>=12.8 | 常规 arch 12.0a |
| CUDA>=13 的 family build | 源码可编译 10.0f、11.0f、12.0f 对应组 | 实际可执行仍由构建产物和 kernel 限制决定 |

“CUDA>=xx”指编译该扩展的 CUDA toolkit，不是单看驱动版本或本机 nvcc 输出。Python capability helper、C++ runtime arch dispatch、构建宏三者须同时成立。

## 8. 全部内置硬件分支与后端矩阵

下表是 **该版本源码的路由/限制表**。表中存在分支不代表已逐个 GPU SKU 实测；官方概览表较粗，未穷举 Blackwell，且与新增的受限分支存在粒度差异。以源码条件为定位依据，验收仍需实际启动和 profiler。

| 硬件/能力族 | 普通 W8A8 FP8 | BLOCK W8A8 FP8 | 权重 FP8、激活16位及限制 |
|---|---|---|---|
| NVIDIA Volta SM70/72 | 无本次支持路径 | 无 | 低于 Fp8Config 最小75及 Marlin 门槛 |
| NVIDIA Turing SM75 | 无原生 W8A8 | 无原生 W8A8 | native fp8 源码可达 Marlin；generator 有受限 SM75 FP16 配置。官方概览未标支持，不作为通用支持承诺 |
| NVIDIA Ampere SM80/86/87 | 无原生 W8A8 | 无原生 W8A8 | native fp8 走 Marlin W8A16；CT tensor/channel 可回退，CT BLOCK 不兼容该 W8A16 scheme |
| NVIDIA Ada SM89 | CUTLASS 普通 FP8；特殊形状可能 generic Triton；不可用时 torch fallback | **Triton BLOCK** | DeepGEMM/FlashInfer BLOCK/CUTLASS BLOCK 均不满足本版条件；可测试强制 Marlin |
| NVIDIA Hopper SM90 | CUTLASS，或 torch fallback | DeepGEMM 优先；可 opt-in FlashInfer；否则 CUTLASS；再 fallback Triton | FlashInfer 内部按 M<32 再选择；2:4 sparse 也只放行 SM90 |
| NVIDIA 10.x（SM100/101/103等） | scalar scale 时可候选 FlashInfer；非 scalar 回 CUTLASS；需对应包/kernel | DeepGEMM eligible 时优先，否则 CUTLASS SM100 | E8M0 oracle 可采用 packed UE8M0 |
| NVIDIA 11.x | 普通逻辑可能选择 FlashInfer；需实际依赖与构建 | 不满足 DeepGEMM family100 或 FI block gate；适配构建时可走 C++ SM100组 | CUDA13 的11.0f是构建候选，不能泛称所有 GPU 均支持 |
| NVIDIA SM120/12.x | scalar scale 时可候选 FlashInfer，其余 CUTLASS SM120；需实际构建 | CUTLASS SM120；**不走本版 DeepGEMM/FlashInfer BLOCK** | 12.x 不等于10.x；源码有 SM120 实现 |
| AMD gfx94x（如 gfx942 MI300/MI325） | ROCm torch/hipBLASLt 路线；M=1可skinny；量化可AITER | AITER启用时CK；否则Triton | gfx94 使用 E4M3FNUZ，加载时 normalization；gfx942 不能走非FNUZ tuned AITER Triton分支 |
| AMD gfx95x（如 gfx950） | ROCm torch/rowwise；量化可AITER；可skinny | AITER CK或有tuned配置的AITER Triton；否则vLLM Triton | 通常 E4M3FN；AITER route仍需具体shape和库支持 |
| AMD gfx12（RDNA4族） | 平台 supports_fp8 为真，具体 PyTorch/ROCm route须支持 | 非gfx9，AITER gate不通过，fallback为Triton | 不从 supports_fp8 单独推导所有模型已支持 |
| AMD gfx90a MI200、gfx11及其他无FP8能力族 | 该源码不将其识别为原生FP8支持 | 不保证可用 | gfx90a 通过AITER package gate不代表FP8硬件支持 |
| Intel XPU | 该原生入口不提供W8A8 | 不从此路径承诺BLOCK支持 | Fp8Config → XPUFp8LinearMethod → torch_ipex.fp8_gemm_w8a16，依赖IPEX实际硬件/API支持 |
| x86 CPU | 无本次稠密W8A8 Linear支持路径 | 无本次稠密BLOCK Linear支持路径 | 有SGL派生的CPU FP8权重内部算子，不等于本LinearOp完整支持 |
| ARM/POWER/RISC-V CPU | 无本次FP8 Linear支持路径 | 无 | 能用CPU做离线量化，不代表能用vLLM跑原生FP8 GEMM |
| TPU、Gaudi、Ascend及其他out-of-tree平台 | 不属于本提交这两条内置LinearOp的已证实后端 | 同左 | 外部插件/独立项目须另查对应版本，不沿用CUDA支持表 |

### 8.1 AMD 判断精确到源码

RocmPlatform.supports_fp8 检查 gcnArchName 是否含 gfx94、gfx95 或 gfx12；is_fp8_fnuz 仅检查 gfx94。

AITER 包支持判断额外要求 on_gfx9，枚举 gfx90a/gfx942/gfx950。启用 requires AITER package 和两个 env；因此不能用“所有 AMD GPU 都有 AITER FP8”替代实际条件。[AMD 平台源码][S13]、[AITER 适配][S14]

### 8.2 CPU 边界

CPU 的 gemm_fp8.cpp 在 AVX512、AVX512BF16、AVX512VNNI 等编译条件下加入，执行中可将 FP8 权重转 BF16 后使用 CPU GEMM。这不是这里 W8A8 LinearOp 的 CPU backend；同版本 CPU bindings 还把若干 FP8 activation quant op 定义成 placeholder。不能因为看到 CPU FP8 C++ 文件就把完整 dense FP8 推理标成支持。

### 8.3 Intel XPU 边界

原生 Fp8Config 在 is_xpu 时走专用方法。apply 明确调用 torch_ipex.fp8_gemm_w8a16。源码不在这段中枚举 Intel GPU SKU，也没有据此证明所有 block checkpoint 都适配；支持范围取决于 IPEX op 和对应设备。[XPU 源码][S15]

## 9. 特殊情况

### 9.1 Marlin W8A16

原生 Fp8LinearMethod 的 use_marlin：

~~~python
not current_platform.has_device_capability(89) or VLLM_TEST_FORCE_FP8_MARLIN
~~~

ROCm 显式关闭；batch-invariant 也关闭。

路径：

~~~text
load-time prepare_fp8_layer_for_marlin
  → pack_fp8_to_int32 / gptq_marlin_repack / scale permutation
forward apply_fp8_marlin_linear
  → _custom_ops.gptq_marlin_gemm
  → _C.gptq_marlin_gemm
  → Marlin CUDA template kernel
~~~

FP8 是权重格式，激活保持 FP16/BF16；apply_fp8_marlin_linear 明确拒绝8位 input_dtype。

CT 在 W8A8 capability 不足时可选择 CompressedTensorsW8A16Fp8，但它只支持 tensor/channel，且最小 capability80。**本项目 FP8_BLOCK 不能据此无缝回退到 Ampere/Turing。** [Marlin helper][S16]

### 9.2 Batch-invariant

只在 native Fp8LinearMethod 的此分支讨论：

- BLOCK：仍委托 W8A8BlockFp8LinearOp。
- 非BLOCK：把权重反量化为 BF16，再调用 F.linear。

不能把“加载FP8文件”直接等同“该层一定执行FP8×FP8”。

### 9.3 2:4 sparse FP8 邻接路径

~~~text
CompressedTensors24.apply_weights
  → QuantFP8
  → _custom_ops.cutlass_scaled_sparse_mm
  → _C.cutlass_scaled_sparse_mm
  → cutlass_scaled_sparse_mm_sm90
~~~

support predicate 写 CC>=90、CUDA>=12.2，但**实际 C++ 分派只接受 SM90 exact**：源码说明9.0a kernel不向前兼容。因此不能凭 predicate 宣称 Blackwell 可用。

### 9.4 编译融合

- RMSNormQuantFusionPass：可将 norm/residual 与 static、per-token、group64/128 量化融合。
- ActivationQuantFusionPass 中的 FP8 pattern：SiLU+Mul+**static tensor**量化，不泛化成所有动态BLOCK。
- ROCm 有额外 AITER norm/group/activation 融合 patterns。
- 禁用某个 CustomOp 时可走 forward_native，再由 torch.compile 编译；具体行为取决于编译配置。

因此图中的逻辑“量化→GEMM”可能在 profiler 中表现成融合 kernel。**不能仅统计某个 quant kernel 名称是否出现来判断激活有没有量化。**

## 10. 项目当前最相关的两条路径

### 本地 quantize_fp8.py 产物

~~~text
compressed-tensors + FP8_BLOCK
  → CompressedTensorsLinearMethod.apply
  → CompressedTensorsW8A8Fp8.apply_weights
  → W8A8BlockFp8LinearOp.apply
  → 在SM89/4090D的常规配置：_run_triton
       ├─ QuantFP8 → forward_cuda → per_token_group_quant_fp8
       │    → contiguous时 _C.per_token_group_fp8_quant
       └─ w8a8_triton_block_scaled_mm → Triton GEMM
~~~

### 官方原生 FP8 BLOCK checkpoint

~~~text
quant_method=fp8 + weight_block_size=[128,128]
  → Fp8LinearMethod.apply
  → 无Marlin时 W8A8BlockFp8LinearOp.apply
  → 后续与上面共享BLOCK后端
~~~

两者入口、scale字段及加载格式不同，但可在后端汇合。仍不能断言量化结果、运行时scale、质量或性能完全相同。

## 11. 校验与使用

在项目 quantization_profiling 目录执行：

~~~powershell
python docs/reference/vllm_v0_14_1/build_callgraph.py
~~~

生成器只读取缓存文本，用标准库解析Python符号、生成图，并检查所有缓存文件SHA256、大小、节点引用和各视图无环。不会导入torch、vLLM或执行量化/推理。源码固定；缓存文件被修改时SHA256校验会失败。

callgraph.json 保留每个节点的文件、行号、固定提交链接、摘要和excerpt；每条关系保留条件、类型和源锚点。外部API节点单独标色；它们不是对外部库内部实现的版本承诺。

真实目标机验收需记录：GPU capability/gcnArchName、vLLM/torch/CUDA或ROCm版本、依赖实际版本、相关env、量化配置、tensor-parallel形状及 M/N/K，再用启动日志和 profiler 证实实际路径。本缓存完成的是源码级调用关系审计，未声称真实GPU性能验证。

## 12. 固定版本主要参考

[S1]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/utils/w8a8_utils.py
[S2]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/utils/fp8_utils.py
[S3]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/fp8.py
[S4]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
[S5]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py
[S6]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/utils/deep_gemm.py
[S7]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/utils/flashinfer.py
[S8]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/input_quant_fp8.py
[S9]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/custom_op.py
[S10]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/_custom_ops.py
[S11]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu
[S12]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/CMakeLists.txt
[S13]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/platforms/rocm.py
[S14]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/_aiter_ops.py
[S15]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/ipex_quant.py
[S16]: https://github.com/vllm-project/vllm/blob/d7de043d55d1dd629554467e23874097e1c48993/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py

官方概览表位于缓存 source/docs/features/quantization/README.md；其硬件标记比源码分支粗，不能代替上述精确条件。
