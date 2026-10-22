# P1：Qwen3-8B 量化与性能诊断

围绕同一份 Qwen3-8B 模型与固定负载，建立 BF16 基线，运行一条 W4A16 量化路径，结合 Nsight 时间线、热点 kernel 和留出集质量回归，解释显存、吞吐、尾延迟与质量之间的取舍。

**当前状态：用户提供的 BF16 五组负载 × 3 轮 × 100 请求实验已通过；现已扩展 BF16、AWQ、官方 FP8 和本地 FP8 的统一运行与 benchmark。** 新测试默认每组每轮 5 次预热、50 次正式请求、1 轮，可通过参数选择轮次和请求数。AWQ/官方 FP8/本地 FP8 的真实 4090D 推理、性能和质量回归仍需目标机验证；Profiler 诊断仍待完成。

**FP8 性能诊断与 Nsight 实操：** [官方/本地 FP8 与 BF16、AWQ 的逐步分析命令](docs/nsight_fp8_performance_analysis.md)。结合 2026-10-22 的 1×80、并发 1 结果，覆盖服务/离线采集、prefill/decode、NCU 指标与受控消融；实际 GPU 归因须在目标机采集验证。

**FP8 自行量化入口：** [LLM Compressor 分块 FP8 转换与验证](docs/fp8_quantization.md)。脚本使用 `FP8_BLOCK`，导出真实 compressed-tensors 权重；当前统一环境支持量化与推理；另保留 `torch==2.8.0+cpu` 的历史 CPU 量化方案。

**统一量化与推理环境：** [沿用 vllm-0112 目录的安装方案](#vllm-0112-统一量化与推理环境)。目标组合为 vLLM 0.14.1、LLM Compressor 0.9.0.5；目标机安装、量化和推理仍需执行验收。

**本周执行入口：[第一周完整运行与验收说明](docs/week1_execution.md)。** 原有短请求测试用于连通性检查；正式性能结果请使用新 benchmark 流程。

## 从哪里开始

1. 阅读 [详细学习与实施规划](planning.md)，按“环境与显存预算 → 数据与口径 → BF16 → Profiling → W4A16 → 质量回归 → 报告”的顺序推进。
2. 对照 [W29–W52 每周计划与验收](../../docs/learning-plan-w29-w52.md)，保留 P1-01～P1-06 的任务编号和验收要求。
3. 模型已下载后，按 [模型运行与库用法教程](docs/model_running.md) 打通短请求、批量推理和 HTTP 调用。
4. 复习 [Nsight 工具手册](../../docs/profiling/nsight-tools.md) 的时间线与热点分析方法；其中现有示例面向 kernel lab，本项目的模型服务采集流程将在后续实现。

计划来源为 `00.AI-system-plan-optimized-v3.xlsx` 的“案例1_量化与Profiling”，此处依据仓库已有周计划与案例说明展开。W29、W30、W32 是原学习周编号；实际补做时间和完成状态以实验记录为准。

## 要完成什么

| 任务 | 原周次 | 目标与交付 | 验收重点 | 当前状态 |
| --- | --- | --- | --- | --- |
| P1-01 | W29 | 环境、模型 revision、模板与生成参数锁定，启动入口 | 可重建环境，复现代表请求；读取实际显存 | BF16 已有真实结果；环境和来源信息仍需人工核对 |
| P1-02 | W29 | 五组 workload 的 BF16 原始记录和基线汇总 | 并发 1 起步；按所选轮数验收；OOM 如实记录 | 已有历史 3×100 基线；新默认 1×50，不覆盖历史结果 |
| P1-03 | W30 | nsys 时间线、3 个代表热点的 ncu 抽样与证据表 | 分清 prefill/decode；至少一个热点有证据和优化假设 | 未开始 |
| P1-04 | W30 | AWQ/W4A16 与官方/本地 FP8 的权重来源及对照结果 | 核验版本/硬件/kernel；相同输入与评测协议 | 四模型运行/配置/benchmark 已支持；真实对照待执行 |
| P1-05 | W30/W32 | 固定 100 条留出任务的逐题质量回归 | 与校准集分离；固定判分；实验前锁定质量门槛 | 数据与判分器已准备，配对回归待执行 |
| P1-06 | W32 | 性能、显存、质量与诊断报告及复现入口 | 每项结论可追溯到实验 ID；不预设加速收益 | 未开始 |

## 目录与文件职责

```text
capstone/quantization_profiling/
├── README.md                       # 项目入口与验收概览
├── planning.md                     # 学习路径、实施顺序与评测约定
├── requirements.txt                # 固定基础/工具依赖；vLLM 另用加速入口安装
├── requirements-quantization.txt   # 独立量化环境：LLM Compressor FP8_BLOCK
├── requirements-vllm-0112.txt      # 统一量化/推理：vLLM 0.14.1 / LLM Compressor 0.9.0.5
├── docs/
│   ├── model_running.md            # 本地推理、HTTP 服务与两个库的详细教程
│   ├── week1_execution.md           # 第一周协议、目标机命令与验收
│   └── fp8_quantization.md          # 自行制作分块 FP8、环境与产物验证
├── configs/
│   ├── bf16.yaml                   # 已实现：BF16 服务配置（JSON 格式的 YAML）
│   ├── awq.yaml                    # 已实现：AWQ/W4A16，awq_marlin + BF16
│   ├── fp8-official.yaml           # 已实现：官方 Qwen3-8B-FP8，原生 fp8 格式
│   ├── fp8.yaml                    # 已实现：本地 FP8，自动识别权重量化格式
│   └── workloads.yaml              # 已实现：五组负载、预热、重复次数与种子
├── scripts/
│   ├── collect_environment.py      # 已实现：并行环境检测、打印与 JSON 快照
│   ├── download_models.sh          # 已实现：批量下载、跳过已有模型与失败重试
│   ├── run_model.py                # 已实现：vLLM/Transformers、token 检查与 HTTP 服务
│   ├── model_profiles.py           # 已实现：四模型别名、量化配置与运行预设
│   ├── test_server.py              # 已实现：远程接口、并发测试、指标和自动日志
│   ├── profiling_middleware.py     # 已实现：服务端 JSON/SSE 耗时与吞吐量扩展
│   ├── quantize_fp8.py             # 已实现：FP8_BLOCK 转换、产物校验与指纹
│   ├── prepare_workloads.py        # 已实现：请求准备、token 校验与数据指纹
│   ├── serve.sh                    # 已实现：BF16 服务启动便捷入口
│   ├── benchmark.py                # 已实现：自动服务管理、正式压测、显存与失败记录
│   ├── evaluate_quality.py         # 已实现：留出集推理、严格判分与逐题结果
│   └── summarize.py                # 已实现：单次实验汇总与报告
├── tests/
│   ├── test_collect_environment.py # 环境探针、并发与预检行为测试
│   ├── test_download_models.py     # 模拟 CLI，验证跳过、批量下载与重试
│   ├── test_prepare_workloads.py   # 生成数量、参数优先级、快照与 benchmark 兼容
│   ├── test_run_model.py           # 模板、token 预算、输出切片与入口行为
│   ├── test_server_client.py       # 本地 HTTP，验证并发、日志、指标与错误处理
│   ├── test_quantize_fp8.py         # FP8 源/产物校验、路径保护与失败留档
│   └── test_profiling_middleware.py # ASGI JSON/SSE 指标、并发隔离和流式转发
├── log/                            # 自动生成 .log / .jsonl / .json；运行产物不提交 Git
├── data/
│   ├── performance.jsonl           # 保留占位；正式记录存于独立实验目录
│   └── quality_holdout.jsonl       # 已准备：100 条自编合成质量任务
└── reports/
    └── Project1_Report.md          # 验收状态；真实实验结论待补
```

文件的输入、输出、数据字段和后续生成产物约定见 [planning.md](planning.md)。模型权重、大型 trace 和运行日志的存放规则也在那里说明。

## Python 依赖说明与安装

本项目按环境使用独立的依赖入口：原 cutedsl-venv 使用 [requirements.txt](requirements.txt)，其中 vLLM 行保持注释，按旧版加速入口单独安装；新 vllm-0112 使用 [requirements-vllm-0112.txt](requirements-vllm-0112.txt)，已包含 vLLM 0.14.1、LLM Compressor 0.9.0.5 及固定的核心依赖。下面列出的是原环境安装前检测结果；新旧入口不要混用。SGLang 留给后续框架对照。

| pip 包名 / 常见名称 | Python 导入名 | 安装前检测版本 | 在本项目中的用途 |
| --- | --- | --- | --- |
| [torch / PyTorch](https://pytorch.org/) | `torch` | 2.8.0+cu129 | 张量计算、GPU 执行和模型前向计算的基础框架；`+cu129` 表示这个 wheel 的 CUDA 12.9 构建，不代表本机 nvcc 版本 |
| [vllm / vLLM](https://docs.vllm.ai/) | `vllm` | missing | 大模型推理与服务引擎，负责请求调度、批处理、KV 管理和模型执行；本项目的服务主线 |
| [transformers / Transformers](https://huggingface.co/docs/transformers/) | `transformers` | 4.57.6 | 提供模型结构、配置、分词器入口与生成接口；此前使用的 AutoModelForCausalLM、AutoTokenizer 就来自这里 |
| [tokenizers / Tokenizers](https://huggingface.co/docs/tokenizers/) | `tokenizers` | 0.22.2 | 以 Rust 实现的高性能分词组件，通常由 Transformers 的 fast tokenizer 调用 |
| [triton / Triton](https://github.com/triton-lang/triton) | `triton` | 3.4.0 | 使用 Python 编写与编译 GPU kernel 的语言和工具链，PyTorch/推理框架可能使用；与 NVIDIA Triton Inference Server 是两个项目 |
| [sglang / SGLang](https://docs.sglang.io/) | `sglang` | missing | 另一套大模型推理与服务框架，后续用于与 vLLM 做同口径对照；本轮依赖不安装它 |
| [modelscope / ModelScope](https://github.com/modelscope/modelscope) | `modelscope` | missing | 魔搭模型仓库客户端和工具；download_models.sh 依赖它提供的 modelscope download 命令 |
| [huggingface-hub / Hub client](https://huggingface.co/docs/huggingface_hub/) | `huggingface_hub` | 0.36.2 | 访问模型仓库、下载/缓存文件、指定 revision；pip 包名中的连字符与导入名中的下划线不同 |
| [safetensors / SafeTensors](https://github.com/safetensors/safetensors) | `safetensors` | 0.8.0 | 读取和保存张量权重文件，例如模型仓库中的 .safetensors；它是文件格式与读写库，不是量化方式 |
| [accelerate / Accelerate](https://huggingface.co/docs/accelerate/) | `accelerate` | missing | 设备分配、模型分发与 offload 等辅助能力；本阶段用于 Transformers 的 device_map="auto"，不代替 vLLM 的服务调度 |

CUDA 驱动、CUDA Toolkit/nvcc 属于系统环境，不能用一个名为“cuda”的 pip 包代替。环境采集脚本和它的 unittest 只用 Python 标准库，本身无需安装上述 GPU 依赖。

### 原 cutedsl-venv 的依赖入口与版本

| 文件 | 用途 |
| --- | --- |
| [requirements.txt](requirements.txt) | 固定当前版本、模型下载与设备分配依赖；vLLM 当前按加速入口单独安装 |

新环境另用 [requirements-vllm-0112.txt](requirements-vllm-0112.txt)，包含 vLLM 0.14.1 / torch 2.9.1+cu128 / LLM Compressor 0.9.0.5；安装方式见 [统一环境说明](#vllm-0112-统一量化与推理环境)。

原 `cutedsl-venv` 环境选择官方 **vLLM 0.11.0+cu129** wheel；requirements 保留其注释直链供参考，并精确固定你已安装的六个版本：

```text
torch==2.8.0+cu129
transformers==4.57.6
tokenizers==0.22.2
triton==3.4.0
huggingface-hub==0.36.2
safetensors==0.8.0
```

vLLM 0.11.0 要求 torch==2.8.0、Transformers>=4.55.2、tokenizers>=0.21.1，与你现有版本的声明约束匹配。依据为 [官方 CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt) 和 [通用依赖](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/common.txt)。

这里显式选择 [官方 0.11.0 CUDA 12.9 wheel](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.11.0)，与现有 torch 的 cu129 构建对应；同时固定 torchaudio 2.8.0+cu129、torchvision 0.23.0+cu129、xformers 0.0.32.post1 等配套依赖。缺失的 ModelScope 和 Accelerate 分别补为 1.31.0、1.11.0；SGLang 仍为后续选做。

目标平台为 Linux x86_64、glibc ≥2.28，建议 Python 3.11/3.12。vLLM 0.11.0 的命令和参数以 [该版本文档](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html) 为准，不能直接照搬新版 CLI。

### 安装与验证

激活你已安装上述版本的目标 Python 环境，然后从仓库根目录执行。requirements 中的精确版本约束会要求保留这六个版本；遇到依赖冲突应先查看 pip 的错误，不要删除 pins 或使用 `--no-deps` 绕过检查。

```bash
mkdir -p out/quantization_profiling/vllm-env

# 保存安装前快照。
python capstone/quantization_profiling/scripts/collect_environment.py \
  --output out/quantization_profiling/vllm-env/environment.before-install.json

python -m pip install -r capstone/quantization_profiling/requirements.txt
# 若尚未安装 vLLM，先执行下方“vLLM 加速安装”章节的命令，再继续验证。
python -m pip check

# 核验 Python 模块和 CUDA，不下载或加载模型。
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; import torch, vllm, accelerate, modelscope; print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda); print('vLLM:', vllm.__version__); print('CUDA available:', torch.cuda.is_available())"
modelscope download --help

python -m pip freeze > out/quantization_profiling/vllm-env/requirements.lock.txt
python capstone/quantization_profiling/scripts/collect_environment.py \
  --strict --expected-gpu 4090D --min-free-gib 20 \
  --output out/quantization_profiling/vllm-env/environment.json
```

如需从零重建环境，可先运行下面两行，再执行上面的安装与验证流程；同一份 requirements 会安装相同的六个基础版本，vLLM 再按加速入口单独安装。

```bash
python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate
```

requirements 固定直接依赖，传递依赖的实际解析结果仍需通过 pip freeze 和环境 JSON 保存。安装通过后检查六个版本是否与安装前一致，再执行真实模型 smoke test。当前已核验官方依赖声明与 wheel 资产，尚未在你的 4090D 上安装或运行模型。

### 原 vLLM 0.11.0 环境的加速安装

当官方 GitHub wheel 下载较慢时，可通过第三方 GitHub 下载代理获取同一版本的文件。先激活目标 Python 环境，并按上方流程保存安装前快照，再执行以下 **Bash** 命令。vLLM wheel 使用代理地址，普通 PyPI 依赖使用清华源，CUDA 12.9 的 PyTorch 配套包仍使用官方 cu129 源；六个已安装版本继续精确固定。

```bash
VLLM_WHEEL="https://ghfast.top/https://github.com/vllm-project/vllm/releases/download/v0.11.0/vllm-0.11.0%2Bcu129-cp38-abi3-manylinux1_x86_64.whl"
VLLM_SHA256="8e223c3a00eccfbc5de26fe55c7b05b9affbee1ecc2a63ca3c13bb17c1e6bbf1"

python -m pip install \
  "${VLLM_WHEEL}#sha256=${VLLM_SHA256}" \
  "torch==2.8.0+cu129" "transformers==4.57.6" \
  "tokenizers==0.22.2" "triton==3.4.0" \
  "huggingface-hub==0.36.2" "safetensors==0.8.0" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu129

python -m pip check
python -c "import torch, vllm, vllm._C; print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda); print('vLLM:', vllm.__version__)"
```

`VLLM_SHA256` 取自 [vLLM 0.11.0 官方发布资产](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.11.0)，pip 会校验下载的 wheel。若 `ghfast.top` 较慢，可将 `VLLM_WHEEL` 开头的 `https://ghfast.top/` 替换为 `https://gh-proxy.com/`，保留后面的官方文件路径和 SHA256。代理速度以目标机实际网络为准；包和扩展导入通过后，仍需执行真实模型生成验证。

该命令用于单独安装 vLLM 及其依赖；ModelScope、Accelerate 等完整项目依赖仍以 [requirements.txt](requirements.txt) 为准。requirements 中的官方 wheel 直链目前保持注释；若启用该行，`pip install -r` 会使用官方 GitHub 地址。

### vllm-0112 统一量化与推理环境

按用户选择，在同一环境中进行 BF16→FP8 量化和 vLLM 推理。沿用 `/workspace/venvs/vllm-0112` 目录及 requirements 文件名，执行下述安装后 **实际 vLLM 版本为 0.14.1**；目录名不再代表安装版本。原 `/opt/cutedsl-venv` 保留用于历史实验复现。

固定版本的依据是两个工具的官方依赖交集，而不是强行忽略 pip 检查：

| 包 | 原 vllm-0112 方案 | 统一环境 |
| --- | --- | --- |
| vllm | 0.11.2 | 0.14.1 |
| llmcompressor | 未安装 | 0.9.0.5 |
| torch | 2.9.0+cu128 | 2.9.1+cu128 |
| compressed-tensors | 0.12.2 | 0.13.0 |
| transformers | 4.57.6 | 4.57.3 |
| tokenizers / huggingface-hub / safetensors | 0.22.2 / 0.36.2 / 0.8.0 | 保留 |
| torchaudio / torchvision / xformers | 旧 torch 配套版本 | 2.9.1+cu128 / 0.24.1+cu128 / 0.0.33.post2 |
| triton / accelerate / datasets / numpy | 按旧环境解析 | 3.5.1 / 1.11.0 / 4.4.1 / 2.2.6 |

LLM Compressor 0.9.0.5 允许 torch 2.9.1 和 Transformers 4.57.3，并要求 compressed-tensors 0.13.0；vLLM 0.14.1 与其有交集。原 0.11.2 + 4.57.6 + compressed-tensors 0.12.2 无法直接补装该量化工具。新版仍使用 FP8_BLOCK 的 E4M3、128×128 权重分块和动态 128 通道激活；版本变化不保证产物逐位相同，源/产物 hash 与质量回归应重新记录。

依赖入口为 [requirements-vllm-0112.txt](requirements-vllm-0112.txt)。普通依赖使用清华源，PyTorch CUDA wheel 使用文件内的官方 cu128 附加源。目标为 Linux x86_64、Python 3.10–3.12、glibc ≥2.31；官方 vLLM 默认 wheel 的 CUDA 构建为 12.9，torch 固定为官方 2.9.1+cu128，驱动需支持 vLLM 的 CUDA 构建。不要用 CPU torch 或另一个 CUDA 变体替换固定组合。

在目标机仓库根目录执行；已有环境直接激活，没有该目录时先创建：

```bash
# 仅首次创建时执行
python3 -m venv /workspace/venvs/vllm-0112
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

# 保存升级前依赖；升级命令会调整上表中的版本
mkdir -p out/quantization_profiling/unified-env
python -m pip freeze > out/quantization_profiling/unified-env/requirements.before.txt

python -m pip install --upgrade pip \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade \
  -r capstone/quantization_profiling/requirements-vllm-0112.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip check
python -c "import torch, vllm, vllm._C, llmcompressor; print('torch:', torch.__version__); print('vllm:', vllm.__version__); print('llmcompressor:', llmcompressor.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python -c "from llmcompressor import oneshot; from llmcompressor.modifiers.quantization import QuantizationModifier; from compressed_tensors.quantization import preset_name_to_scheme; print(preset_name_to_scheme('FP8_BLOCK', ['Linear']).model_dump(mode='json'))"
```

本 requirements 包含量化工具，无需另建 GPU 量化环境。不要混装 requirements.txt（原推理环境）或 requirements-quantization.txt（原 CPU 量化配方），也不要使用 `--no-deps`。原环境留下的 xformers/torchvision/torchaudio 等包通过新文件中的配套 pins 一并更新。

在同一环境中先量化、后推理，两个模型进程分别运行。输出目录必须尚不存在；先停止占显存的模型服务，约 20GiB 空闲显存仍需实际确认转换峰值：

```bash
python capstone/quantization_profiling/scripts/quantize_fp8.py --dry-run
python capstone/quantization_profiling/scripts/quantize_fp8.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --output /workspace/weights_pub/Qwen3-8B-FP8-local-gpu \
  --device cuda:0

python capstone/quantization_profiling/scripts/run_model.py \
  --model /workspace/weights_pub/Qwen3-8B-FP8-local-gpu \
  --thinking off --prompt "请简要解释 KV cache。" --max-new-tokens 32

python -m pip freeze > out/quantization_profiling/unified-env/requirements.lock.txt
python capstone/quantization_profiling/scripts/collect_environment.py \
  --output out/quantization_profiling/unified-env/environment.json
```

已有合格 FP8 产物可直接加载；无需仅因环境升级重做转换。`--device cpu` 也可在本环境中使用，但 CUDA 版 torch 不会因此变成 CPU wheel。默认 `fp8-local` 别名和 configs/fp8.yaml 仍指向 Qwen3-8B-FP8-local；测试上面的新 GPU 输出目录时显式指定模型路径，正式 benchmark 使用修改 model 路径后的单模型配置。

切换环境：

```bash
# 当前统一量化/推理环境
source /workspace/venvs/vllm-0112/bin/activate

# 历史 cutedsl 环境；按需要分别执行 source 命令
source /opt/cutedsl-venv/bin/activate
echo "$VIRTUAL_ENV"
which python
```

本地已用临时解析器验证 Python 3.10 / Linux x86_64 的完整依赖解析，共194个包；未在本机安装GPU包或执行目标机量化。benchmark 已增加这套完整统一版本组合的预检入口，并把实际组合写入 run.json；旧版组合保留历史复现。预检、依赖解析和模拟测试不等于目标 GPU 实测，先完成短请求验证，再在同一新环境重新建立 BF16/AWQ/FP8 对照，不能把跨版本差异全部归因于量化。

依据：[vLLM 0.14.1 CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.14.1/requirements/cuda.txt)、[通用依赖](https://github.com/vllm-project/vllm/blob/v0.14.1/requirements/common.txt)、[LLM Compressor 0.9.0.5 发行元数据](https://pypi.org/pypi/llmcompressor/0.9.0.5/json)、[PyTorch 2.9.1 安装说明](https://pytorch.org/get-started/previous-versions/#v291)、[compressed-tensors 0.13.0 FP8_BLOCK 定义](https://github.com/vllm-project/compressed-tensors/blob/0.13.0/src/compressed_tensors/quantization/quant_scheme.py)。

### ModelScope 启动报错：缺少 pkg_resources

当目标环境使用 `modelscope==1.31.0` 和 `setuptools==84.0.0` 时，执行 `modelscope download` 会在导入插件管理模块时出现 `ModuleNotFoundError: No module named 'pkg_resources'`。ModelScope 1.31.0 仍依赖该模块，而 [setuptools 从 82.0.0 起已移除它](https://setuptools.pypa.io/en/latest/history.html#v82-0-0)。本项目在 requirements 中固定 `setuptools==80.9.0`，提供该旧接口并避免后续安装再次选中不兼容版本。

在目标机已激活的 `cutedsl-venv` 中执行下面的最小修复命令，再验证 CLI：

```bash
python -m pip install "setuptools==80.9.0" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import importlib.metadata as m; import pkg_resources; print('setuptools:', m.version('setuptools')); print('pkg_resources:', pkg_resources.__file__)"
modelscope download --help
python -m pip check

# CLI 验证通过后，用相同参数重试；已完成的模型仍会跳过。
bash capstone/quantization_profiling/scripts/download_models.sh \
  --model Qwen/Qwen3-8B \
  --model Qwen/Qwen3-8B-AWQ
```

`pkg_resources` 仍可能打印弃用警告；警告与模块缺失异常不同。下载失败留下的同来源目录缺少完成标记，脚本会重新调用 ModelScope 重试/续传，无需手动删除目录。

## 模型下载与环境检测

当前目标实验机为 **RTX 4090D，约 20G 可用显存**；脚本按实际运行机器检测，默认核对 4090D 和至少 20 GiB 空闲显存。20 GiB 是可调整的预检目标，不是启动模型的显存预留；若你说的 20G 是十进制容量或近似值，请按实际需求设置 `--min-free-gib`。

以下命令从仓库根目录执行。环境检测使用当前激活环境的 Python；先激活准备运行 vLLM 的环境，再执行检测，避免查到其他 Conda/venv 的包。

```bash
# 打印环境，并额外保存完整 JSON；缺少组件时仍输出所有检查结果。
python capstone/quantization_profiling/scripts/collect_environment.py \
  --output out/quantization_profiling/environment.json

# 目标机预检；条件不满足时返回退出码 1。
python capstone/quantization_profiling/scripts/collect_environment.py \
  --expected-gpu 4090D --min-free-gib 20 --strict \
  --output out/quantization_profiling/environment.json

# 只预览：已有模型显示 SKIP，缺失模型显示下载命令，不修改文件。
bash capstone/quantization_profiling/scripts/download_models.sh --dry-run

# 使用已安装且支持 modelscope download 的 CLI，默认仅下载 Qwen/Qwen3-8B。
bash capstone/quantization_profiling/scripts/download_models.sh
```

下载脚本沿用参考脚本的 ModelScope CLI，默认模型根目录为 `/workspace/weights_pub`。本项目使用的预训练模型统一从该目录下的模型子目录读取：BF16 为 `/workspace/weights_pub/Qwen3-8B`，AWQ 为 `/workspace/weights_pub/Qwen3-8B-AWQ`，本地 FP8 为 `/workspace/weights_pub/Qwen3-8B-FP8-local`，官方 FP8 为 `/workspace/weights_pub/Qwen3-8B-FP8`。已有权重直接复用；本次四模型运行/benchmark 使用 BF16、AWQ、官方 FP8、本地 FP8 配置；当前运行入口见后文。支持 `MODEL_ROOT` 环境变量、`--model-root`、重复 `--model`（替换默认列表）、`--revision`、`--dry-run` 与 `--help`，不会自动安装依赖。

**一次下载多个模型**：重复传入 `--model`，按顺序处理；只列出需要的模型。下面以 BF16 和 AWQ 为例，已有 `Qwen3-8B` 非空目录时显示 `SKIP`，继续下载缺失的 `Qwen3-8B-AWQ`。

```bash
# 预览批量任务，不要求安装 ModelScope。
bash capstone/quantization_profiling/scripts/download_models.sh \
  --model Qwen/Qwen3-8B \
  --model Qwen/Qwen3-8B-AWQ \
  --dry-run

# 实际执行；默认保存到 /workspace/weights_pub。
bash capstone/quantization_profiling/scripts/download_models.sh \
  --model Qwen/Qwen3-8B \
  --model Qwen/Qwen3-8B-AWQ

# 四项对比所需的三份仓库权重；本地 FP8 由量化脚本另行生成。
bash capstone/quantization_profiling/scripts/download_models.sh \
  --model Qwen/Qwen3-8B \
  --model Qwen/Qwen3-8B-AWQ \
  --model Qwen/Qwen3-8B-FP8
```

```bash
# 替换为 ModelScope 上可用的 revision；固定版本时不同 revision 使用不同目录。
bash capstone/quantization_profiling/scripts/download_models.sh \
  --model Qwen/Qwen3-8B \
  --revision "<ModelScope revision>" \
  --model-root /workspace/weights_pub/revisions/example
```

ModelScope 与 Hugging Face 的 revision 不必相同，不能直接混用两边的 commit。未指定 revision 时跟随源仓库默认版本，不能据此宣称完成不可变快照锁定。参数依据 [ModelScope CLI 官方源码](https://github.com/modelscope/modelscope/blob/v1.31.0/modelscope/cli/download.py)；使用前可通过 `modelscope download --help` 核对已安装版本。

目录处理与重试规则：

- 已有非空目录没有脚本标记时，视为已有模型，打印 `SKIP` 并继续后续模型；不添加标记或修改原文件，也不校验权重是否完整。所有模型都跳过时，无需安装 ModelScope。
- 新目录或空目录正常下载。`.modelscope-download-source` 保存请求的模型 ID 和 revision；仅 ModelScope 成功返回后写入 `.modelscope-download-complete`。两份标记均匹配的模型下次直接跳过。
- 来源标记匹配但缺少有效完成标记时，重新调用 ModelScope 重试/续传；旧版脚本留下的同来源目录也走这条路径。来源标记损坏、模型 ID 或 revision 不匹配的已有目录则跳过并提示另选 `--model-root`，不会覆盖已有内容。
- 某个模型下载失败后仍继续处理后面的模型，最终输出下载、跳过、失败数量，并返回首个下载失败的非零退出码。重新运行同一命令即可重试未完成项，已完成项会跳过。
- `--dry-run` 只展示跳过项与待下载/续传命令，不创建目录或写入标记。完成标记表示 CLI 成功返回，不代替权重校验，也不记录默认分支解析后的 commit；需要指定其他 revision 时使用不同根目录。

无需 GPU、网络或真实权重即可运行脚本行为测试（需要 Bash；Windows 可使用 Git Bash）：

```bash
python -m unittest discover -s capstone/quantization_profiling/tests -p "test_download_models.py" -v
```

环境脚本使用标准库，通过线程池并行执行包元数据、GPU、PyTorch/CUDA、nvcc 和 Nsight 探针：

- 打印当前 Python 路径，以及 torch、vLLM、Transformers、tokenizers、Triton、SGLang、ModelScope 等版本；包元数据不等于模块导入或服务验证通过。
- 读取物理 GPU 名称、UUID、驱动与总/已用/空闲显存；通过独立子进程检查 PyTorch 导入、CUDA 可用性、可见设备、计算能力、BF16 支持及显存。
- 分别标注 nvidia-smi 的驱动支持 CUDA 上限、nvcc 的本机 Toolkit 版本、PyTorch 构建所用 CUDA 版本；三者无需完全相同。
- 尊重当前 Python 的 CUDA 设备可见性；被隐藏的物理 GPU 不会被当作可运行目标。PyTorch 初始化前后采样时刻可能不同，显存以各自采样值为准。
- 缺包、命令不可用和探针超时均保留诊断；`--timeout` 控制每条外部命令超时，`--workers` 控制并行度，nvcc/Nsight 缺失不会单独阻止预检通过。
- 默认只打印；`--output` 才写 JSON。普通盘点模式成功收集后返回 0；`--strict` 额外要求目标 GPU/显存、Linux、PyTorch CUDA 与 torch/vLLM 包存在。预检通过仍需后续真实模型服务 smoke test。

本地验证环境为 Windows/RTX 5060，已验证只读采集与缺包诊断；目标 4090D 环境尚待运行以上命令。当前脚本不加载模型，也未执行真实权重下载。

## 运行已下载的模型

通用入口为 [scripts/run_model.py](scripts/run_model.py)，支持 Transformers、vLLM 本地生成，以及 vLLM HTTP 服务。按被测路线激活对应 GPU 推理环境；原 0.11.0 环境用于历史复现，本地 FP8 按上面的统一环境流程验证。

`--model` 支持以下别名，也继续接受本地模型目录；`MODEL_ROOT` 可以修改别名对应的根目录。

| 模型别名 | 默认本地目录 | vLLM 运行方案 |
| --- | --- | --- |
| `qwen3-8b` | `/workspace/weights_pub/Qwen3-8B` | BF16，无权重量化 |
| `qwen3-8b-awq` | `/workspace/weights_pub/Qwen3-8B-AWQ` | W4A16，`awq_marlin`，BF16 激活 |
| `qwen3-8b-fp8` | `/workspace/weights_pub/Qwen3-8B-FP8` | 官方分块 FP8，原生 `fp8` 加载器 |
| `qwen3-8b-fp8-local` | `/workspace/weights_pub/Qwen3-8B-FP8-local` | FP8 权重/动态 FP8 激活，按 config 自动识别 compressed-tensors |

每次只启动一个模型。下面的 AWQ 服务示例可将别名替换为其它三种模型；已有服务占用 7081 时先正常停止它：

```bash
python capstone/quantization_profiling/scripts/run_model.py \
  --serve --model qwen3-8b-awq \
  --host 0.0.0.0 --port 7081 --max-num-seqs 1
```

AWQ 预设使用支持 BF16 的 Marlin 路径；若手工指定旧 `--quantization awq`，需要按该后端要求使用 `--dtype float16`，并记录 dtype 与后端变化。本地 FP8 产物保留 `quant_method=compressed-tensors`；不要为使用 FP8 而手动把它改成 `fp8`。Transformers 入口仍仅用于未量化模型。

```bash
# 先检查模板、thinking 开关和输入 token，不加载权重。
python capstone/quantization_profiling/scripts/run_model.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --thinking off --tokenize-only --prompt "请解释 KV cache。"

# 使用 vLLM 生成一条短回答；默认上下文 2048、调度序列上限 1、eager 执行。
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm --model /workspace/weights_pub/Qwen3-8B \
  --dtype bfloat16 --thinking off --max-new-tokens 128 \
  --prompt "请用三句话解释 prefill 和 decode。" \
  --output out/quantization_profiling/vllm-smoke.json

# 上一进程退出后，用 Transformers 跑同一问题。
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend transformers --model /workspace/weights_pub/Qwen3-8B \
  --dtype bfloat16 --thinking off --max-new-tokens 128 \
  --prompt "请用三句话解释 prefill 和 decode。" \
  --output out/quantization_profiling/transformers-smoke.json

# 启动常驻服务；prompt、thinking、生成长度由客户端请求设置。
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm --serve --model /workspace/weights_pub/Qwen3-8B \
  --dtype bfloat16 --host 0.0.0.0 --port 7081 --served-model-name local-model \
  --concurrency 4
```

从另一台机器执行远程接口测试（SERVER_IP 替换为服务器实际 IP）：

```bash
python capstone/quantization_profiling/scripts/test_server.py \
  --base-url http://SERVER_IP:7081 --model local-model \
  --concurrency 4 --requests 20 --require-server-metrics
```

服务默认监听 `0.0.0.0:7081`；客户端访问 `http://SERVER_IP:7081`。测试脚本只使用 Python 标准库，默认检查 `GET /health`、`GET /v1/models` 和对话生成。服务端 `--concurrency 4` 是 `--max-num-seqs 4` 的别名，限制 vLLM 同时调度的序列数；客户端 `--concurrency 4` 是同时在途的 HTTP 请求数，`--requests 20` 是总请求数，空闲线程会继续领取后续请求。未传 `--requests` 时等于客户端并发数。客户端可以超过服务端调度上限，此时部分请求排队，延迟可能增加；4090D 初次运行也可先将两侧都设为 1，再逐步增加并观察显存。

每次测试自动写入 **客户端所在项目的 `capstone/quantization_profiling/log/`**，与执行命令时的当前目录无关，文件名包含 UTC 时间与随机 ID：

- `test_server_*.log`：检查结果、回答、逐请求指标与汇总信息。
- `test_server_*.jsonl`：逐请求成功或失败记录、usage、服务端/客户端指标和分组汇总，运行中逐条刷新。
- `test_server_*.json`：完整测试报告，包括参数、检查、错误与汇总。`--output` 可额外导出一份，`--log-dir` 可更改默认目录。

**指标口径**：本地推理记录加载耗时、生成耗时和实际输出 token 吞吐量，加载时间单独统计；HTTP 服务在响应的 `profiling` 字段返回 `inference_seconds`、`input_tokens`、`output_tokens`、`output_tokens_per_second` 等指标。服务端耗时包含排队、模板/tokenizer 处理和生成，流式模式还可能受发送背压影响，不代表纯 GPU kernel 时间。客户端另行记录包含网络的端到端耗时。流式指标在 `[DONE]` 之前通过额外的 `choices=[]` SSE 事件返回，真实 token 数来自 `usage`，不使用文本片段数代替。

汇总吞吐量为 **成功请求的实际输出 token 总数 ÷ 整组测试的墙钟时间**，不会把单请求的 tokens/s 相加；同时记录成功 req/s、失败率、成功请求延迟 p50/p95。缺少 usage 时吞吐量为 `null`。默认不自动预热，首次请求的初始化开销可能进入测量，这些探索性测试不能直接视为正式 BF16 基线。

使用更新后的 `run_model.py --serve` 重启服务后，`--require-server-metrics` 会核验服务端指标是否存在。`--stream-only` 只测试流式请求；`--stream` 先测试非流式，再测试流式，每一组各执行 `--requests` 次；`--check-only` 只检查健康和模型列表。部分请求失败仍会继续完成其余请求并保存日志，最终返回非零退出码。更多说明见 [教程第 5 节](docs/model_running.md#5-启动-http-服务并用-python-调用)。

默认 vLLM 显存预算为物理总显存的 0.75，单步 token 预算为 2048，并启用 chunked prefill。`run_model.py` 和 `test_server.py` 用于功能/连通性测试；固定负载、预热、轮次和显存采样使用下面的正式 benchmark。新增加的 AWQ/官方 FP8/本地 FP8 路线仍需在目标 4090D 上实际运行验证。

完整说明见 [模型运行与 vLLM / Transformers 详细教程](docs/model_running.md)：包括 AWQ、批量和多轮输入、参数对照、两个库的独立 Python 示例、HTTP 客户端、流式输出与显存排错。

## 四模型正式 benchmark

`--all-models` 依次运行以下四项，使用相同的 BF16 tokenizer、冻结请求和性能协议，结果按各自标签保存：

| benchmark 选择 | 模型目录 | 配置文件 |
| --- | --- | --- |
| `bf16` | `/workspace/weights_pub/Qwen3-8B` | `configs/bf16.yaml` |
| `awq` | `/workspace/weights_pub/Qwen3-8B-AWQ` | `configs/awq.yaml` |
| `fp8-official` | `/workspace/weights_pub/Qwen3-8B-FP8` | `configs/fp8-official.yaml` |
| `fp8-local` | `/workspace/weights_pub/Qwen3-8B-FP8-local` | `configs/fp8.yaml` |

官方 FP8 使用 [Qwen 官方配置](https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/config.json) 声明的原生 `fp8`、E4M3、动态激活和 `[128, 128]` 权重分块；本地产物保留 `compressed-tensors` 加载方式。两个 FP8 方案独立列示。`fp8` 短别名为兼容旧命令仍表示 `fp8-local`；选择官方权重请使用 `fp8-official` 或 `qwen3-8b-fp8`。

若尚未下载官方 FP8，可在已安装 ModelScope CLI 的环境中执行：

```bash
bash capstone/quantization_profiling/scripts/download_models.sh --model Qwen/Qwen3-8B-FP8
```

正式执行接受原 cutedsl 完整版本组合或上面的统一环境组合，不允许混用两者；实际版本组合写入 run.json。本地 FP8 先按 [统一环境说明](#vllm-0112-统一量化与推理环境) 完成加载验证，再运行同环境四模型对照。以下参数可用，目标 GPU 结果仍需实测。

先准备 BF16 tokenizer 的固定请求快照；已有有效的 `data/frozen` 可以直接复用，不能覆盖。生成方式、文件含义与自定义目录见 [benchmark 测试数据生成说明](docs/week1_execution.md#2-冻结数据与复用旧快照)。在 Linux 目标机执行：

```bash
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

# 首次生成：默认每组50条正式请求；输出目录必须尚不存在。
python capstone/quantization_profiling/scripts/prepare_workloads.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --config capstone/quantization_profiling/configs/workloads.yaml \
  --output capstone/quantization_profiling/data/frozen
```

默认生成 **275 条性能请求：五组，每组5条预热、50条正式请求**，完整套用聊天模板后校验最终输入 token 数；四模型共用这份数据。生成过程不启动 GPU 推理，但会读取 BF16 权重计算 SHA256；另将已有 100 条质量任务和判分规则复制到快照。

**生成数量可设置。** 生成脚本新增 `--requests N`：每组正式请求数，不含预热；覆盖配置值，省略时按配置生成。以下示例为每组200条，五组共1025条性能记录（1000条正式、25条预热）；输出到新目录以保留原快照：

```bash
python capstone/quantization_profiling/scripts/prepare_workloads.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --config capstone/quantization_profiling/configs/workloads.yaml \
  --requests 200 \
  --output capstone/quantization_profiling/data/frozen_200

# 使用这份快照，实际运行每组200条、1轮；可先追加 --plan-only 预览。
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot capstone/quantization_profiling/data/frozen_200 \
  --all-models --requests 200 --rounds 1
```

生成脚本的 `--requests` 决定快照容量，benchmark 的 `--requests` 决定每组每轮实际测量数量。benchmark 默认仍为50，生成200条后需显式传 `--requests 200` 才会全部使用；四模型1轮合计4000条正式、100条预热请求。若五组合计只需要200条正式请求（每个模型），两个命令均改用 `--requests 40`。更多数量和旧快照复用说明见 [自定义数据数量](docs/week1_execution.md#25-已有快照重建和自定义数量)。

以下命令继续使用默认 `data/frozen` 快照：

```bash
# 校验快照并预览四模型计划，不启动 GPU 服务。
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot capstone/quantization_profiling/data/frozen --all-models --plan-only

# 默认 BF16：五组负载，每组预热5次、正式50次，共1轮。
python capstone/quantization_profiling/scripts/benchmark.py

# 选择 AWQ；轮数可选，下面每组做2轮，每轮50次正式请求。
python capstone/quantization_profiling/scripts/benchmark.py --model awq --rounds 2

# 单独测试官方 FP8。
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-official

# 选择本地 FP8，先只测短请求一组。
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-local --case w128_o128

# 先预览四模型计划，再依次运行；脚本自动启动/停止受管服务。
python capstone/quantization_profiling/scripts/benchmark.py --all-models --plan-only
python capstone/quantization_profiling/scripts/benchmark.py --all-models --rounds 1 --requests 50

# 也可以选择两种模型；--model 可重复，支持上面的 qwen3 别名。
python capstone/quantization_profiling/scripts/benchmark.py --model bf16 --model awq
```

默认每种模型共 **250 个正式请求 + 25 个预热请求**；四模型合计 **1000 + 100**。`--rounds N` 设置重复轮数，`--requests N` 设置每组每轮的正式请求数，均须为正整数，正式请求数不能超过快照中可用记录数。`--config <文件>` 可指定单模型自定义配置，与 `--model`/`--all-models` 互斥。

旧的 100 请求/3 轮快照无需重建：本次默认确定性地选择每组前 50 个正式请求，所有模型和轮次使用相同子集；原协议、本次有效协议和请求子集 hash 会另行记录，实际选中的记录保存为 selected_requests.jsonl。要重跑旧的 3×100 协议，使用包含 100 条正式请求的旧快照并显式设置 `--rounds 3 --requests 100`。不传生成参数时新快照默认每组50个正式请求，可用 prepare_workloads.py --requests N 扩大容量。

日志位于 `log/benchmark_<ID>/<模型>/<负载>/`，保留每种模型的原始请求、启动日志、实际引擎 dtype/量化方法和显存；根目录记录 run.json 和 summary.json。单轮结果可用于初步对照，但不能判断跨轮波动；50 个样本的 p95 也比原 100 样本更不稳定。历史 3×100 BF16 报告保留原实验 ID、协议和结果。

### benchmark 结束后生成报告

benchmark 结束时打印 `Artifacts: .../log/benchmark_<ID>`；随后手工运行 `summarize.py`，传入这个**实验根目录**，其中应有 `run.json` 和根 `summary.json`。下面以历史实验 ID 为例，请替换成实际目录；命令从仓库根目录执行：

```bash
python capstone/quantization_profiling/scripts/summarize.py \
  "capstone/quantization_profiling/log/benchmark_20261020T084357Z_294018cc"
```

默认在该实验目录生成 `Week1_Report.md`（按模型、负载、正式轮次列示结果与协议验收）和 `performance.jsonl`（合并请求明细，包含预热/正式记录）。它不重新压测，也不会启动模型。需要把 Markdown 报告放到 reports 时：

```bash
python capstone/quantization_profiling/scripts/summarize.py \
  "capstone/quantization_profiling/log/benchmark_20261020T084357Z_294018cc" \
  --output capstone/quantization_profiling/reports/Benchmark_Report.md
```

`--output` 只指定 Markdown 文件位置，`performance.jsonl` 仍写入原实验目录。汇总只需 Python 标准库；将完整实验目录复制到 Windows 后也能运行。参数、Windows 命令和结果解读见 [benchmark 报告汇总说明](docs/week1_execution.md#42-汇总已有-benchmark-结果)。

## 核心实验约定

- 模型固定为 Qwen3-8B；优先使用 4090D 单卡，其他设备独立记录。模型、tokenizer、chat template、thinking 开关、生成参数和 runtime 版本均需锁定。
- 固定输入/目标输出 token：`128/128`、`1024/128`、`4096/256`、`8192/128`、`1024/512`。从并发 `1` 开始，容量允许再测 `8/16`，并发 `32` 选做。
- 同一对照组使用相同预热、请求子集与所选轮数；默认每轮预热 5 次、正式 50 次、1 轮，可用 `--rounds` 增加重复。记录实际 token、TTFT、TPOT、端到端时延、吞吐、显存和错误；正式计时与 profiler 采集分开。
- AWQ 为首选 W4A16 方案；本次预设 awq_marlin + BF16；本地 FP8 使用自身量化配置。实际后端由启动日志核验，具体 GPU kernel 另用 Profiler 确认。INT8 仍为选做。
- 质量集固定 100 条，独立于校准集。准确率下降不超过 **2 个百分点**作为待锁定的项目目标，不能当作已取得的结果或统计保证。
- BF16 与量化使用共同可运行配置做配对比较；量化独有的长上下文/大并发容量收益单独报告。保留 OOM、失败与局限。

## 完成后的交付标准

能够从环境与数据指纹出发，复跑一个 BF16/W4A16 配对 case，并追溯到原始性能记录、质量逐题结果、trace 和最终结论。报告必须解释至少一个真实瓶颈，并说明量化为何在某些负载上有效或无效；性能与质量结论均以实测为准。

官方学习资料与版本核验入口集中在 [planning.md](planning.md)；模型运行与服务调用见 [详细教程](docs/model_running.md)，固定 workload 的 BF16/AWQ/官方 FP8/本地 FP8 正式基准测试见 [第一周执行说明](docs/week1_execution.md)。

## 自行制作 Qwen3-8B 分块 FP8

详见 [完整说明](docs/fp8_quantization.md)。默认使用 CPU 版 PyTorch 做离线转换；普通依赖用清华镜像，`2.8.0+cpu` wheel 使用 requirements 中的上海交大 PyTorch CPU 镜像（官方 CPU 源作为备用）。先建立独立环境：

```bash
python3 -m venv /workspace/venvs/qwen-fp8-quant
source /workspace/venvs/qwen-fp8-quant/bin/activate
python -m pip install \
  -r capstone/quantization_profiling/requirements-quantization.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
```

然后运行：

```bash
python capstone/quantization_profiling/scripts/quantize_fp8.py --dry-run
python capstone/quantization_profiling/scripts/quantize_fp8.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --output /workspace/weights_pub/Qwen3-8B-FP8-local --device cpu
python capstone/quantization_profiling/scripts/quantize_fp8.py \
  --output /workspace/weights_pub/Qwen3-8B-FP8-local --verify-only
```

权重为 E4M3、128×128 分块；激活动态按每 token 的 128 通道分组。无需任务校准集。导出 quant_method 为 compressed-tensors，不手动改为官方原生 fp8。默认 CPU 转换以避免约20GiB空闲显存的峰值限制，仍需足够主存；`--device cuda:0` 可在上面的统一环境中使用，独立 CPU 版环境不能使用。输出目录必须不存在；失败留档后换新目录重试。转换完可直接在统一环境中用 run_model.py 加载。真实8B转换、质量及目标GPU推理尚待验收。
