# P1：Qwen3-8B 量化与性能诊断

围绕同一份 Qwen3-8B 模型与固定负载，建立 BF16 基线，运行一条 W4A16 量化路径，结合 Nsight 时间线、热点 kernel 和留出集质量回归，解释显存、吞吐、尾延迟与质量之间的取舍。

**当前状态：已实现模型下载和并行环境检测，模型实验尚未开始。** 当前目标为 RTX 4090D、约 20G 可用显存；下载与检测脚本可用，目标机待验证。其余配置、业务脚本、数据与报告保持空占位，未预填实验结果。

## 从哪里开始

1. 阅读 [详细学习与实施规划](planning.md)，按“环境与显存预算 → 数据与口径 → BF16 → Profiling → W4A16 → 质量回归 → 报告”的顺序推进。
2. 对照 [W29–W52 每周计划与验收](../../docs/learning-plan-w29-w52.md)，保留 P1-01～P1-06 的任务编号和验收要求。
3. 复习 [Nsight 工具手册](../../docs/profiling/nsight-tools.md) 的时间线与热点分析方法；其中现有示例面向 kernel lab，本项目的模型服务采集流程将在后续实现。

计划来源为 `00.AI-system-plan-optimized-v3.xlsx` 的“案例1_量化与Profiling”，此处依据仓库已有周计划与案例说明展开。W29、W30、W32 是原学习周编号；实际补做时间和完成状态以实验记录为准。

## 要完成什么

| 任务 | 原周次 | 目标与交付 | 验收重点 | 当前状态 |
| --- | --- | --- | --- | --- |
| P1-01 | W29 | 环境、模型 revision、模板与生成参数锁定，启动入口 | 可重建环境，复现代表请求；读取实际显存 | 进行中：工具已实现，目标机待验证 |
| P1-02 | W29 | 五组 workload 的 BF16 原始记录和基线汇总 | 并发 1 起步；预热后至少 3 轮；OOM 如实记录 | 未开始 |
| P1-03 | W30 | nsys 时间线、3 个代表热点的 ncu 抽样与证据表 | 分清 prefill/decode；至少一个热点有证据和优化假设 | 未开始 |
| P1-04 | W30 | AWQ 优先的 W4A16 配置、权重来源与对照结果 | 核验版本/硬件/kernel；相同输入与评测协议 | 未开始 |
| P1-05 | W30/W32 | 固定 100 条留出任务的逐题质量回归 | 与校准集分离；固定判分；实验前锁定质量门槛 | 未开始 |
| P1-06 | W32 | 性能、显存、质量与诊断报告及复现入口 | 每项结论可追溯到实验 ID；不预设加速收益 | 未开始 |

## 目录与文件职责

```text
capstone/quantization_profiling/
├── README.md                       # 项目入口与验收概览
├── planning.md                     # 学习路径、实施顺序与评测约定
├── configs/
│   ├── bf16.yaml                   # 待实现：BF16 服务与生成配置
│   ├── awq.yaml                    # 待实现：AWQ/W4A16 配置与权重信息
│   └── workloads.yaml              # 待实现：负载、并发、重复次数与种子
├── scripts/
│   ├── collect_environment.py      # 已实现：并行环境检测、打印与 JSON 快照
│   ├── download_models.sh          # 已实现：ModelScope 模型下载与 revision 选择
│   ├── prepare_workloads.py        # 待实现：请求准备、token 校验与数据指纹
│   ├── serve.sh                    # 待实现：按配置启动服务
│   ├── benchmark.py                # 待实现：流式压测与逐请求记录
│   ├── evaluate_quality.py         # 待实现：留出集推理、判分与逐题结果
│   └── summarize.py                # 待实现：结果校验、汇总与报告素材
├── tests/
│   └── test_collect_environment.py # 环境探针、并发与预检行为测试
├── data/
│   ├── performance.jsonl           # 空占位：未来逐请求性能记录
│   └── quality_holdout.jsonl       # 空占位：未来冻结的 100 条质量任务
└── reports/
    └── Project1_Report.md          # 空占位：未来实验报告
```

文件的输入、输出、数据字段和后续生成产物约定见 [planning.md](planning.md)。模型权重、大型 trace 和运行日志的存放规则也在那里说明。

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

# 只预览下载命令：不需要安装 ModelScope，也不创建模型目录。
bash capstone/quantization_profiling/scripts/download_models.sh --dry-run

# 使用已安装且支持 modelscope download 的 CLI，默认仅下载 Qwen/Qwen3-8B。
bash capstone/quantization_profiling/scripts/download_models.sh --model-root /data/models
```

下载脚本沿用参考脚本的 ModelScope CLI，默认目录为仓库的 `models/Qwen3-8B/`，该目录已加入 Git 忽略。支持 `MODEL_ROOT` 环境变量、`--model-root`、重复 `--model`（替换默认列表）、`--revision`、`--dry-run` 与 `--help`。例如需要 AWQ 权重时显式指定 `--model Qwen/Qwen3-8B-AWQ`；不会自动下载所有精度的权重，也不会自动安装依赖。

```bash
# 替换为 ModelScope 上可用的 revision；固定版本时不同 revision 使用不同目录。
bash capstone/quantization_profiling/scripts/download_models.sh \
  --model Qwen/Qwen3-8B \
  --revision "<ModelScope revision>" \
  --model-root /data/models-fixed
```

ModelScope 与 Hugging Face 的 revision 不必相同，不能直接混用两边的 commit。未指定 revision 时跟随源仓库默认版本，不能据此宣称完成不可变快照锁定。参数依据 [ModelScope CLI 官方源码](https://github.com/modelscope/modelscope/blob/v1.31.0/modelscope/cli/download.py)；使用前可通过 `modelscope download --help` 核对已安装版本。

下载目标内的 `.modelscope-download-source` 保存请求的模型 ID 和 revision，用于避免不同来源混写。已有非空目录没有该标记，或同名目录的模型/revision 不匹配时，脚本会退出并提示另选目录。下载失败保留非零退出码，可用相同参数重试；标记不表示下载已完整，也不记录默认分支解析后的 commit。

环境脚本使用标准库，通过线程池并行执行包元数据、GPU、PyTorch/CUDA、nvcc 和 Nsight 探针：

- 打印当前 Python 路径，以及 torch、vLLM、Transformers、tokenizers、Triton、SGLang、ModelScope 等版本；包元数据不等于模块导入或服务验证通过。
- 读取物理 GPU 名称、UUID、驱动与总/已用/空闲显存；通过独立子进程检查 PyTorch 导入、CUDA 可用性、可见设备、计算能力、BF16 支持及显存。
- 分别标注 nvidia-smi 的驱动支持 CUDA 上限、nvcc 的本机 Toolkit 版本、PyTorch 构建所用 CUDA 版本；三者无需完全相同。
- 尊重当前 Python 的 CUDA 设备可见性；被隐藏的物理 GPU 不会被当作可运行目标。PyTorch 初始化前后采样时刻可能不同，显存以各自采样值为准。
- 缺包、命令不可用和探针超时均保留诊断；`--timeout` 控制每条外部命令超时，`--workers` 控制并行度，nvcc/Nsight 缺失不会单独阻止预检通过。
- 默认只打印；`--output` 才写 JSON。普通盘点模式成功收集后返回 0；`--strict` 额外要求目标 GPU/显存、Linux、PyTorch CUDA 与 torch/vLLM 包存在。预检通过仍需后续真实模型服务 smoke test。

本地验证环境为 Windows/RTX 5060，已验证只读采集与缺包诊断；目标 4090D 环境尚待运行以上命令。当前脚本不加载模型，也未执行真实权重下载。

## 核心实验约定

- 模型固定为 Qwen3-8B；优先使用 4090D 单卡，其他设备独立记录。模型、tokenizer、chat template、thinking 开关、生成参数和 runtime 版本均需锁定。
- 固定输入/目标输出 token：`128/128`、`1024/128`、`4096/256`、`8192/128`、`1024/512`。从并发 `1` 开始，容量允许再测 `8/16`，并发 `32` 选做。
- 同一对照组预热后至少测 3 轮；记录实际输入/输出 token、TTFT、TPOT、端到端时延、吞吐、显存和错误。正式计时与 profiler 采集分开。
- AWQ 为目录对应的首选 W4A16 方案；具体工具、格式和 kernel 在锁版本时核验。AWQ 不可用时可独立记录 GPTQ 替代路径。INT8/FP8 选做，不替代 W4A16。
- 质量集固定 100 条，独立于校准集。准确率下降不超过 **2 个百分点**作为待锁定的项目目标，不能当作已取得的结果或统计保证。
- BF16 与量化使用共同可运行配置做配对比较；量化独有的长上下文/大并发容量收益单独报告。保留 OOM、失败与局限。

## 完成后的交付标准

能够从环境与数据指纹出发，复跑一个 BF16/W4A16 配对 case，并追溯到原始性能记录、质量逐题结果、trace 和最终结论。报告必须解释至少一个真实瓶颈，并说明量化为何在某些负载上有效或无效；性能与质量结论均以实测为准。

官方学习资料与版本核验入口集中在 [planning.md](planning.md)；模型服务与基准测试的实测命令待后续实现时补充。
