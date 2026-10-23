# 案例一：四模型质量对比

在 Linux 4090D 主机的仓库根目录执行。自动比较 BF16、AWQ/W4A16、官方 FP8、本地 FP8；每种模型完成相同的 100 条质量任务，共 400 次请求。

脚本负责逐个启动服务、等待就绪、核验引擎配置、评测、停止该服务和生成报告。一次只加载一种模型。运行前停止其它占用 7081 端口的服务，并结束性能压测和 Profiler 采集。

## 1. 运行命令

使用已经安装好的统一环境，无需重新安装依赖：

```bash
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

# 预览：校验冻结数据与判分器 hash，不加载模型或启动服务。
python capstone/quantization_profiling/scripts/compare_quality.py --plan-only

# 正式评测：默认四种模型，自动生成独立日志目录和质量报告。
python capstone/quantization_profiling/scripts/compare_quality.py
```

默认读取 `data/frozen`；不需要重新生成质量数据。若本次性能实验使用的是另一份冻结快照，可指定同一目录：

```bash
python capstone/quantization_profiling/scripts/compare_quality.py \
  --snapshot capstone/quantization_profiling/data/frozen_200
```

质量任务始终是 100 条，不受性能快照的 `--requests 50/80/200` 或轮次影响。质量脚本不接受性能测试的 `--requests`、`--rounds` 参数。

只比较 BF16 和 AWQ：

```bash
python capstone/quantization_profiling/scripts/compare_quality.py \
  --model bf16 --model awq
```

模型路径沿用 benchmark 的配置和别名：

| 选择 | 默认模型目录 | 请求的量化方式 |
| --- | --- | --- |
| bf16 | /workspace/weights_pub/Qwen3-8B | 未量化，BF16 |
| awq | /workspace/weights_pub/Qwen3-8B-AWQ | awq_marlin，BF16 激活 |
| fp8-official | /workspace/weights_pub/Qwen3-8B-FP8 | 原生 fp8 |
| fp8-local | /workspace/weights_pub/Qwen3-8B-FP8-local | 按 compressed-tensors 配置识别 |

四种模型统一使用冻结 BF16 tokenizer、模板和 thinking 开关。模型文件另存 SHA256，核对配置与服务日志中的 dtype、量化方法；这些核验不能替代实际 CUDA kernel 的 trace。

默认 `--enforce-eager false`、`--fp8-backend auto`、`--max-model-len 2048`，与最近允许 CUDA Graph 的性能对照保持同一设置方向。需要评估 eager 模式时显式传 `--enforce-eager true`，结果另开实验；模型上下文预算必须容纳最长质量输入加 128 个输出 token。

若性能实验专门采用官方 FP8 的 Marlin/W8A16 路径，质量对照也应显式选择该路径，不能用 auto 的质量结果替代：

```bash
python capstone/quantization_profiling/scripts/compare_quality.py \
  --model bf16 --model fp8-official --fp8-backend marlin
```

vLLM 0.14.1 的本地 compressed-tensors 路径不使用该强制 Marlin 选项，默认四模型比较请保留 auto。

## 2. 测什么、怎样判分

| 类型 | 数量 | 判分内容 |
| --- | --- | --- |
| arithmetic | 25 | 整数计算结果严格一致 |
| extraction | 25 | 提取正确的订单编号 |
| choice | 25 | 最大数对应的选项字母 |
| json | 25 | JSON 对象的键、值、类型严格一致 |

数据和规则来自快照中的 `quality_holdout.jsonl`、`quality_protocol.json`，沿用现有 `evaluate_quality.py`，不改旧规则或旧 manifest。判分器文件固定 LF 换行，防止 Windows 自动转换 CRLF 改变冻结 hash。

生成固定为 temperature=0、top_p=1、top_k=-1、seed=42、thinking=false、max_tokens=128、ignore_eos=false，并发为 1。输出使用正常 EOS 结束，不设置固定长度的 min_tokens，不对性能压测回答评分。

- 文本仅去除首尾空白；多余解释、大小写或格式变化仍按冻结规则判错。
- JSON 忽略对象键顺序；拒绝重复键、错误类型、额外字段、代码围栏。
- `finish_reason=length` 计为质量错误，即使截断前的文本恰好包含答案。
- HTTP 错误、超时和缺题属于未完成；不自动重试，不将失败请求当作错误答案计入准确率。
- 服务返回的每题 `usage.prompt_tokens` 必须与冻结模板渲染出的实际 token 长度一致，否则该模型未完成。

预先冻结的门槛：**量化准确率 − BF16 准确率 ≥ −2 个百分点**，包含恰好下降 2 的情况。例如 BF16 为 98/100，AWQ 为 96/100，恰好通过；95/100 则未通过。门槛只针对总体分数，分项变化单独展示。

同时统计 BF16 对/量化错的新增错误、BF16 错/量化对的修复错误，以及双方都对、双方都错。净分数可能抵消新增与修复，不能只看总分。

## 3. 结果在哪里

脚本打印 `Artifacts: .../log/quality_<ID>` 和报告路径。每次运行建立新目录；指定 `--output` 时目标目录必须不存在。

```text
log/quality_<ID>/
├── run.json                      # 配置、环境版本、协议、源码和快照指纹
├── environment.json / .log       # 环境检测结果
├── pip-freeze.txt                # 实际安装依赖
├── inputs/                      # 冻结任务、规则、模板、模型信息与原 manifest
├── quality_requests.jsonl        # 全部 100 题 messages、渲染文本、token IDs
├── <模型>/
│   ├── deployment.json           # 启动命令、权重指纹、实际引擎核验与错误
│   ├── server.log                # 完整服务日志
│   └── evaluation/
│       ├── results.jsonl         # 原始回答、结束原因、usage、逐题分数/请求错误
│       └── summary.json          # 单模型原始评测汇总
├── summary.json                 # 四模型结果与相对 BF16 的质量门槛
├── quality.csv                  # 总体得分和配对变化
├── quality_categories.csv       # 每类正确数、分母、准确率和百分点差值
├── paired_results.jsonl         # 每个量化模型与 BF16 的逐题配对
├── regressions.jsonl            # 全部新增错误：问题、参考答案、双方原始回答
└── Quality_Report.md             # 可直接阅读的质量对比报告
```

汇总会重新使用冻结判分器复核原始回答，并检查 ID、输入 token、模型标签、快照指纹和结果文件 hash。底层单模型 `evaluation/summary.json` 中的 `precision_verified=false` 保持原含义；自动流程另在上层 `deployment.json` 记录权重身份和服务实际引擎核验。

启动失败也保存 deployment.json/server.log，继续尝试后续模型；若 BF16 未完成，量化模型可以保留自身分数，但所有相对差值和门槛判断留空。缺数据使用空值，不输出 0 分或伪造通过。

程序退出码：0=所选模型全部完成且质量门槛通过；1=运行失败或结果不完整；2=完整评测后质量门槛未通过（命令参数错误也可能返回 2）。最终以 `summary.json` 的 status 为准：passed / failed / incomplete。

## 4. 重建报告与分析新增错误

评测结束已经自动生成报告。需要重建时，替换为打印的实际目录：

```bash
python capstone/quantization_profiling/scripts/compare_quality.py \
  --summarize capstone/quantization_profiling/log/quality_实际实验ID
```

重建仅用 Python 标准库，不启动模型、也不重新推理。完整复制质量实验目录到 Windows 后可执行同一命令；判分器须保持冻结版本。性能报告仍用 `summarize.py`，质量报告用本节命令，二者分别保留。

先查看 Quality_Report.md 的总体与分项，再逐条打开 regressions.jsonl。区分计算/抽取答案变化、格式不合规和长度截断；结合 deployment.json 确认比较的是相同执行设置。保留未通过结论，调整量化配方后建立新实验，不能覆盖原输出或事后放宽判分规则。

这 100 条是自编基础合成回归题，可能四模型都满分；这表示在这套题上未观察到差异。它不覆盖复杂推理、代码、知识或长文本任务。总体 1 题=1 个百分点，分项 1 题=4 个百分点；通过不证明广泛任务无退化。不要用本留出集校准、调参后继续把它当作未见测试集。

本地自动测试使用模拟 HTTP 响应，只验证流程、计分和错误处理。真实 4090D 得分必须由上述命令在目标机生成；代码实现完成不等于已经取得模型质量结论。
