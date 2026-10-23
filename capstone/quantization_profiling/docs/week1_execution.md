# 第一周后续：BF16、AWQ、官方 FP8 和本地 FP8 的统一评测

更新：2026-10-22。用户提供的 BF16 实验 `20261020T084357Z_294018cc` 已完成五组负载、每组三轮、每轮100个正式请求。该报告与快照保持原样。本次实现扩展 AWQ/官方 FP8/本地 FP8，并把新运行的默认值改为每组每轮50个正式请求、1轮；新路线的目标机推理、性能与质量仍需实测。

**当前统一环境：** 按 [README 安装入口](../README.md#vllm-0112-统一量化与推理环境) 使用 vLLM 0.14.1、LLM Compressor 0.9.0.5、torch 2.9.1+cu128、Transformers 4.57.3 和 compressed-tensors 0.13.0，目录沿用 vllm-0112。benchmark 版本预检已接受该完整组合并记录到run.json；先做短请求验证，再执行四模型同环境对照。历史BF16仍按旧版本追溯，新组合尚未在目标GPU完成正式矩阵。

## 1. 当前协议与模型

配置文件使用 JSON 语法（YAML 1.2 的子集），脚本用标准库读取，请保留 JSON 格式。

| benchmark 选择 | 运行脚本别名 | 默认本地目录 | 方案 |
| --- | --- | --- | --- |
| bf16 | qwen3-8b | /workspace/weights_pub/Qwen3-8B | BF16，未量化权重 |
| awq | qwen3-8b-awq | /workspace/weights_pub/Qwen3-8B-AWQ | W4A16，awq_marlin + BF16 |
| fp8-official | qwen3-8b-fp8 | /workspace/weights_pub/Qwen3-8B-FP8 | 官方分块FP8，原生fp8加载器 |
| fp8-local | qwen3-8b-fp8-local | /workspace/weights_pub/Qwen3-8B-FP8-local | 分块FP8，按config自动识别compressed-tensors |

benchmark 的 `--model` 也接受上表 qwen3 别名。模型目录名不能证明精度：脚本核对 config 的量化信息和服务启动日志中的实际引擎 dtype/量化方法。AWQ 默认使用支持 BF16 的 Marlin 路径；若改用只支持 FP16 的原 AWQ 后端，应另建配置并记录 dtype 差异。本地 FP8 产物保留自身 `quant_method=compressed-tensors`，不要改成 `fp8`。官方 FP8 单独使用原生 `quant_method=fp8`，核验 E4M3、dynamic 激活和 `[128, 128]` 权重分块；配置依据 [官方 config.json](https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/config.json)。这两个部署标签不证明权重来源相同，仍需记录来源和文件指纹。为兼容旧命令，`fp8` 短别名仍指向本地 FP8；官方请显式使用 `fp8-official` 或 `qwen3-8b-fp8`。

共同控制变量：

- 五组输入/输出长度为128/128、1024/128、4096/256、8192/128、1024/512；默认客户端并发1。
- 每组每轮先预热5次，再正式50次；默认1轮。各模型、各轮使用同一请求子集和顺序。单模型五组共250个正式请求、25个预热请求；四模型共1000个正式请求、100个预热请求。
- `--rounds N` 设置轮数，`--requests N` 设置每组每轮正式请求数；均为正整数。请求数不能超过快照可用记录数。需要衡量跨轮波动时显式增加轮数，例如 `--rounds 3`。
- 性能采样 greedy、seed=42、thinking关闭；设置ignore_eos=true且min_tokens=max_tokens，强制输出长度。实际token数不符记失败；这些回答不用于质量评分。
- 关闭跨请求prefix caching，请求内部KV cache正常使用。每个模型的每组负载单独重启服务，同组各轮之间不清空CUDA allocator。预热与正式记录分开。
- 使用固定并发closed-loop请求。总吞吐=成功输出token总数/首个实际提交至最后一个完成或失败的墙钟窗口，失败等待也计入窗口；线程池排队等待不混入单请求E2E。
- p50/p95为成功请求nearest-rank分位数，失败率单列。TTFT取首个非空文本事件，TPOT取首末文本事件间隔/(输出token数−1)，均为客户端近似。SSE chunk不等于单个token，不能解释为纯GPU prefill/decode时间。

当前每轮50样本的p95比100样本更不稳定；一轮不能估计跨轮波动。报告应明确实际样本和轮数，不能将新50/1结果与旧100/3结果直接混合求均值。

## 2. 冻结数据与复用旧快照

### 2.1 生成 benchmark 测试数据

测试数据由 `scripts/prepare_workloads.py` 在本地构造，无需额外下载数据集。输入是 BF16 源模型的 tokenizer/模板和 `configs/workloads.yaml`，输出是 benchmark 读取的固定请求快照。BF16、AWQ、官方 FP8 和本地 FP8 共用同一份快照，不要按量化模型分别生成不同请求。

在 Linux 4090D 主机上，从仓库根目录执行：

```bash
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

# 首次生成：默认按配置每组50条正式请求，输出目录必须尚不存在。
python capstone/quantization_profiling/scripts/prepare_workloads.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --config capstone/quantization_profiling/configs/workloads.yaml \
  --output capstone/quantization_profiling/data/frozen
```

使用前，`--model` 目录应已有未量化 Qwen3-8B 的 config、tokenizer/模板和 safetensors 权重；脚本使用 `local_files_only=True`，不会自动下载缺失文件。准备请求及正式服务使用同一套锁定依赖；旧 cutedsl 环境仅用于历史复现，不与新环境结果混合。

生成阶段不启动 vLLM、不把模型权重加载到 GPU，也不生成模型回答；它会读取完整权重文件计算 SHA256，因此仍需本地权重，耗时取决于磁盘速度。看到 `Hashing local weights and tokenizer files` 时等待完成，最后输出 `Frozen: ...`。

参数含义：

| 参数 | 用途 |
| --- | --- |
| `--model` | BF16 源模型目录，用于 tokenizer、结构信息、权重指纹和理论显存预算；不要传 AWQ/FP8 模型 |
| `--config` | 冻结前的负载配置，决定输入/输出 token 长度、每组请求数、预热与评测协议 |
| `--requests N` | 每组负载生成的正式请求数，正整数，不含预热；覆盖配置中的 `requests`，省略时使用配置值（仓库默认50） |
| `--output` | 新快照目录；包含后续 benchmark 所需的请求、模板、规则和文件指纹 |
| `--revision` | 可选，只填写已确认的源仓库不可变 commit；省略时如实记录 `unknown`，仍用本地文件 SHA256 固定内容 |

### 2.2 数据内容与 token 长度

脚本按请求编号构造合成提示，在正文中重复 ` x` 调整长度；先完整渲染 Qwen3 聊天模板（thinking 关闭，添加生成提示），再用真实 tokenizer 编码，反复调整直到最终输入长度精确相等。长度包含模板、角色标记和生成提示，不能用字符数代替。无法构造精确长度时直接退出，不截断模板。

未传 `--requests` 时，仓库默认配置生成：

| 负载 | 最终输入 token | 目标输出 token | 预热请求 | 正式请求 |
| --- | --- | --- | --- | --- |
| `w128_o128` | 128 | 128 | 5 | 50 |
| `w1024_o128` | 1024 | 128 | 5 | 50 |
| `w4096_o256` | 4096 | 256 | 5 | 50 |
| `w8192_o128` | 8192 | 128 | 5 | 50 |
| `w1024_o512` | 1024 | 512 | 5 | 50 |

默认共 **275 条性能请求（25 条预热 + 250 条正式）**；指定 `--requests 200` 时，五组共 **1025 条（25 条预热 + 1000 条正式）**，每组的 token 长度不变。`requests.jsonl` 每行包含请求 ID、负载、预热/正式标记、messages、渲染文本、`prompt_token_ids` 和目标输入/输出长度。输出 token 数是运行时要满足的预算，快照不包含性能请求的参考答案。这些合成请求用于控制长度和比较性能，不用于评估回答质量。

正式运行调用 `/v1/completions` 并直接发送冻结的 token IDs，避免重复套模板；响应 usage 再次核验输入/输出长度。默认关闭 prefix caching，保证四种模型使用同一请求口径。

### 2.3 生成的文件

默认目录为 `capstone/quantization_profiling/data/frozen/`：

| 文件 | 内容 |
| --- | --- |
| `requests.jsonl` | 性能请求及最终 token IDs；默认275条，使用 `--requests 200` 时为1025条 |
| `workloads.yaml` | 本次有效负载和协议副本，保存 `--requests` 覆盖后的数量；原配置文件不变 |
| `chat_template.jinja` | 本次实际使用的聊天模板 |
| `model.json` | BF16 源模型路径、来源 revision、结构、文件 SHA256、参数量及权重/KV 理论预算 |
| `quality_holdout.jsonl` | 从仓库 `data/` 复制的100条质量任务；独立于性能请求，不受 `--requests` 影响 |
| `quality_protocol.json` | 从仓库复制的质量判分规则 |
| `manifest.json` | 快照文件的 SHA256 和质量评分脚本指纹，用于校验与追溯 |

理论显存预算不含激活、workspace、allocator、分页和运行时余量。生成期间保持模型目录内容不变；生成后不要手工修改快照文件，否则后续 hash 校验会失败。

### 2.4 校验快照并运行 benchmark

生成成功后，先检查快照文件 hash、请求数量和负载配置，打印四模型运行计划；这一步不启动 GPU 服务，也不验证实际模型权重是否可加载：

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot capstone/quantization_profiling/data/frozen \
  --all-models --requests 50 --rounds 1 --plan-only
```

确认四个模型的本地路径和量化配置正确、7081 端口空闲后，去掉 `--plan-only` 执行正式测试：

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot capstone/quantization_profiling/data/frozen \
  --all-models --requests 50 --rounds 1
```

四模型复用同一份 275 条快照；每模型五组各运行一次，合计 1000 条正式请求和 100 条预热请求。只测 BF16 可把 `--all-models` 换成 `--model bf16`；只验收短请求一组可加 `--case w128_o128`。

### 2.5 已有快照、重建和自定义数量

**已有有效快照可以直接复用。** 脚本拒绝覆盖已有输出目录，即使目录为空也会拒绝。扩展50条快照到200条时，指定新目录（例如 `data/frozen_200`），保留旧实验的快照和 hash。

两个脚本的同名参数用途不同：

| 参数 | 控制内容 |
| --- | --- |
| `prepare_workloads.py --requests N` | 在新快照中为每组生成 N 条正式请求，预热按配置另加；数量写入冻结的 `workloads.yaml` |
| `benchmark.py --requests N` | 每个模型、每组、每轮实际执行 N 条正式请求；默认50，不能超过所选快照的容量，不会额外生成数据 |

每组200条的生成、预览和正式运行可使用下面的配套命令；修改数量时也选择一个尚不存在的快照目录：

```bash
# Linux Bash，从仓库根目录执行。数量可改为100、200、500等正整数。
REQUESTS_PER_CASE=200
SNAPSHOT_DIR="capstone/quantization_profiling/data/frozen_200"

# 生成新的固定请求快照，原 configs/workloads.yaml 不变。
python capstone/quantization_profiling/scripts/prepare_workloads.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --config capstone/quantization_profiling/configs/workloads.yaml \
  --requests "$REQUESTS_PER_CASE" \
  --output "$SNAPSHOT_DIR"

# 校验快照并预览四模型测试计划，不启动GPU服务。
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot "$SNAPSHOT_DIR" --all-models \
  --requests "$REQUESTS_PER_CASE" --rounds 1 --plan-only

# 正式执行每组200条、1轮；确保模型已就绪且7081端口空闲。
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot "$SNAPSHOT_DIR" --all-models \
  --requests "$REQUESTS_PER_CASE" --rounds 1
```

`--requests 200` 表示**每组200条正式请求**，不是五组合计200条。保持五组和每组5条预热时：

| 每组正式请求数 | 快照正式记录 | 快照预热记录 | 快照性能记录总数 | 四模型1轮正式请求总数 |
| --- | --- | --- | --- | --- |
| 50（默认配置） | 250 | 25 | 275 | 1000 |
| 200 | 1000 | 25 | 1025 | 4000 |

四模型1轮均另执行100条预热请求；100条质量任务不随性能数量变化。如果需要五组合计200条正式请求（每个模型），使用 `--requests 40`，另有25条预热记录，共225条性能记录。一般公式为：快照性能记录数 = 负载组数 ×（每组正式请求数 + 每组预热数）。增加 `--rounds` 只会重复使用快照，不增加快照中的记录。

**生成200条后，benchmark 仍须显式传 `--requests 200`。** 如果省略 benchmark 的该参数，仍默认只选每组前50条；也可有意用200条快照执行 `--requests 50`，先做较短测试。四模型、各轮始终复用相同的请求子集和顺序。

**旧100条/3轮快照无需重建。** benchmark不会修改它，默认按顺序选择前50条；显式使用 `--requests 100 --rounds 3` 可以重跑旧协议。新运行保存原快照协议、本次有效协议、选中请求及子集hash。需要超过现有容量时，重新生成到新目录，而不是手工改动已冻结文件。仍可通过自定义配置文件中的 `requests` 设定生成数量；同时指定CLI参数时以 `--requests` 为准。

## 3. 选择模型、轮数和请求数

先检查快照完整性并生成启动命令，不启动GPU服务；预览需要有效快照，但不检查本地模型权重是否存在或可加载：

```bash
# 默认只选BF16，每组50次正式请求、1轮。
python capstone/quantization_profiling/scripts/benchmark.py --plan-only

# 一次预览四种模型。
python capstone/quantization_profiling/scripts/benchmark.py --all-models --plan-only
```

正式运行：

```bash
# 默认BF16，五组共250次正式请求。
python capstone/quantization_profiling/scripts/benchmark.py

# 单个量化模型。
python capstone/quantization_profiling/scripts/benchmark.py --model awq
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-official
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-local

# 仅验收短请求一组：每种模型预热5次、正式50次、1轮。
python capstone/quantization_profiling/scripts/benchmark.py \
  --all-models --case w128_o128

# 四模型依次运行；每组两轮，每轮50次正式请求。
python capstone/quantization_profiling/scripts/benchmark.py --all-models --rounds 2

# 自选两种模型，--model可重复；缩小到每轮20次正式请求。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model bf16 --model awq --requests 20 --rounds 1

# 复用原来100请求快照，显式恢复原轮次/样本数。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model bf16 --snapshot capstone/quantization_profiling/data/frozen \
  --requests 100 --rounds 3
```

`--case` 可重复以筛选五组中的若干组。`--config <JSON格式YAML文件>` 可指定单模型自定义配置，与 `--model`、`--all-models` 互斥。默认模型配置为 configs/bf16.yaml、configs/awq.yaml、configs/fp8-official.yaml、configs/fp8.yaml，最后一项保留为本地 FP8 配置。

benchmark在本机自动管理服务，默认监听 `0.0.0.0:7081`，客户端访问回环地址；各模型串行测试。**7081必须空闲**，请先正常停止手工启动的旧服务。脚本不会杀死或复用外部服务；只有本次创建的进程组会在对应负载结束后清理。

每组上下文上限为输入+输出token预算；固定eager、关闭prefix caching、KV dtype为auto。服务启动日志必须核验实际dtype和量化方法，不能只凭模型名宣称BF16/AWQ/FP8生效。脚本记录被测模型自身文件指纹，并区分冻结请求对应的BF16源模型，量化权重不需要与BF16权重文件hash相同。

所有模型共用冻结token IDs及tokenizer口径；量化服务的--tokenizer显式指向快照model.json中记录的BF16源模型目录。换机器后该路径若已变化，可在单模型自定义配置中显式指定tokenizer目录，并核验快照对应的tokenizer文件指纹。脚本核对模型结构并保存被测权重自身hash；现成AWQ若缺少可确认的基础revision/校准信息，应称为部署方案比较，同tokenizer或同架构不能证明来自相同基础权重。本地FP8若有quantization_manifest.json，会额外核验源和产物hash；缺少manifest时不能强认与BF16同源。

请求必须返回profiling，token数须与usage/快照一致。缺少指标、长度不符、HTTP拒绝、超时、OOM均保留错误。预热失败不开始该轮正式测量，后续负载仍保留尝试/失败记录。

## 4. 日志、显存和报告

### 4.1 日志与显存

运行结束打印 `log/benchmark_<时间>_<ID>/`，不同运行创建独立目录：

- 根run.json/summary.json：模型选择、完整配置、原快照与本次有效协议、请求子集指纹和分模型汇总；selected_requests.jsonl保存实际选中的请求记录。
- environment.json/.log、pip-freeze.txt、code.json：目标机环境、包清单、源码hash、Git commit/工作区状态。
- `<模型>/<负载>/server.log`：完整服务启动、实际引擎配置和错误日志。
- 对应requests.jsonl：模型、预热/正式、轮次、请求ID、token数、回答、客户端/服务端计时和失败类型。
- 对应gpu.jsonl和summary.json：加载前/中/后、预热/正式阶段的设备显存采样及按GPU UUID统计的采样峰值。

nvidia-smi默认每0.5秒采样，是含其它进程的整卡占用，可能漏过瞬时峰值，不等于PyTorch allocated/reserved。没有有效采样则该组不能通过。socket timeout为单次连接/读超时，不是整个请求绝对deadline。正式计时期间不挂nsys/ncu。

### 4.2 汇总已有 benchmark 结果

`summarize.py` 在 benchmark 完成后手工执行，读取一次实验的已有结果，不启动服务、不重新推理。benchmark 结束会打印 `Artifacts: .../log/benchmark_<ID>`；将该目录作为唯一的位置参数 `run`。不要传 `data/frozen`、整个 `log/`、单个模型/负载子目录或 `summary.json` 文件本身。

Linux 命令（从仓库根目录执行）：

```bash
cd /workspace/AI_system

# 查看现有实验目录，选择要汇总的一次运行。
ls -d capstone/quantization_profiling/log/benchmark_*/

# 以下是历史实验 ID 示例，请改为本次 benchmark 打印的目录。
RUN_DIR="capstone/quantization_profiling/log/benchmark_20261020T084357Z_294018cc"
python capstone/quantization_profiling/scripts/summarize.py "$RUN_DIR"
```

脚本读取实验根目录中的 `run.json`（模型、负载、请求数、轮数与快照指纹）和 `summary.json`（各组已计算的性能统计、状态和失败信息），并按记录的相对路径读取各模型/负载的 `requests.jsonl`。四模型在同一次 `--all-models` 实验中完成时，一次汇总就包含四种模型；分别运行产生的多个实验目录需要分别汇总，此入口不会合并多个实验。

| 输出文件 | 默认位置与内容 |
| --- | --- |
| `Week1_Report.md` | 实验根目录；包含实验/快照/请求子集指纹、有效协议、分模型/负载/正式轮次的成功数、吞吐、E2E p95、TTFT 文本事件 p95，以及验收状态、失败与限制 |
| `performance.jsonl` | 实验根目录；合并各组已有的逐请求明细，附加 `run_id` 与 `model_profile`；包含预热和正式记录，可用 `phase` 区分 |

显存仍需查看各模型/负载的 `summary.json` 和 `gpu.jsonl`；该 Markdown 报告会说明这些记录的位置与测量限制。报告表格使用 benchmark 已保存的统计，不会重新计算原始请求的 p95。它不包含质量回归或 Nsight 分析。

### 4.3 自定义报告路径与 Windows 汇总

可选参数 `--output` 指定 Markdown 文件路径，父目录会自动创建：

```bash
python capstone/quantization_profiling/scripts/summarize.py "$RUN_DIR" \
  --output capstone/quantization_profiling/reports/Benchmark_Report.md
```

`--output` 只改变 Markdown 位置，`performance.jsonl` 仍写在 `RUN_DIR`。重复执行会覆盖同路径的 Markdown 和该实验目录中的 `performance.jsonl`；根 `data/performance.jsonl` 不自动覆盖。

汇总脚本只依赖 Python 标准库，无需安装 torch、Transformers 或 vLLM。也可以把 Linux 的**完整** `log/benchmark_<ID>/` 目录复制到 Windows 项目的 `log/` 下，保留模型/负载子目录和原有文件，再用 PowerShell 执行：

```powershell
cd D:\workspace\learing\AI_system

# 替换为已复制到本机的真实实验目录。
$benchmarkRun = "capstone/quantization_profiling/log/benchmark_20261020T084357Z_294018cc"
python capstone/quantization_profiling/scripts/summarize.py $benchmarkRun
```

模型权重无需复制。根 `run.json` 和 `summary.json` 用于生成表格，各组 `requests.jsonl` 用于导出完整请求明细；只复制两个根 JSON 会使明细不完整。普通 `test_server_*.json` 日志不是此脚本的输入。

报告按本次声明的模型、负载、轮数和请求数验收，不强制至少三轮；只选部分 case 时按所选 case 验收，并说明覆盖范围。历史单模型报告仍按原记录读取。失败配置没有指标时保留空缺，不填 0ms。命令成功生成文件不等于实验全部通过，应查看报告中的“本次所选模型/负载全部满足有效协议”和失败列表。

## 5. 质量对比

四类固定任务各 25 条，合计 100 条；沿用已冻结的原始判分器。质量生成正常 EOS 停止，与性能测试的强制输出长度协议分开。

从 Linux 4090D 主机的仓库根目录执行，先停止占用 7081 的已有服务：

```bash
source /workspace/venvs/vllm-0112/bin/activate
python capstone/quantization_profiling/scripts/compare_quality.py --plan-only
python capstone/quantization_profiling/scripts/compare_quality.py
```

默认 BF16、AWQ、官方 FP8、本地 FP8，共 400 次请求，自动顺序启动和停止服务。使用另一份快照时增加 `--snapshot <目录>`；性能数据数量不影响质量集 100 题。结果保存在 `log/quality_<ID>/`，自动生成 Quality_Report.md、总体/分类 CSV 和逐题配对/新增错误 JSONL。

门槛已冻结为相对 BF16 下降不超过 2 个百分点。HTTP/超时/缺题为 incomplete，不填质量分；格式错误和截断按原规则判错。服务实际 dtype/量化方式与模型文件指纹一并保存。

完整协议、模型选择、错误解读、Windows 离线重建和输出文件见 [四模型质量对比](quality_comparison.md)。已有远程服务的单模型入口 `evaluate_quality.py` 仍保留；自动四模型对比优先使用 compare_quality.py。

## 6. 验收与剩余工作

1. 核对GPU、依赖、其它进程和环境快照，补查unknown来源；模型权重与冻结请求的指纹可追溯。
2. 四模型对相同tokenizer/token IDs和请求子集运行；每个所选case达到本次声明的轮数/请求数，失败和OOM如实保留。
3. 实际引擎dtype/量化配置、服务端指标、显存采样及原始请求齐全；量化实际kernel另用Profiler确认。
4. 结合真实结构与显存记录解释至少一组权重/KV预算；分别报告50样本p95与单轮波动的限制。
5. 完成四模型100条留出集质量回归、Nsight阶段/热点诊断及报告。这些仍是后续实测任务，代码支持不等于已经完成。

接口依据：[vLLM 0.11.0官方服务接口](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)、[该版本请求协议源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/protocol.py)。
