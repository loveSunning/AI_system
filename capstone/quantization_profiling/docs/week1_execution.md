# 第一周后续：BF16、AWQ 和本地 FP8 的统一评测

更新：2026-10-22。用户提供的 BF16 实验 `20261020T084357Z_294018cc` 已完成五组负载、每组三轮、每轮100个正式请求。该报告与快照保持原样。本次实现扩展 AWQ/本地 FP8，并把新运行的默认值改为每组每轮50个正式请求、1轮；新路线的目标机推理、性能与质量仍需实测。

**当前统一环境：** 按 [README 安装入口](../README.md#vllm-0112-统一量化与推理环境) 使用 vLLM 0.14.1、LLM Compressor 0.9.0.5、torch 2.9.1+cu128、Transformers 4.57.3 和 compressed-tensors 0.13.0，目录沿用 vllm-0112。benchmark 版本预检已接受该完整组合并记录到run.json；先做短请求验证，再执行三模型同环境对照。历史BF16仍按旧版本追溯，新组合尚未在目标GPU完成正式矩阵。

## 1. 当前协议与模型

配置文件使用 JSON 语法（YAML 1.2 的子集），脚本用标准库读取，请保留 JSON 格式。

| benchmark 选择 | 运行脚本别名 | 默认本地目录 | 方案 |
| --- | --- | --- | --- |
| bf16 | qwen3-8b | /workspace/weights_pub/Qwen3-8B | BF16，未量化权重 |
| awq | qwen3-8b-awq | /workspace/weights_pub/Qwen3-8B-AWQ | W4A16，awq_marlin + BF16 |
| fp8-local | qwen3-8b-fp8-local | /workspace/weights_pub/Qwen3-8B-FP8-local | 分块FP8，按config自动识别compressed-tensors |

benchmark 的 `--model` 也接受上表 qwen3 别名。模型目录名不能证明精度：脚本核对 config 的量化信息和服务启动日志中的实际引擎 dtype/量化方法。AWQ 默认使用支持 BF16 的 Marlin 路径；若改用只支持 FP16 的原 AWQ 后端，应另建配置并记录 dtype 差异。FP8 产物保留自身 `quant_method=compressed-tensors`，不要改成 `fp8`。

共同控制变量：

- 五组输入/输出长度为128/128、1024/128、4096/256、8192/128、1024/512；默认客户端并发1。
- 每组每轮先预热5次，再正式50次；默认1轮。各模型、各轮使用同一请求子集和顺序。单模型五组共250个正式请求、25个预热请求；三模型共750个正式请求、75个预热请求。
- `--rounds N` 设置轮数，`--requests N` 设置每组每轮正式请求数；均为正整数。请求数不能超过快照可用记录数。需要衡量跨轮波动时显式增加轮数，例如 `--rounds 3`。
- 性能采样 greedy、seed=42、thinking关闭；设置ignore_eos=true且min_tokens=max_tokens，强制输出长度。实际token数不符记失败；这些回答不用于质量评分。
- 关闭跨请求prefix caching，请求内部KV cache正常使用。每个模型的每组负载单独重启服务，同组各轮之间不清空CUDA allocator。预热与正式记录分开。
- 使用固定并发closed-loop请求。总吞吐=成功输出token总数/首个实际提交至最后一个完成或失败的墙钟窗口，失败等待也计入窗口；线程池排队等待不混入单请求E2E。
- p50/p95为成功请求nearest-rank分位数，失败率单列。TTFT取首个非空文本事件，TPOT取首末文本事件间隔/(输出token数−1)，均为客户端近似。SSE chunk不等于单个token，不能解释为纯GPU prefill/decode时间。

当前每轮50样本的p95比100样本更不稳定；一轮不能估计跨轮波动。报告应明确实际样本和轮数，不能将新50/1结果与旧100/3结果直接混合求均值。

## 2. 冻结数据与复用旧快照

### 2.1 生成 benchmark 测试数据

测试数据由 `scripts/prepare_workloads.py` 在本地构造，无需额外下载数据集。输入是 BF16 源模型的 tokenizer/模板和 `configs/workloads.yaml`，输出是 benchmark 读取的固定请求快照。BF16、AWQ 和本地 FP8 共用同一份快照，不要按量化模型分别生成不同请求。

在 Linux 4090D 主机上，从仓库根目录执行：

```bash
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

# 首次生成：输出目录必须尚不存在。
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
| `--output` | 新快照目录；包含后续 benchmark 所需的请求、模板、规则和文件指纹 |
| `--revision` | 可选，只填写已确认的源仓库不可变 commit；省略时如实记录 `unknown`，仍用本地文件 SHA256 固定内容 |

### 2.2 数据内容与 token 长度

脚本按请求编号构造合成提示，在正文中重复 ` x` 调整长度；先完整渲染 Qwen3 聊天模板（thinking 关闭，添加生成提示），再用真实 tokenizer 编码，反复调整直到最终输入长度精确相等。长度包含模板、角色标记和生成提示，不能用字符数代替。无法构造精确长度时直接退出，不截断模板。

当前配置生成：

| 负载 | 最终输入 token | 目标输出 token | 预热请求 | 正式请求 |
| --- | --- | --- | --- | --- |
| `w128_o128` | 128 | 128 | 5 | 50 |
| `w1024_o128` | 1024 | 128 | 5 | 50 |
| `w4096_o256` | 4096 | 256 | 5 | 50 |
| `w8192_o128` | 8192 | 128 | 5 | 50 |
| `w1024_o512` | 1024 | 512 | 5 | 50 |

共 **275 条性能请求（25 条预热 + 250 条正式）**。`requests.jsonl` 每行包含请求 ID、负载、预热/正式标记、messages、渲染文本、`prompt_token_ids` 和目标输入/输出长度。输出 token 数是运行时要满足的预算，快照不包含性能请求的参考答案。这些合成请求用于控制长度和比较性能，不用于评估回答质量。

正式运行调用 `/v1/completions` 并直接发送冻结的 token IDs，避免重复套模板；响应 usage 再次核验输入/输出长度。默认关闭 prefix caching，保证三种模型使用同一请求口径。

### 2.3 生成的文件

默认目录为 `capstone/quantization_profiling/data/frozen/`：

| 文件 | 内容 |
| --- | --- |
| `requests.jsonl` | 275 条性能请求及最终 token IDs |
| `workloads.yaml` | 本次冻结的负载和评测协议副本 |
| `chat_template.jinja` | 本次实际使用的聊天模板 |
| `model.json` | BF16 源模型路径、来源 revision、结构、文件 SHA256、参数量及权重/KV 理论预算 |
| `quality_holdout.jsonl` | 从仓库 `data/` 复制的 100 条质量任务，不计入 275 条性能请求 |
| `quality_protocol.json` | 从仓库复制的质量判分规则 |
| `manifest.json` | 快照文件的 SHA256 和质量评分脚本指纹，用于校验与追溯 |

理论显存预算不含激活、workspace、allocator、分页和运行时余量。生成期间保持模型目录内容不变；生成后不要手工修改快照文件，否则后续 hash 校验会失败。

### 2.4 校验快照并运行 benchmark

生成成功后，先检查快照文件 hash、请求数量和负载配置，打印三模型运行计划；这一步不启动 GPU 服务，也不验证实际模型权重是否可加载：

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot capstone/quantization_profiling/data/frozen \
  --all-models --requests 50 --rounds 1 --plan-only
```

确认三个模型的本地路径和量化配置正确、7081 端口空闲后，去掉 `--plan-only` 执行正式测试：

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot capstone/quantization_profiling/data/frozen \
  --all-models --requests 50 --rounds 1
```

三模型复用同一份 275 条快照；每模型五组各运行一次，合计 750 条正式请求和 75 条预热请求。只测 BF16 可把 `--all-models` 换成 `--model bf16`；只验收短请求一组可加 `--case w128_o128`。

### 2.5 已有快照、重建和自定义数量

**已有有效的 `data/frozen` 可以直接复用，无需重复生成。** 脚本拒绝覆盖已存在的输出目录，即使目录为空也会拒绝。修改 tokenizer、模板或负载后需要新快照时，指定另一个目录，并在 benchmark 中传入相同路径：

```bash
python capstone/quantization_profiling/scripts/prepare_workloads.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --config capstone/quantization_profiling/configs/workloads.yaml \
  --output capstone/quantization_profiling/data/frozen_v2

python capstone/quantization_profiling/scripts/benchmark.py \
  --snapshot capstone/quantization_profiling/data/frozen_v2 \
  --all-models --requests 50 --rounds 1 --plan-only
```

**旧 100 请求/3 轮快照无需重建。** benchmark 不会修改它，默认按固定顺序取每组前 50 个正式请求；所有被测模型和轮次复用相同子集。新运行保存原快照协议、本次有效协议、选中的请求及子集 hash。比较结果时同时核对模型、原快照指纹、子集、轮数、并发和缓存配置。

减少正式请求数或增加轮数只需设置 benchmark 的 `--requests N`、`--rounds N`，不必重新生成；每轮复用同一组请求。若要使用超过快照容量的请求数，例如在新默认 50 条快照上运行 `--requests 100`，应先另存一份 JSON 格式的 workload 配置，把顶层 `requests` 改为 100，再通过生成脚本的 `--config` 冻结到新目录。benchmark 的 `--requests` 只选择已有请求，不会生成额外数据。

## 3. 选择模型、轮数和请求数

先检查快照完整性并生成启动命令，不启动GPU服务；预览需要有效快照，但不检查本地模型权重是否存在或可加载：

```bash
# 默认只选BF16，每组50次正式请求、1轮。
python capstone/quantization_profiling/scripts/benchmark.py --plan-only

# 一次预览三种模型。
python capstone/quantization_profiling/scripts/benchmark.py --all-models --plan-only
```

正式运行：

```bash
# 默认BF16，五组共250次正式请求。
python capstone/quantization_profiling/scripts/benchmark.py

# 单个量化模型。
python capstone/quantization_profiling/scripts/benchmark.py --model awq
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-local

# 仅验收短请求一组：每种模型预热5次、正式50次、1轮。
python capstone/quantization_profiling/scripts/benchmark.py \
  --all-models --case w128_o128

# 三模型依次运行；每组两轮，每轮50次正式请求。
python capstone/quantization_profiling/scripts/benchmark.py --all-models --rounds 2

# 自选两种模型，--model可重复；缩小到每轮20次正式请求。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model bf16 --model awq --requests 20 --rounds 1

# 复用原来100请求快照，显式恢复原轮次/样本数。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model bf16 --snapshot capstone/quantization_profiling/data/frozen \
  --requests 100 --rounds 3
```

`--case` 可重复以筛选五组中的若干组。`--config <JSON格式YAML文件>` 可指定单模型自定义配置，与 `--model`、`--all-models` 互斥。默认模型配置为 configs/bf16.yaml、configs/awq.yaml、configs/fp8.yaml。

benchmark在本机自动管理服务，默认监听 `0.0.0.0:7081`，客户端访问回环地址；各模型串行测试。**7081必须空闲**，请先正常停止手工启动的旧服务。脚本不会杀死或复用外部服务；只有本次创建的进程组会在对应负载结束后清理。

每组上下文上限为输入+输出token预算；固定eager、关闭prefix caching、KV dtype为auto。服务启动日志必须核验实际dtype和量化方法，不能只凭模型名宣称BF16/AWQ/FP8生效。脚本记录被测模型自身文件指纹，并区分冻结请求对应的BF16源模型，量化权重不需要与BF16权重文件hash相同。

所有模型共用冻结token IDs及tokenizer口径；量化服务的--tokenizer显式指向快照model.json中记录的BF16源模型目录。换机器后该路径若已变化，可在单模型自定义配置中显式指定tokenizer目录，并核验快照对应的tokenizer文件指纹。脚本核对模型结构并保存被测权重自身hash；现成AWQ若缺少可确认的基础revision/校准信息，应称为部署方案比较，同tokenizer或同架构不能证明来自相同基础权重。本地FP8若有quantization_manifest.json，会额外核验源和产物hash；缺少manifest时不能强认与BF16同源。

请求必须返回profiling，token数须与usage/快照一致。缺少指标、长度不符、HTTP拒绝、超时、OOM均保留错误。预热失败不开始该轮正式测量，后续负载仍保留尝试/失败记录。

## 4. 日志、显存和报告

运行结束打印 `log/benchmark_<时间>_<ID>/`，不同运行创建独立目录：

- 根run.json/summary.json：模型选择、完整配置、原快照与本次有效协议、请求子集指纹和分模型汇总；selected_requests.jsonl保存实际选中的请求记录。
- environment.json/.log、pip-freeze.txt、code.json：目标机环境、包清单、源码hash、Git commit/工作区状态。
- `<模型>/<负载>/server.log`：完整服务启动、实际引擎配置和错误日志。
- 对应requests.jsonl：模型、预热/正式、轮次、请求ID、token数、回答、客户端/服务端计时和失败类型。
- 对应gpu.jsonl和summary.json：加载前/中/后、预热/正式阶段的设备显存采样及按GPU UUID统计的采样峰值。

nvidia-smi默认每0.5秒采样，是含其它进程的整卡占用，可能漏过瞬时峰值，不等于PyTorch allocated/reserved。没有有效采样则该组不能通过。socket timeout为单次连接/读超时，不是整个请求绝对deadline。正式计时期间不挂nsys/ncu。

```bash
python capstone/quantization_profiling/scripts/summarize.py \
  capstone/quantization_profiling/log/benchmark_<实际实验ID>
```

输出Week1_Report.md和performance.jsonl。报告按模型/负载/轮次分开，验收使用本次声明的所选轮数和请求数，不再强制至少三轮；只选部分case时按所选case验收，并说明覆盖范围。历史单模型报告仍按原记录读取。根data/performance.jsonl不自动覆盖；失败配置没有指标时保留空缺，不填0ms。

## 5. 质量数据和后续回归

`data/quality_holdout.jsonl`仍为100条，不随性能请求数减少。四类各25条：整数运算、最大值选择、字段抽取、JSON结构指令；样本ID唯一，参考答案显式保存。它是自编教学回归集，难度/覆盖有限。

判分只去除答案首尾空白；文本严格匹配；JSON忽略键顺序，但键、值和类型必须一致，重复键和代码围栏判错，截断判错。基础设施失败单列，不自动重试；出现此类失败则总体准确率为空并标记incomplete。后续量化门槛预设为较BF16下降不超过2个百分点。

质量集不参与性能请求或量化校准/调参；BF16→本地FP8的RTN转换见 [FP8说明](fp8_quantization.md)。质量评测采用正常EOS停止，不启用ignore_eos/min_tokens。benchmark结束会停止服务，另开终端启动对应模型，再运行质量入口：

```bash
# 终端1：先测BF16，后续逐个替换成AWQ/本地FP8。
python capstone/quantization_profiling/scripts/run_model.py \
  --serve --model qwen3-8b --host 0.0.0.0 --port 7081 --max-model-len 2048

# 终端2：label仅是标签，需关联真实服务日志。
python capstone/quantization_profiling/scripts/evaluate_quality.py \
  --label bf16 --base-url http://127.0.0.1:7081 \
  --output capstone/quantization_profiling/log/quality_bf16_001
```

分别保存BF16/AWQ/本地FP8逐题结果后再比较，不能把性能输出当质量答案。不要针对这100题调参后继续称为未见留出集。

## 6. 验收与剩余工作

1. 核对GPU、依赖、其它进程和环境快照，补查unknown来源；模型权重与冻结请求的指纹可追溯。
2. 三模型对相同tokenizer/token IDs和请求子集运行；每个所选case达到本次声明的轮数/请求数，失败和OOM如实保留。
3. 实际引擎dtype/量化配置、服务端指标、显存采样及原始请求齐全；量化实际kernel另用Profiler确认。
4. 结合真实结构与显存记录解释至少一组权重/KV预算；分别报告50样本p95与单轮波动的限制。
5. 完成三模型100条留出集质量回归、Nsight阶段/热点诊断及报告。这些仍是后续实测任务，代码支持不等于已经完成。

接口依据：[vLLM 0.11.0官方服务接口](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)、[该版本请求协议源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/protocol.py)。
