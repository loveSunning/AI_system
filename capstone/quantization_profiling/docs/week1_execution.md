# 第一周：固定数据、BF16 基线与验收

更新：2026-10-20。实现已交付，本地测试不代表目标 4090D 上的实验通过。上一份真实日志证明远程生成可用，但缺少服务端 profiling。新流程自动启动带中间件的服务；必须通过下述目标机实测才能关闭该未完成项。

## 1. 已实现的协议

配置文件使用 JSON 语法（YAML 1.2 的子集），脚本仅使用标准库读取，不增加 PyYAML 依赖。请保留 JSON 格式。

- `configs/workloads.yaml`：128/128、1024/128、4096/256、8192/128、1024/512 五组。
- 默认客户端并发 1；每组每轮预热 5 次，再正式 100 次，重复 3 轮。所有轮次使用同一请求序列。共 1500 个正式请求、75 个预热请求。
- 性能采样为 greedy、seed=42、thinking 关闭。设置 ignore_eos=true 且 min_tokens=max_tokens，强制输出长度；任何实际 token 数不符均记录失败。这些文本不能用来做质量评测。
- 关闭跨请求 prefix caching；请求内部 KV cache 正常启用。每组重新启动服务，同组各轮之间不清空 CUDA allocator。预热覆盖相同输入/输出 shape，预热数据与正式统计分开。
- 固定并发 closed-loop 请求，线程空闲后处理下一条。总吞吐=成功输出 token 总数/首个实际提交至最后一个完成或失败的墙钟窗口；失败等待也计入窗口。线程池排队等待不混入单请求 E2E。
- p50/p95 为成功请求 nearest-rank 分位数；失败率单列。TTFT 使用首个非空文本事件，TPOT 使用首末文本事件间隔/(输出 token 数-1)，均是客户端近似。SSE chunk 不等于 token，因此不声称精确 GPU decode 时间。
- 配置修改后生成新的快照和实验目录；不覆盖已有快照、不覆盖旧结果，不把不同配置混合求均值。

## 2. 目标机运行顺序

在 Linux 4090D 主机、已安装 vLLM 0.11.0 的虚拟环境中，从仓库根目录执行。模型必须在 `/workspace/weights_pub/Qwen3-8B`。

第一步，冻结请求、模板、模型文件内容和质量规则：

```bash
python capstone/quantization_profiling/scripts/prepare_workloads.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --output capstone/quantization_profiling/data/frozen
```

如果已查到 ModelScope/Hugging Face 源仓库实际不可变 commit，可增加 `--revision <真实commit>`。不要填写猜测值。未填写时来源 revision 记为 unknown；脚本仍对本地权重、配置和 tokenizer 文件做完整 SHA256，固定实际使用的内容。哈希可能需要数分钟，模型目录应保持静止。

脚本使用目标机实际 tokenizer，完整渲染模板，再校验最终输入长度；保存 messages、渲染文本和 token IDs。精确长度构造失败就退出，不截断模板。性能调用 `/v1/completions` 并直接发送这些 token IDs，避免服务端重复套 chat template；响应 usage 再次验证输入和输出长度。

快照包括 requests.jsonl、workloads.yaml、chat_template.jinja、model.json、质量数据/规则、manifest.json。model.json 包含实际结构、权重参数量和 BF16 payload、每 token KV 字节数及各负载的 KV 理论预算。理论预算未包含激活、workspace、allocator、分页和运行时余量，须与实测对照。

第二步，检查命令和快照完整性，不启动 GPU 服务：

```bash
python capstone/quantization_profiling/scripts/benchmark.py --plan-only
```

第三步，运行主矩阵：

```bash
python capstone/quantization_profiling/scripts/benchmark.py
```

benchmark 在本机管理服务，监听 `0.0.0.0:7081`，客户端访问回环地址。**7081 必须空闲**；请先正常停止你之前启动的旧服务。脚本不会杀死或复用已有服务。若要并存，改 bf16.yaml 的端口并将它视为新配置。

也可先只验收第一组（仍然是 5 次预热/100 次正式/3 轮），再运行全矩阵；两次结果分别保存：

```bash
python capstone/quantization_profiling/scripts/benchmark.py --case w128_o128
```

每组使用 input+output 作为上下文上限，BF16 dtype 显式指定，拒绝带 quantization_config 的模型；核对权重/tokenizer 文件 hash 和 torch/Transformers/vLLM 版本。服务启动日志必须出现已解析的 `dtype=torch.bfloat16`，否则停止该组，不能仅凭配置名称宣称 BF16 已验证。KV dtype 固定 auto，当前随 BF16 模型；enforce_eager 固定 true，CUDA Graph 优化属于后续独立实验。

请求必须返回 profiling 且其 token 数与 usage/快照一致。缺少指标、长度不符、HTTP 拒绝、超时、OOM 都保留原始错误。预热失败不开始该轮正式测量；下一负载仍继续尝试。每组结束只清理由本次启动的进程组。

## 3. 结果和显存记录

运行结束会打印 `log/benchmark_<时间>_<ID>/`。其中包含：

- run.json：完整配置、协议、快照指纹和模型信息。
- environment.json/.log、pip-freeze.txt、code.json：目标机环境、包列表、源码 hash、Git commit/工作区状态。
- 每个 workload 的 server.log：完整服务启动和错误日志。
- requests.jsonl：预热/正式标记、轮次、请求 ID、token 数、回答、计时、服务端指标、错误类型。
- gpu.jsonl：加载前、加载中、加载后、预热和正式阶段的设备显存采样。
- summary.json：逐轮延迟、吞吐、错误率和按阶段/GPU UUID 记录的显存采样峰值。

nvidia-smi 默认每 0.5 秒采样，是设备总占用（含其它进程），不是 PyTorch allocated/reserved，也可能漏过瞬时峰值。没有有效采样则该组不能通过验收。socket timeout 为单次读/连接超时，不是整个请求绝对 deadline。运行可能持续较长时间；正式计时期间不启用 nsys/ncu。

生成报告及带实验 ID 的性能汇总：

```bash
python capstone/quantization_profiling/scripts/summarize.py \
  capstone/quantization_profiling/log/benchmark_<实际实验ID>
```

输出在该实验目录的 Week1_Report.md 和 performance.jsonl。根 data/performance.jsonl 不自动覆盖，避免把多次实验混到同一文件。失败配置也会出现在报告中；没有测量值时保留空缺，不填 0ms。

## 4. 质量数据和判分

`data/quality_holdout.jsonl` 已包含 100 条项目自编合成任务，每类 25 条：整数运算、最大值选择、字段抽取、JSON 结构指令。样本 ID 唯一，参考答案显式保存；数据来源和固定规则写入 quality_protocol.json。它是基础教学回归集，难度/覆盖有限，不能代表通用能力或广泛量化质量。

判分只去除答案首尾空白：文本严格匹配；JSON 忽略键顺序，但键、值和类型必须一致，重复键和代码围栏均判错。截断判错。基础设施失败单列，不自动重试，不把未答题当成正常错误；存在此类失败则总体准确率为空并标记 incomplete。预设后续量化门槛为较 BF16 下降不超过 2 个百分点。

质量集不参与性能请求、校准或调参。本周不做自行量化；后续需要独立准备校准数据。判分器 hash 随快照冻结，修改评分代码后必须新建快照。

本周要求冻结质量集与规则；完整 BF16/AWQ 配对质量回归属于后续 S5。已提供评测入口，可先保存 BF16 质量基线。benchmark 完成会停止服务，需要另起终端：

```bash
bash capstone/quantization_profiling/scripts/serve.sh \
  --model /workspace/weights_pub/Qwen3-8B --max-model-len 2048

# 在另一个终端运行：
python capstone/quantization_profiling/scripts/evaluate_quality.py \
  --label bf16 --base-url http://127.0.0.1:7081 \
  --output capstone/quantization_profiling/log/quality_bf16_001
```

label 是标记，不证明远程服务器精度；需关联真实服务启动日志。质量评测不启用 ignore_eos/min_tokens，最大输出 128，greedy、thinking 关闭。不要通过调参针对该 100 题优化后继续称它为未见留出集。

## 5. 本周验收条件

1. 目标 GPU、实际依赖和环境快照齐全；模型内容固定，unknown revision 如实保留并补查来源。
2. 五组请求均经过真实 tokenizer 校验，快照完整性验证通过。
3. 100 条质量任务和判分规则已冻结。
4. 五组负载均有结果或明确容量/失败证据；共同可运行配置每组至少 3 轮正式测量。
5. BF16 引擎日志证据、服务端指标、显存采样和原始请求可追溯。
6. 能用理论权重/KV 与实测显存解释至少一组负载，并在报告中说明差额。

当前本地实现验证包含原有 70 项测试，以及新增 6 项协议/评分/HTTP 集成测试。模拟 HTTP 测试验证了中间件到客户端的指标链路，不是 GPU 性能数据。目标机真实 BF16 启动、精确 tokenizer 快照、15 轮测量和显存预算解释仍待执行，不能提前标记为完成。

接口依据：[vLLM 0.11.0 官方服务接口](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)、[该版本请求协议源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/protocol.py)。
