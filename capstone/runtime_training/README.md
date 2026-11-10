# P3 vLLM、SGLang 服务与训练架构闭环

本周从 **W31：vLLM API 服务与流式客户端** 开始。以 Qwen3-8B 服务为实践载体，学习一条请求从 HTTP、模板与分词、引擎、调度与 KV 管理，到模型执行和流式返回的过程。本周固定运行配置建立基线，W32 再做调度与显存参数消融。

- 学习清单、20 小时步骤、源码入口：[W31 学习指南](docs/w31_learning.md)。
- 真实 GPU 结果填写：[W31 报告模板](reports/W31_Report.md)。
- 后续安排见本文末尾的 P3 任务表。

当前交付为代码与学习材料。**真实模型加载、GPU 推理、attention backend 和性能数值均待在 GPU 环境验证**；CPU 上的模拟 API 检查仅验证客户端、SSE、指标统计与文件输出。

## 目录与分工

| 路径 | 用途 |
| --- | --- |
| `configs/server_w31.json` | 固定服务配置，目标版本 vLLM 0.14.1 |
| `configs/workloads_w31.jsonl` | 三类中文请求：短输入短输出、较长输入短输出、短输入较长输出 |
| `scripts/serve.py` | 校验配置、记录启动信息，直接启动 vLLM 官方 API server |
| `scripts/model_config.py` | 本项目的本地模型别名、路径、精度与量化配置校验 |
| `scripts/http_client.py` | 本项目的 HTTP、认证、JSON 与 SSE 协议处理 |
| `scripts/common.py` | 请求参数、客户端测量、运行记录与汇总口径 |
| `scripts/client.py` | 单次非流式／流式请求与结果落盘 |
| `scripts/benchmark.py` | 预热、并发与多轮基线；保存请求记录和计时窗口 |
| `scripts/summarize.py` | 从原始记录重算 JSON／CSV 汇总 |
| `tests/` | 无 GPU、无 vLLM 的模拟 HTTP／SSE 检查 |
| `runs/` | 本地运行产物，已忽略，不作为已测结果提交 |

本项目的启动器、模型配置、HTTP／SSE 客户端、压测与汇总代码全部位于 `runtime_training`，可整体复制到其他位置使用。客户端和压测只依赖 Python 标准库；GPU 服务依赖本目录的 [requirements-server.txt](requirements-server.txt)，锁定 **vLLM 0.14.1**。启动器直接运行 `python -m vllm.entrypoints.openai.api_server`，使用 vLLM 官方响应协议；性能记录全部采用客户端近似口径。模型权重通过 `MODEL_ROOT` 或本地绝对路径提供。

## 先在 Windows 检查代码

在 PowerShell 中设置项目位置。使用脚本绝对路径时可以从任意工作目录执行；默认配置／负载／输出位置以 P3 目录为基准；显式传入的相对路径仍以当前工作目录为基准，因此下列命令使用绝对路径。

```powershell
$p3 = 'D:\workspace\learing\AI_system\capstone\runtime_training'
python "$p3\scripts\serve.py" --config "$p3\configs\server_w31.json" --dry-run
python "$p3\scripts\benchmark.py" --workloads "$p3\configs\workloads_w31.jsonl" --output-dir "$p3\runs\w31_plan" --dry-run
python -m unittest discover -s "$p3\tests" -v
```

`serve.py --dry-run` 只校验配置并输出实际启动命令，不加载 GPU 模型；`benchmark.py --dry-run` 保存负载与实验计划，不连接服务。模拟 API 测试通过也不表示 Qwen3-8B 已运行。当前本机 Python 为 3.13.5，未安装 vLLM；真实 serving 请使用安装本项目服务依赖且满足显存要求的 Linux GPU 环境。

## 在 Linux GPU 环境安装并启动服务

以下先在 GPU 机器的 `runtime_training` 目录设置 `P3`；之后可以切换工作目录。该目录可单独复制到 GPU 机器，不要求保留仓库层级。模型根目录默认 `/workspace/weights_pub`，如模型位于别处，按真实位置设置 `MODEL_ROOT`。

```bash
export P3="$PWD"
export MODEL_ROOT=/workspace/weights_pub
python -m pip install -r "$P3/requirements-server.txt"
python -c "import torch, vllm; print('PyTorch:', torch.__version__); print('vLLM:', vllm.__version__)"
python "$P3/scripts/serve.py" --config "$P3/configs/server_w31.json" --dry-run
python "$P3/scripts/serve.py" --config "$P3/configs/server_w31.json"
```

服务器默认监听 `0.0.0.0:7081`，Windows 客户端默认访问 `http://192.168.88.91:7081`，API 模型名称为 `qwen3-8b`；模型别名通过本项目 `scripts/model_config.py` 查找本地权重。基线固定 `max_model_len=2048`、`max_num_seqs=4`、eager 开、prefix caching 关、chunked prefill 开。以 `--dry-run` 的实际命令和启动日志为准，并把实际 attention backend 填入报告。


配置修改后需要重启正在运行的服务才能生效。如果旧进程仍在运行，先在它的终端按 `Ctrl+C` 停止，再执行上面的正式启动命令，核对日志中的监听地址和端口。

若服务运行在容器内，容器还需将端口发布到宿主机，例如在原有 `docker run` 命令中加入 `-p 7081:7081`。`192.168.88.91` 应为 Windows 可达的服务宿主机地址；端口映射、路由和防火墙问题按下文步骤排查。
如果 BF16 无法稳定容纳此负载，可在本项目中核验本地 AWQ W4A16 权重后，改用对应别名并重新记录完整基线：

```bash
python "$P3/scripts/serve.py" --config "$P3/configs/server_w31.json" --model qwen3-8b-awq
```

本周服务支持本地 BF16／FP16 未量化权重与 AWQ INT4 权重；AWQ 的自动选择使用 `awq_marlin`，实际 backend 与执行精度仍需 GPU 日志验证。权重别名与 API 名称是不同概念；客户端继续使用配置中的 `served_model_name`。若改服务配置，另存配置并记录差异，避免把 BF16 与量化结果放进同一组。

## 在 Windows 访问远程 GPU 服务

Windows 只运行标准库客户端，不需要安装 `torch` 或 `vllm`。先确认 Linux 服务已按上述配置重启，再在 PowerShell 检查 TCP、HTTP 和模型名称。

```powershell
$p3 = 'D:\workspace\learing\AI_system\capstone\runtime_training'
$apiBase = 'http://192.168.88.91:7081'
Test-NetConnection 192.168.88.91 -Port 7081
$apiHeaders = @{}
if ($env:VLLM_API_KEY) { $apiHeaders['Authorization'] = "Bearer $env:VLLM_API_KEY" }
Invoke-RestMethod -Uri "$apiBase/health" -Headers $apiHeaders
Invoke-RestMethod -Uri "$apiBase/v1/models" -Headers $apiHeaders
python "$p3\scripts\client.py" --base-url $apiBase --model qwen3-8b --stream --no-proxy --thinking off --max-tokens 128 --prompt '请用三句话解释 prefill 和 decode。' --output-dir "$p3\runs\w31_remote_stream"
python "$p3\scripts\benchmark.py" --base-url $apiBase --model qwen3-8b --workloads "$p3\configs\workloads_w31.jsonl" --concurrency 1 2 --requests-per-workload 2 --warmup 1 --repeats 1 --no-proxy --output-dir "$p3\runs\w31_remote_smoke"
```

客户端默认地址已经是 `http://192.168.88.91:7081`；这里显式指定便于核对。Python 命令通过 `--no-proxy` 直连局域网地址；需要认证时自动读取现有 `VLLM_API_KEY` 环境变量，也可用 `--api-key-env` 指定其他变量名。

`TcpTestSucceeded` 为 false 时，先检查服务是否启动并监听 `0.0.0.0:7081`、容器是否发布端口、两台机器的路由及端口防火墙规则。TCP 成功但 HTTP 失败时检查服务日志、URL 与认证；`/v1/models` 成功后再检查模型别名和生成响应。以上步骤用于定位问题，不会修改路由或防火墙。单元测试中的本地模拟 HTTP／SSE 服务只用于代码检查，不代表此远程部署已经连通。

## 单请求、流式与压测

在另一个终端设置相同 `P3` 后执行。客户端默认发送一条非流式请求；`--stream` 时即时输出 SSE 文本增量。

```bash
python "$P3/scripts/client.py" --base-url http://192.168.88.91:7081 --model qwen3-8b --thinking off --max-tokens 128 --prompt "请用三句话解释 prefill 和 decode。" --output-dir "$P3/runs/w31_client"
python "$P3/scripts/client.py" --base-url http://192.168.88.91:7081 --model qwen3-8b --thinking off --max-tokens 128 --prompt "请用三句话解释 KV cache。" --stream --output-dir "$P3/runs/w31_stream"
python "$P3/scripts/benchmark.py" --base-url http://192.168.88.91:7081 --model qwen3-8b --workloads "$P3/configs/workloads_w31.jsonl" --concurrency 1 4 --requests-per-workload 20 --warmup 2 --repeats 3 --output-dir "$P3/runs/w31"
```

先完成单请求，再跑小批量冒烟（如 `--concurrency 1 2 --requests-per-workload 2 --warmup 1 --repeats 1`），最后跑正式基线。正式压测为每类请求分别测试并发 1／4，每组预热后测量三轮，**没有把三类负载混在一个调度窗口中**；长短混合负载留到 W32。

负载文件中的名称描述相对长度；它们不是预先精确校准的 128／1024／4096 模型 token 输入。`max_tokens` 是输出上限，模型可能提前停止。实际输入／输出 token 数以服务返回的 `usage` 为准；缺失就保留缺失，不能用字符数补成 token 数。

每次客户端／压测在 `--output-dir` 下创建 UTC 时间戳与随机 ID 子目录，避免覆盖；`manifest.json` 保存本次参数与状态，压测还保存负载及客户端环境；压测的 `warmup.jsonl` 单独保存预热。终端会打印本次实际运行目录。使用该目录重算汇总（把占位路径换成终端输出）：

```bash
python "$P3/scripts/summarize.py" "/absolute/path/to/the/run-directory"
```

压测保留 `metrics.jsonl`（逐请求）、`cases.jsonl`（每类负载 × 并发 × 重复轮次的墙钟窗口）、`summary.json` 和 `summary.csv`。单请求默认保存前三项，可用 `summarize.py` 追加 CSV。汇总按 case 分开，不把不同负载、并发或重复轮次的吞吐直接相加。


服务端生成的 `manifest.json` 与 `startup.log` 保留完整启动计划及日志。压测可追加 `--server-manifest "/absolute/path/to/server-run/manifest.json"`，将服务配置复制到本次记录作为来源说明；客户端不会据此证明当前连接的服务就是该 manifest 对应实例，仍应核对实际服务地址、日志和进程。服务端 `host=0.0.0.0` 表示监听本机所有网络接口；客户端连接目标使用 `192.168.88.91`，不能把 `0.0.0.0` 作为目的地址。Linux 服务所在主机也可显式传 `--base-url http://127.0.0.1:7081` 做本机检查；该 loopback 地址只指向执行客户端的那台机器。

## 指标必须这样读

| 指标 | 当前客户端口径 |
| --- | --- |
| 请求总耗时 | 请求开始至完整响应结束的客户端墙钟时间 |
| TTFT | 请求开始至首个非空文本 SSE 事件；是客户端可见首输出近似值 |
| TPOT | 依据首／末文本事件时间和实际输出 token 数估算；仅有一个输出 token 或无可靠计数时留空 |
| p50／p95 | 各指标分别汇总并给出有效样本数；默认只统计成功请求中的有效值 |
| 输出吞吐 | 每个测量窗口成功输出 token 数／该窗口墙钟时间；分母含失败请求花费，不含预热 |
| 错误率 | 失败请求数／全部测量请求数，保留失败信息 |
| `finish_reason` | `stop` 表示停止条件命中，`length` 表示达到长度限制；仅凭 `stop` 不能断言命中了哪个 EOS token |

一个 SSE 事件可能包含多个 token；文本事件近似 TTFT／TPOT 不能当作 GPU kernel 时间或严格的逐 token ITL。非流式请求无法观测首个生成 token 时间。更完整的口径和边界见学习指南。

## W31 验收

- [ ] 环境、模型、revision／配置、精度、模板和完整启动命令已记录。
- [ ] 非流式与流式均可成功调用，`finish_reason` 和 `usage` 已保存。
- [ ] 能解释 API server、engine、scheduler、KV manager、model runner、attention backend 的职责和数据流。
- [ ] 三类请求的并发 1／4 基线已完成预热与三轮测量，错误未被隐藏。
- [ ] TTFT、TPOT、总耗时各自的 p50／p95 与样本数已核对，缺失值有说明。
- [ ] 报告中填入真实结果和原始运行目录，保留可复现配置。

## P3 后续任务

来源：`00.AI-system-plan-optimized-v3.xlsx`／`案例3_Runtime与训练闭环`。W31 的代码准备与 GPU 验收分开记录；后续计划沿用原表。

| 任务ID | 周次 | 实施任务 | 具体步骤 | 交付物 | 验收标准 | 资源与依赖 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P3-01 | W31 | API与流式客户端 | 部署Qwen3-8B，固定采样/长度；记录请求到首token及完成时间 | server/client、metrics.jsonl | 可流式生成；EOS与长度截断记录明确 | 本地BF16或W4A16权重 | 代码已准备；GPU待实测 |
| P3-02 | W32 | 调度消融 | batch/KV预算/chunked prefill/prefix cache/Graph每次改一个 | 参数矩阵、原始CSV | 展示吞吐与p95 TTFT/TPOT变化 | 先并发1/8/16；32按显存扩展 | 未开始 |
| P3-03 | W35 | SGLang对照 | 同模型同精度同请求顺序；重复前缀与随机前缀；冷/热缓存分开 | 双runtime报告 | 质量、TTFT、TPOT、吞吐口径一致 | 不预设某框架更快 | 未开始 |
| P3-04 | W36 | 推理项目交付 | 核验实际attention backend；连接P2测量结果解释服务瓶颈 | Project3推理报告、演示 | 15分钟讲清API→调度→KV→kernel；开始持续投递 | FlashInfer是kernel库，不等于完整runtime | 未开始 |
| P3-05 | W39–W40 | SFT到Serving | 固定小数据集LoRA/QLoRA；保存/恢复adapter；Liger A/B；部署适配后模型 | 训练配置、checkpoint、服务回归 | 至少100 optimizer steps；留出质量回归和恢复测试 | 微调不替代预训练；无需8B全参 | 未开始 |
| P3-06 | W41 | DDP/NCCL | 小模型2卡；对齐有效batch；测通信与step时间 | collective/训练trace | 解释AllReduce/AllGather/ReduceScatter用途 | 无多卡则标待验证 | 未开始 |
| P3-07 | W43 | MCore单卡闭环 | 用官方组件搭小GPT；保存并恢复optimizer/checkpoint；记录格式 | MCore训练/恢复日志 | 100 step、loss有限、恢复状态正确 | 先小模型；不是Qwen3权重转换已完成 | 未开始 |
| P3-08 | W44 | MCore多卡扩展 | 同型号2卡TP或DP；固定global batch与序列；可选Qwen3导入 | 多卡报告、兼容性记录 | 真实多卡step/通信数据；说明Qwen3导入是否验证 | 2卡不能完成所有TP/PP/DP组合；EP仅概念 | 未开始 |
| P3-09 | W49 | 稳定性 | 30分钟混合请求压测；测试取消/超时/OOM后的恢复 | 运行手册、监控样例 | 错误率、重启恢复、内存增长有记录 | 不把本地实验声明为生产规模 | 未开始 |
