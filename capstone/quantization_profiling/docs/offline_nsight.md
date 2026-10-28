# 非 server 的 vLLM NSYS / NCU 测试方案

**完整参数说明：** [NSYS/NCU 参数、kernel 名称/正则、阶段窗口与粒度](offline_nsight_parameters.md)。推荐使用命名参数；原位置参数命令继续支持。

本方案直接调用 LLM.generate，不启动 OpenAI API server，不经过 benchmark.py、HTTP 客户端、端口和 middleware。NSYS 与 NCU 使用同一个工作负载入口、冻结请求、模型配置和 ProfilingWorker。适用原 Linux GPU 环境：vLLM 0.14.1；当前 Windows 主机只做源码、CPU 测试和报告查看。

## 1. 文件与执行路径

| 文件 | 用途 |
|---|---|
| scripts/profile_kernels.py | 共用离线入口：校验快照、初始化 LLM、5 次预热、正式 generate、记录配置和耗时 |
| scripts/profile_offline_nsys.sh | NSYS 时间线；支持四模型、eager/Graph、稳态/完整启动窗口 |
| scripts/profile_offline_ncu.sh | NCU 核心 kernel；支持四模型的矩阵热点及 FP8 量化热点 |
| scripts/check_nsys_capture.py | 导出 SQLite 的基本 GPU/NVTX/Graph 覆盖检查 |

~~~text
NSYS ─┐
      ├─ profile_kernels.py → LLM.generate → vLLM Engine → GPU Worker → GPU
NCU ──┘

两个工具分别运行，不嵌套、不同时采同一进程。
~~~

LLM.generate 仍会使用 vLLM 的模型加载、显存探测、KV cache、调度、编译/Graph、prefill、decode 和 sampling；也仍可能启动 worker 子进程。它不是绕开框架的单 GEMM 测试。原 server benchmark 可以继续用于服务层实验，两类结果分开记录。

## 2. 在原 GPU 主机准备环境

以下命令在同一个 Linux Bash 终端运行。路径不同则修改对应变量，沿用原环境，不顺便更换 torch/vLLM。

~~~bash
cd /workspace/AI_system
source /workspace/venvs/vllm-0112/bin/activate

export MODEL_ROOT=/workspace/weights_pub
export REQUESTS=/workspace/AI_system/capstone/quantization_profiling/data/frozen/requests.jsonl
export NSYS_OUT=/workspace/AI_system/capstone/quantization_profiling/log/nsys_offline
export NCU_OUT=/workspace/AI_system/capstone/quantization_profiling/log/ncu_offline

# 两个 wrapper 共用的工作负载设置。
export PROFILE_CASE=w128_o128
export PROFILE_BATCH_SIZE=1
export PROFILE_BATCH_TOKENS=2048
export PROFILE_COUNT=1

mkdir -p "$NSYS_OUT" "$NCU_OUT"
nvidia-smi
nsys --version
ncu --version
python -m pip show vllm torch triton nvtx
~~~

worker 阶段标记需要 nvtx 包；若原环境尚未安装，可执行：

~~~bash
python -m pip install -r capstone/quantization_profiling/requirements-profiling.txt
~~~

权重目录应有 Qwen3-8B、Qwen3-8B-AWQ、Qwen3-8B-FP8、Qwen3-8B-FP8-local 中所选模型。只比较 BF16 和本地 FP8 时不需要加载另外两个。

REQUESTS 必须是冻结快照中的 requests.jsonl，同目录应保留 manifest.json、model.json、workloads.yaml 及 manifest 所列文件。共用入口调用与 benchmark 相同的 load_snapshot 验证文件哈希，并从冻结 model.json 取得参考 tokenizer 路径。若权重搬迁导致参考 tokenizer 路径失效，直接运行 Python 入口时可显式传入 --tokenizer；应记录该覆盖，不手改冻结文件。

可先检查配置计划，不导入 vLLM、不加载权重到 GPU：

~~~bash
python capstone/quantization_profiling/scripts/profile_kernels.py \
  --profile fp8-local --requests-file "$REQUESTS" --case "$PROFILE_CASE" \
  --graph --nvtx --plan-only --output "$NSYS_OUT/plan_unused.workload.json"
~~~

plan-only 将计划打印到 stdout，不创建 output 文件。它仍需能读取模型 config.json。程序验证模型格式、快照哈希和长度，并记录实际模型配置 SHA256；不会重新扫描全部模型权重 SHA256，也不宣称模型质量或权重来源已重新验收。

## 3. NSYS：预热后的完整请求时间线

先比较 FP8-local 与 BF16 的 Graph 执行：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh fp8-local graph steady
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh bf16 graph steady
~~~

比较 eager：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh fp8-local eager steady
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh bf16 eager steady
~~~

四个模型 × 两种执行模式，串行生成 8 份报告：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh all all steady
~~~

每组都会重新初始化自己的模型，完成 5 次完整预热，再采正式请求。默认一个正式 generate、batch=1、128 输入/128 输出，包含 prefill 和后续 decode。多份报告串行采集，避免互相抢 GPU 资源。

steady 的控制顺序为：

~~~text
导入 / 加载 / profile_memory / KV cache / 编译与 Graph 捕获
    ↓
5 次完整预热
    ↓
LLM.start_profile：在真正的 CUDA worker 内触发 ProfilerStart
    ↓
offline/measure：正式 LLM.generate
    ↓
关闭 offline/measure 范围
    ↓
LLM.stop_profile：worker 触发 ProfilerStop
~~~

此时 NSYS 使用 cudaProfilerApi 作为 capture range；初始化和预热仍实际执行，只是不写入这份稳态时间线。NSYS 的 delay/max iteration 均为 0，不会只采两个 decode step。

## 4. NSYS：加载、显存探测和 Graph 捕获也一起采

第三个参数改为 all：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh fp8-local graph all
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh bf16 eager all

# 需要完整的 8 组启动 + 推理报告时：
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh all all all
~~~

完整窗口不启用 CUDA profiler gate，由 NSYS 从程序启动开始记录，包括初始化、5 次预热和正式请求。不要把全报告 kernel Summary 当成正式请求的 kernel 耗时，必须选取 offline/measure 中的工作。

worker 的 NVTX 标记包括：

~~~text
worker/import_gpu_worker
worker/load_model
worker/profile_memory
worker/initialize_kv_cache
worker/compile_and_graph_capture
worker/prefill
worker/decode
worker/sampling
~~~

控制进程另有 offline/initialize_engine、offline/warmup、offline/measure、offline/generate。导入 GPU Worker 的标记只涵盖扩展模块内可观察的导入，不代表整个 Python/vLLM 导入时间。

完整和稳态报告可有不同采集扰动，不能直接把两者耗时差视为加载成本；加载成本应在完整报告里按对应阶段解释。

### NSYS 产物与覆盖检查

每次输出的唯一前缀包括模型、eager/graph、steady/all、case、UTC 时间和 PID：

~~~text
<prefix>.nsys-rep       原始 NSYS 报告
<prefix>.sqlite         自动导出的 SQLite
<prefix>.coverage.json  GPU、worker NVTX、测量范围及 Graph 基本覆盖检查
<prefix>.workload.json  共用入口的完整配置、请求/快照哈希、版本、每次 generate 耗时
<prefix>.events/        controller 和 worker 的阶段 JSONL
<prefix>.log            程序与采集器日志
<prefix>.settings.txt   本次 wrapper 设置
<prefix>.version.txt    NSYS 版本
~~~

稳态报告不要求出现 worker/load_model，因为该阶段在采集前已结束。两种窗口都会要求测量范围内存在 GPU 执行；Graph 模式还要求测量范围内同时有 GraphLaunch 和设备执行证据。发现导出数据中的 CUDA event-loss 告警会返回非零退出码。

这是基本覆盖检查，不能证明每一步尾部都完整；仍需核对采集日志、各步 CUDA API 和关联 GPU 活动。更换成离线入口也不会自动修复驱动/采集器不匹配或 CUDA 事件丢失。

## 5. NCU：使用相同非 server 工作负载采核心 kernel

先设置与 NSYS graph 一致的模式：

~~~bash
export NCU_GRAPH=1
export NCU_CLOCK_CONTROL=none
export NCU_CACHE_CONTROL=all

# 每份先看第一层的四个矩阵 launch，核对映射。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode first4 basic
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode first4 basic

# 如需另外两种模型：
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-official decode first4 basic
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh awq decode first4 basic
~~~

当前 kernel 家族根据原始报告选择：两种 FP8 使用 _w8a8_triton_block_scaled_mm；BF16 decode 使用 internal::gemvx::kernel，prefill 使用 CUTLASS BF16 GEMM；AWQ 使用 marlin::Marlin。离线入口必须重新核对这些实现是否仍被选中，不能因名字相同就直接认定所有配置都相同。

重点比较 down、o 和 gate_up：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode down detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode down detail

bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode o detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode o detail

bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode gate_up detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 decode gate_up detail
~~~

量化与 prefill 对照：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode quant detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode norm_quant detail

bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local prefill gate_up detail
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh bf16 prefill gate_up detail
~~~

切换 eager 时，两种模型都设置 NCU_GRAPH=0，再执行相同命令。脚本不只关闭 Graph 回放：该选项对应 enforce_eager=true，也会影响编译路径，因此两种模式分开解释。

NCU 同样先执行 5 次预热。decode 使用 worker delay=8、max=2，从正式请求较后的 decode 中选少量 kernel；在 v0.14.1 的实现中第 8 次模型迭代前开始，第 10 次前停止。prefill 使用 delay=0、max=1。它们是模型迭代边界，不是直接传入的 token 编号。

first4 采 4 个匹配 launch；qkv/o/gate_up/down 分别跳过 0/1/2/3 个匹配矩阵 kernel，再采 1 个。该映射限定原 w128_o128、batch=1；脚本拒绝对其他 case 或 batch 直接套用这些投影位置。其他 shape 应先用 NSYS 重新定位，改用 --kernel-name 或 --kernel-regex，并根据实际调度调整 delay/max 窗口。

basic 采 LaunchStats、SpeedOfLight、Occupancy；detail 再采 MemoryWorkloadAnalysis、ComputeWorkloadAnalysis、SchedulerStats、WarpStateStats、SourceCounters。使用 kernel replay 与 graph-profiling node，生成 ncu-rep、metrics.csv、details.txt、workload.json、events、日志和版本记录。

具体硬件指标判读和原 FP8/BF16 差异见 [投影归因与 NCU 分析文档](fp8_local_bf16_ncu_20261028.md)。


### 自定义 kernel 名称或正则

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model fp8-local --phase decode --level detail \
  --kernel-name '_w8a8_triton_block_scaled_mm' \
  --launch-skip 3 --launch-count 1

bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model bf16 --phase prefill --level basic \
  --kernel-regex 'cutlass::Kernel2<cutlass_80_tensorop_bf16_' \
  --launch-count 1
~~~

--kernel-name 为精确匹配，--kernel-regex 为部分正则匹配，二者与 --target 互斥。
--kernel-name-base 默认 demangled，可选 function/mangled。
prefill/decode 现在还会增加真实 worker 的对应 NVTX 阶段过滤；
basic 仍为 3 个 section，detail 仍为 8 个。
launch-skip/count 只统计匹配阶段与 kernel 的调用，不是 token 数。
完整默认值、窗口覆盖和其他 case 的用法见上面的参数说明。

## 6. 无 profiler 的非 server 测试

NSYS/NCU 命令运行时间包含采集开销。关闭 profiler 和 NVTX，单独记录 LLM.generate 的离线 wall time：

~~~bash
BASELINE_OUT=/workspace/AI_system/capstone/quantization_profiling/log/offline_baseline
mkdir -p "$BASELINE_OUT"

for model in bf16 fp8-local; do
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  python capstone/quantization_profiling/scripts/profile_kernels.py \
    --profile "$model" --requests-file "$REQUESTS" --case w128_o128 \
    --count 5 --batch-size 1 --warmup 5 --batch-tokens 2048 \
    --profiler none --no-nvtx --graph \
    --output "$BASELINE_OUT/$model-$stamp.json"
done
~~~

去掉 --graph 即为 eager。输出 records 保存逐次 generate 耗时；summary.output_tokens_per_second 为生成 token 数除以各次 generate wall time 的和，排除初始化和预热。它包含引擎调度、模型计算及结果处理；不是 HTTP 吞吐、TTFT 或逐 token TPOT。batch-size 大于 1 时是一次 generate 提交一个 batch，也不等于 HTTP 请求并发到达。

采集模式下同样会生成 summary，但数值受 profiler 影响，只用于诊断记录，不能与无 profiler 的结果混作性能结论。

## 7. 如何保证 NSYS 与 NCU 的可比性

两个 wrapper 现在都显式开启 --nvtx，使用同一个 ProfilingWorker，不再一边扩展 worker、一边使用默认 worker。共同读取快照与模型配置，并记录 engine 参数、请求/配置哈希、软件版本和影响执行的环境变量。

比较前应核对两份 workload.json：

- profile、模型路径与模型配置 SHA256；
- manifest/request SHA256、输入 token 数、输出长度和请求 ID；
- dtype、量化 loader、KV cache dtype、TP、batch size、batch token budget；
- enforce_eager、worker_cls、包版本和后端环境；
- kernel 名称、M/N/K、grid/block、registers/shared，以及所选 layer/token 位置。

两个工具的 profiler_config gate 不同，这是采集窗口差异；其余引擎配置应对齐。NCU 单 kernel replay 还会改变缓存/并发/时钟条件，因此不应要求 NCU duration 与 NSYS 原时间线完全相同，也不能将 NCU 的进程总时长作为模型延迟。

## 8. 本地验证与目标机验收

已完成 CPU 侧的快照校验、四模型参数、参考 tokenizer、预热/测量顺序、异常停止 profiler、worker 标记回归、SQLite 覆盖检查、Bash 语法及模拟 profiler 命令测试。测试不启动 CUDA，不产生真实 GPU 报告。

~~~bash
python -m unittest discover -s capstone/quantization_profiling/tests -p test_profile_kernels.py
python -m unittest discover -s capstone/quantization_profiling/tests -p test_nsys_capture.py
python -m unittest discover -s capstone/quantization_profiling/tests -p 'test_profiling*.py'
python -m unittest discover -s capstone/quantization_profiling/tests -p test_offline_profile_cli.py
~~~

当前可访问机器为 Windows RTX 5060，缺原 vLLM/Triton 环境。目标 RTX 4090D 的非 server NSYS/NCU 真实采集仍需运行上述命令；这里没有把 mock 检查写成 GPU 实测成功。

官方接口依据：[vLLM 0.14.1 LLM](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/entrypoints/llm.py)、[worker profiler gate](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/profiler/wrapper.py)、[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。
