# Nsight 快速采集：eager、no_eager、阶段日志与 NVTX

适用：Linux Bash、RTX 4090D、项目固定的 vLLM 0.14.1。更新：2026-10-26。

先采时间线，再按需增加CPU栈或显存跟踪，最后用NCU研究单个热点。下面命令均调用现有benchmark，保留5次预热、1次正式请求、1轮。Profiler和阶段日志会增加开销，采集结果用于诊断，不代替无profiler的正式性能结果。

新增 `benchmark.py --nvtx` 会同时标记控制进程和实际GPU worker；默认关闭，不在父进程导入torch/CUDA。worker扩展按vLLM 0.14.1接口实现，其他版本不能直接套用。

## 1. 一次性准备：在同一个 Bash 终端执行

```bash
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

# 清除旧教程/排查命令遗留的覆盖，后续新进程使用默认临时目录。
unset NSYS_TMPDIR TMPDIR VLLM_RPC_BASE_PATH

# 只安装轻量NVTX Python注解依赖，不更换torch/vLLM。
python -m pip install -r capstone/quantization_profiling/requirements-profiling.txt
python -c 'import nvtx; import importlib.metadata as m; print("nvtx",m.version("nvtx"),"vllm",m.version("vllm"))'
nsys --version
nsys status --environment
python -c 'import tempfile; print("Python/vLLM默认临时目录:", tempfile.gettempdir())'

OUT="/workspace/AI_system/capstone/quantization_profiling/log/profiling"
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
CASE_ID=w128_o128
MODEL_PROFILE=fp8-local
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
# 默认只打印prefill、阶段转换和每32步decode；NVTX仍标记每一步。
export QUANT_PROFILE_LOG_EVERY_N_STEPS=32
mkdir -p "$OUT"

wait_7081() {
  local attempt
  for attempt in {1..45}; do
    if python -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0",7081)); s.close()' 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "7081仍不可绑定，请检查旧服务及TCP状态。" >&2
  ss -tanp '( sport = :7081 )'
  return 1
}

NSYS_TIMELINE_ARGS=(
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --cuda-event-trace=false
  --cuda-graph-trace=node
  --cuda-flush-interval=1000
  --trace-fork-before-exec=false
  --wait=all
)
NSYS_ARGS=("${NSYS_TIMELINE_ARGS[@]}")
```

这是本项目当前推荐的基础采集配置：只指定报告输出目录OUT，临时文件沿用系统默认位置，Linux通常为 `/tmp`；不额外设置NSYS_TMPDIR、TMPDIR或VLLM_RPC_BASE_PATH。上面的unset用于清除此前命令留下的覆盖；以Python打印的实际临时目录和nsys日志中的Generating路径为准。它只影响当前终端及随后启动的子进程。

使用项目的默认冻结快照 `data/frozen`；快照必须已经存在，1个正式请求仍执行冻结的5次预热。

- `cuda`：CUDA API、kernel、memcpy及Graph活动。
- `nvtx`：下面新增的加载、prefill、decode和请求范围。
- `osrt`：受支持的OS运行库调用，例如锁、等待、I/O。它不等于CPU采样或线程调度轨迹。
- `sample=none`：关闭CPU指令/调用栈采样；`cpuctxsw=none`：关闭CPU线程调度跟踪。先获取较低开销的CUDA/NVTX/OSRT时间线。
- `cuda-graph-trace=node`：允许看到Graph内的节点kernel；开销高于graph整图模式，适合热点分析。eager也显式使用同一采集选项以方便配对。
- `cuda-event-trace=false`：关闭额外CUDA event completion跟踪，减少扰动；不会关闭kernel或Graph活动追踪。
- `cuda-flush-interval=1000`：每秒刷新CUDA缓冲，帮助减少异常退出时的数据损失；有额外开销，不能保证最后尾部数据完整。
- `trace-fork-before-exec=false`：当前使用spawn，先关闭fork到exec之间的特殊追踪；不是关闭正常exec后子进程追踪。
- `wait=all`：等待被重新托管的子进程。benchmark也会给整个本次服务进程组保留15秒退出宽限期，不再把包装进程退出等同于worker已退出。

若采集器仍提示CUDA 13.0驱动使用12.9追踪库，先核对支持目标驱动的Nsight安装。修改采集参数或升级Windows查看器不能补回旧文件缺失的活动。

## 2. 定义统一采集函数：保存日志并检查数据覆盖

继续在同一终端执行。这个函数不仅检查nsys-rep文件存在，还会检查实际GPU活动、worker NVTX和正式请求范围。

```bash
collect_one() {
  local mode_label="$1" eager_value="$2" model_profile="$3"
  shift 3
  local prefix="$OUT/${mode_label}_${model_profile}_${CASE_ID}_${RUN_TAG}"
  local nsys_rc audit_rc
  local -a audit_args=(--require-worker-ranges) pipeline_status

  wait_7081 || return 1
  if nsys profile "${NSYS_ARGS[@]}" -o "$prefix" \
      python capstone/quantization_profiling/scripts/benchmark.py \
        --model "$model_profile" --case "$CASE_ID" \
        --enforce-eager "$eager_value" --nvtx \
        --requests 1 --rounds 1 "$@" 2>&1 | tee "${prefix}.nsys.log"; then
    pipeline_status=("${PIPESTATUS[@]}")
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi

  nsys_rc=${pipeline_status[0]}
  if [ "$nsys_rc" -ne 0 ] || [ "${pipeline_status[1]}" -ne 0 ] || [ ! -s "${prefix}.nsys-rep" ]; then
    echo "本次运行失败：nsys退出码=$nsys_rc，tee退出码=${pipeline_status[1]}；检查${prefix}.nsys.log和server.log" >&2
    find "$OUT" /tmp -maxdepth 4 -type f -name '*.qdstrm' -ls 2>/dev/null || true
    return 1
  fi

  nsys export --quiet=true --type=sqlite \
    --output="${prefix}.sqlite" "${prefix}.nsys-rep" || return 1
  if [ "$eager_value" = false ]; then
    audit_args+=(--expect-graph)
  fi
  if python capstone/quantization_profiling/scripts/check_nsys_capture.py \
      "${prefix}.sqlite" "${audit_args[@]}" | tee "${prefix}.audit.json"; then
    pipeline_status=("${PIPESTATUS[@]}")
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi
  audit_rc=${pipeline_status[0]}
  [ "${pipeline_status[1]}" -eq 0 ] || return 1
  return "$audit_rc"
}
```

PIPESTATUS必须在管道后立即读取，避免把tee成功当成nsys成功。同一RUN_TAG重复同一模式/模型会重名；重采前重新设置RUN_TAG即可，无需创建或重新指定临时目录，不覆盖有用的原始文件。

nsys-rep文件生成成功不代表benchmark成功；服务启动失败时，报告可能只记录失败的启动过程。本函数遇到非零退出码会停止后续SQLite导出和覆盖检查，应先排查server.log。

覆盖检查退出0只表示关键数据存在，仍要看Diagnostics Summary中的丢事件警告。no_eager的Graph检查失败也可能表示框架实际没有使用Graph，应结合server.log判断，不能只归因于采集器。

## 3. eager 与 no_eager 两套命令

### 3.1 单模型 eager

```bash
MODEL_PROFILE=fp8-local
collect_one eager true "$MODEL_PROFILE"
```

传给benchmark的是 `--enforce-eager true --nvtx`。输出前缀为 `eager_fp8-local_<CASE_ID>_<RUN_TAG>`。

### 3.2 同一模型 no_eager

```bash
MODEL_PROFILE=fp8-local
collect_one no_eager false "$MODEL_PROFILE"
```

传给benchmark的是 `--enforce-eager false --nvtx`，允许框架使用CUDA Graph。必须看到实际回放记录才能确认，而不是只看Graph Creation。

要采官方模型，改成 `MODEL_PROFILE=fp8-official`；`fp8-origin`是其benchmark别名。BF16/AWQ分别用bf16/awq。函数支持追加benchmark参数，例如 `collect_one eager_marlin true fp8-official --fp8-backend marlin`；当前compressed-tensors本地FP8_BLOCK加载路径不接受此请求。

### 3.3 四模型完整配对

```bash
for MODEL_PROFILE in bf16 awq fp8-official fp8-local; do
  collect_one eager true "$MODEL_PROFILE" || break
  collect_one no_eager false "$MODEL_PROFILE" || break
done
```

失败会停止后续实验，保留已有文件。只采本地/官方FP8时，将列表改成 `fp8-local fp8-official`。

改变负载只需先设置 `CASE_ID=w1024_o128`、`w4096_o256`、`w8192_o128`或`w1024_o512`，文件名和命令会同步变化。

### 3.4 在原终端重跑一次

已定义第2节collect_one后，重跑时使用新的RUN_TAG。若原终端执行过旧文档或临时目录排查命令，先清除残留覆盖：

```bash
unset NSYS_TMPDIR TMPDIR VLLM_RPC_BASE_PATH
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
MODEL_PROFILE=fp8-local

collect_one eager true "$MODEL_PROFILE" && \
  collect_one no_eager false "$MODEL_PROFILE"
```

第一份采集与覆盖检查通过后才开始第二份；两份报告共用该RUN_TAG，前缀区分模式。eager在本版本日志中还会将优化等级覆盖为-O0，比较结果不能全部归因于CUDA Graph。

## 4. 日志和NVTX里新增了什么

| 范围/日志 | 所在进程 | 含义 |
| --- | --- | --- |
| controller/verify_weights | benchmark | 哈希/身份验证，不是GPU模型加载 |
| controller/server_startup | benchmark | 启动子进程到服务健康和引擎配置核验 |
| worker/load_model | GPU worker | 实际模型构造、权重加载和该方法内初始化 |
| worker/profile_memory | GPU worker | 初始化显存探测/dummy profile |
| worker/initialize_kv_cache | GPU worker | KV cache初始化 |
| worker/compile_and_graph_capture | GPU worker | 编译、内部warmup、Graph捕获；不等于正式decode |
| controller/warmup、controller/measure | benchmark | HTTP预热/正式阶段，包含该阶段全部请求 |
| client/request | 客户端线程 | 一个HTTP请求的外层范围 |
| client/wait_first_text | 客户端线程 | 等待首个文本，包含网络/排队，不是纯GPU prefill |
| client/first_text_to_stream_end | 客户端线程 | 首文本到SSE完成，包含响应尾部，不是纯GPU decode |
| worker/prefill | GPU worker | 当前调度步仍处理prompt token |
| worker/decode | GPU worker | 当前调度步处理prompt之后的生成token |
| worker/mixed | GPU worker | 同一调度步包含prefill和decode |
| worker/unknown、worker/idle | GPU worker | 元数据不足或没有调度token，不猜测阶段 |
| worker/sampling | GPU worker | 独立的token sampling调用，不混作forward |
| controller/shutdown | benchmark | 终止本次服务进程组 |

worker按 `num_computed_tokens` 与prompt长度判断，最后一个只有1token的分块prefill仍标为prefill；不会用“本步只有1token”简单判断decode。NVTX范围覆盖CPU提交作用域，没有逐步加入CUDA synchronize；异步GPU duration仍从关联的CUDA活动读取。

默认decode日志在阶段转换和每32步输出，prefill/mixed会输出；如需逐步打印，采集前设置 `export QUANT_PROFILE_LOG_EVERY_N_STEPS=1`。这只调整日志密度，NVTX始终标记每步。

产物：

```text
log/benchmark_<id>/events.jsonl                 控制/请求阶段日志
log/benchmark_<id>/<model>/<case>/server.log    服务及worker日志
log/benchmark_<id>/<model>/<case>/worker_events/worker_<pid>.jsonl
log/profiling/<prefix>.nsys.log                benchmark与nsys控制台输出
log/profiling/<prefix>.audit.json              数据覆盖检查
```

JSON日志带UTC、monotonic_ns、PID、线程ID、阶段及请求/轮次等字段。worker的cpu_wall_seconds是方法/调度步的CPU墙钟，不是纯GPU kernel时间；实际load范围在worker行，server_startup还包括服务初始化等工作。

## 5. 更全面的CPU栈与调度采集：单独做一次

如果GPU kernel之间有明显空隙，基础配置的OSRT只能看到部分等待/I/O，不能自动解释CPU在执行哪个函数。以下独立开启CPU采样与线程调度：

```bash
NSYS_ARGS=(
  --trace=cuda,nvtx,osrt
  --sample=process-tree
  --cpuctxsw=process-tree
  --backtrace=dwarf
  --cuda-event-trace=false
  --cuda-graph-trace=node
  --cuda-flush-interval=1000
  --trace-fork-before-exec=false
  --wait=all
)
collect_one eager_cpu true fp8-local
collect_one no_eager_cpu false fp8-local

# 做完后恢复基础时间线配置。
NSYS_ARGS=("${NSYS_TIMELINE_ARGS[@]}")
```

先看nsys status --environment确认perf相关权限。若采样/调度不被允许，保留基础配置并记录限制；不要把未采集到的CPU图解释为空闲。CPU native栈采样也不等于完整Python源码级调用栈。

按问题增加其它选项，不建议一次全部打开：

| 选项 | 目的与限制 |
| --- | --- |
| --trace=cuda,nvtx,osrt,cublas | 关联cuBLAS API与BF16/GEMV；Triton GEMM不经过cuBLAS，不会因此获得Triton源码 |
| --cuda-memory-usage=true | 增加CUDA内存分配信息，适合加载/缓存分析；有额外开销，不是GEMM内部DRAM带宽 |
| --cuda-graph-trace=graph | 整图粒度，开销较低；看Graph回放总时序，不提供相同的节点热点细节 |
| --python-backtrace=cuda + --cudabacktrace=all | 仅在本机help支持时与CPU采样一起试用，查看CUDA API关联的Python栈；不能只在sample=none的基础配置后加一个backtrace开关 |
| --python-sampling=true --python-sampling-frequency=100 | 本机支持时采样Python栈；适合查GIL/Python处理开销，较低频率先试，与native CPU栈采样是不同选项 |

先执行 `nsys profile --help` 核对版本。需要memory等额外选项时，用 `NSYS_ARGS+=(--cuda-memory-usage=true)`；随后更换mode_label，例如eager_memory，避免与基础结果混淆。不同采集配置的墙钟/吞吐不能直接混为性能对照。

## 6. 采集完成后：先验收，再分析正式请求

### 6.1 选择报告并运行覆盖检查

```bash
PREFIX="$OUT/no_eager_fp8-local_${CASE_ID}_${RUN_TAG}"
REP="${PREFIX}.nsys-rep"
SQLITE="${PREFIX}.sqlite"

# collect_one已导出SQLite；对手工采集的报告，先执行一次：
# nsys export --type=sqlite --output="$SQLITE" "$REP"
python capstone/quantization_profiling/scripts/check_nsys_capture.py \
  "$SQLITE" --require-worker-ranges --expect-graph
```

检查eager时去掉 `--expect-graph`。检查以前未加本项目NVTX的报告时，同时去掉 `--require-worker-ranges`，否则自然会报缺少新增范围。

重点看：

| 字段 | 解释 |
| --- | --- |
| kernel_records | 节点/普通kernel执行记录数量 |
| gpu_graph_records | 整图GPU执行记录数量；node模式下可能是0 |
| graph_resource_records | Graph资源事件；再多也不能证明执行数据存在 |
| measure_gpu_records | 与正式请求范围重叠的GPU活动数量 |
| graph_launch_api_records、graph_node_kernel_records | 全程Graph launch与节点执行证据；measure_graph_*对应正式区间，Graph验收优先检查正式区间 |
| worker_ranges | 是否包含实际load_model、prefill、decode等范围 |
| incomplete_cuda_warnings | 已知CUDA事件不完整警告数量；非0时仍须人工检查 |
| key_data_present | 关键数据是否存在；不是逐事件完整性保证 |

旧的四个no_eager报告只有Graph资源而无GPU执行时间，不能通过该检查。文件大小、benchmark的passed和Graph Creation数量都不能替代此检查。

### 6.2 命令行先看整个实验的结构

```bash
nsys stats --help-reports
nsys stats --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,osrt_sum \
  "$SQLITE"
```

全程表包含加载、dummy profile、编译、Graph捕获、HTTP预热和正式请求。这里只用于确认阶段和实际kernel名字，不直接拿全程kernel占比比较模型速度。

### 6.3 只看正式请求：同一时间窗口跨进程过滤

```bash
nsys stats --filter-nvtx 'controller/measure@quantization_profiling' \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum,osrt_sum \
  "$SQLITE"

# 导出CSV。每种report生成一个文件；指定独立前缀。
nsys stats --filter-nvtx 'controller/measure@quantization_profiling' \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum \
  --format csv --output "${PREFIX}_measure" "$SQLITE"
```

`--filter-nvtx` 是按范围的时间窗口过滤，能选中benchmark范围时间内另一个worker进程的活动。默认只取第一个同名范围；本例只有1轮。多轮时用 `controller/measure@quantization_profiling/1` 选择第二轮，索引从0开始。

过滤会纳入与窗口重叠的完整事件，不会把跨边界kernel的duration精确裁短。比较时对齐输入、输出、请求数、eager模式及采集参数，并检查窗口边界。

### 6.4 看prefill/decode内各kernel的成本

```bash
# CPU范围统计、GPU投影及每个范围内的kernel明细。
nsys stats --filter-nvtx 'controller/measure@quantization_profiling' \
  --report nvtx_sum,nvtx_gpu_proj_sum,nvtx_kern_sum "$SQLITE"

# 保存详细时序，便于按Start、PID/TID、关联ID继续核对。
nsys stats --filter-nvtx 'controller/measure@quantization_profiling' \
  --report cuda_gpu_trace,cuda_kern_exec_trace,nvtx_gpu_proj_trace \
  --format csv --output "${PREFIX}_measure_detail" "$SQLITE"
```

在NVTX投影表中找worker/prefill、worker/decode和worker/mixed。GPU投影通过范围内发起的CUDA API关联GPU活动；它不是简单计算CPU范围长度。父进程中的controller/measure没有直接提交worker的CUDA kernel，通常不会拥有该跨进程GPU投影，需用它做时间过滤，再分析worker范围。

Graph模式下检查cudaGraphLaunch及Graph节点关联；若当前工具版本无法给Graph节点完整生成NVTX投影，就按GUI中的单步范围、launch与GPU节点时间线关联，不能将空投影表解释为没有计算。

| 表/列 | 应该回答的问题 |
| --- | --- |
| cuda_gpu_kern_sum：Total Time、Instances、Avg/Med/Max、Time % | 慢在调用更多，还是单次kernel更慢？占比只相对于所选kernel累计时间 |
| cuda_kern_exec_sum/trace：API、Queue、Kernel时间 | CPU提交、排队还是GPU执行更长？Queue可能是正常异步排队，需结合GPU空隙判断 |
| nvtx_sum：Total Time、Range Instances | CPU提交/控制范围多长、多少步？不能直接当GPU计算时间 |
| nvtx_gpu_proj_sum：Total Proj Time、Proj Avg、Total GPU Ops | 各阶段关联的GPU执行跨度；可能包含内部空隙，不等于所有kernel duration之和 |
| nvtx_kern_sum：NVTX Range、Kern Inst、Total Time | 哪个kernel在prefill或decode中花费最多 |
| cuda_api_sum：Num Calls、Avg、Total Time | launch、同步、分配是否频繁；CPU API耗时不是对应kernel的GPU duration |
| osrt_sum：Name、Total Time、Num Calls | 是否存在锁/等待/I/O热点；整段等待可能只是服务空闲或关闭 |

不要相加存在父子嵌套的NVTX总时间；也不要把多stream并行kernel的累计duration当GPU忙碌区间的并集。

四模型重点对照：

- fp8-local / fp8-official：确认是否仍是 `_w8a8_triton_block_scaled_mm` 与 `per_token_group_quant_8bit_kernel`；分别对照prefill和decode的次数、单次时间与kernel间隙。两个格式不同的模型可能分派到相同kernel家族。
- BF16：确认实际cuBLAS/GEMV/GEMM路径，按相同阶段作基线。
- AWQ：确认Marlin kernel，记录权重位宽和解包/反量化路径。AWQ Marlin与FP8 Marlin代表不同格式/计算路线，不只按名字比较。
- eager / no_eager：先确认后者实际Graph回放，再比较单步GPU跨度、CPU提交数量和GPU空隙。Graph减少提交开销，不保证原GEMM本身变快。

### 6.5 GUI逐步查看

1. 将nsys-rep复制到Windows，用兼容该报告格式的Nsight Systems打开；SQLite只用于CLI分析，不替代原报告。
2. 先查看Diagnostics Summary的采集器/worker警告。展开CUDA GPU行，确认正式区间确有kernel或Graph执行。
3. 在NVTX事件中查找domain `quantization_profiling`。先看worker/load_model、worker/profile_memory、worker/compile_and_graph_capture，区分加载和内部warmup。
4. 在benchmark进程找到controller/measure，选择并缩放到这个时间段；worker行按同一横坐标查看。
5. 展开实际GPU worker的线程、NVTX和CUDA API行，再展开GPU的Context/Stream。prefill通常出现在请求开始，后面是多个decode步；长输入可能有多个prefill块。
6. 选择worker/prefill或worker/decode范围，查看下方Events/Details；选择范围内CUDA launch，再定位关联GPU kernel。Graph模式则定位cudaGraphLaunch及Graph节点。
7. 查看同一个decode步内的量化kernel、矩阵kernel、attention、sampling及中间空隙。CPU范围结束后GPU仍可能继续执行，这是异步提交行为。
8. 在选定的正式时间窗口生成Statistics，查看CUDA GPU Kernel Summary、CUDA API Summary、NVTX GPU Projection；确认统计过滤已应用到所选区间。
9. 打开另一模型/模式报告，对齐同等阶段和输出token数比较。不要用整份报告的总长度或全程Time %作为服务推理速度。

界面名称和菜单位置可能随版本不同，判断依据是时间范围、PID/TID、关联CUDA活动和实际GPU duration。

### 6.6 从JSONL读取阶段日志

将BENCH_DIR替换为本次控制台打印的Artifacts目录：

```bash
BENCH_DIR=/workspace/AI_system/capstone/quantization_profiling/log/benchmark_实际ID
python - "$BENCH_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
paths = [root / "events.jsonl", *root.glob("*/*/worker_events/worker_*.jsonl")]
records = [json.loads(line) for path in paths if path.exists()
           for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in sorted(records, key=lambda x: x["monotonic_ns"]):
    if row["event"] in {"stage_end", "request_end", "client_first_text"}:
        print(json.dumps(row, ensure_ascii=False))
PY
```

该命令合并同一主机的单调时钟时间；不能把JSONL的monotonic_ns直接当成SQLite相对时间戳。NSYS内跨进程对齐使用工具自身的时间线。

## 7. 需要GPU硬件指标时，再采NCU

Nsight Systems先回答“耗时在哪个阶段、哪个kernel、GPU是否有空隙”。DRAM吞吐、SM/Tensor计算占用、occupancy等，交给Nsight Compute。不要同时运行nsys和ncu。

先用nsys确认实际kernel名，再在eager模式下小范围采样：

```bash
ncu --version
ncu --list-sets

# 0表示先看最早出现的匹配kernel，可能属于内部warmup，不能当作正式decode。
# 确定要跳过的匹配次数后修改MATCH_SKIP；改负载/框架版本后重新核对。
MATCH_SKIP=0
ncu --target-processes all \
  --kernel-name 'regex:.*_w8a8_triton_block_scaled_mm.*' \
  --launch-skip "$MATCH_SKIP" --launch-count 2 \
  --set basic \
  -o "$OUT/ncu_eager_fp8-local_${CASE_ID}_${RUN_TAG}" \
  python capstone/quantization_profiling/scripts/benchmark.py \
    --model fp8-local --case "$CASE_ID" --enforce-eager true --nvtx \
    --requests 1 --rounds 1
```

`--launch-skip`针对匹配的kernel，不是跳过HTTP请求数。NCU只捕获少量热点；采集区间是否对应正式prefill/decode仍需验证，不能固定照抄以前的skip数。basic通过后才考虑把 `--set basic` 改为 `--set full`，多pass/replay会显著延长时间。

NCU大幅拖慢服务时，用自定义benchmark配置增加 `startup_timeout_seconds` 和 `request_socket_timeout_seconds`，模型配置其余项保持同一基线。若出现ERR_NVGPUCTRPERM，需按目标机策略配置性能计数器权限；没有计数器数据就不能推断带宽瓶颈。BF16/AWQ要替换kernel过滤器，不会匹配这条Triton表达式。

更深入的受控实验与指标含义见 [FP8性能分析手册](nsight_fp8_performance_analysis.md)。

## 8. 常见失败与排查顺序

### 8.1 file_size / qdstrm导入失败，未生成nsys-rep

这是采集/导入链路失败信息，不能据此判定模型运行失败或某个GPU kernel出错。

```bash
# 保留完整控制台日志、版本和仍在的临时文件。
nsys --version
nsys status --environment
find "$OUT" /tmp -maxdepth 4 -type f -name '*.qdstrm' -ls 2>/dev/null || true
df -h "$OUT" /tmp
```

先根据nsys日志中Generating给出的实际路径查找qdstrm，再核对默认临时目录是否可写且空间足够、文件是否被外部清理、Linux采集器与QDSTRM importer及驱动是否兼容、worker是否被强杀或出现丢事件。成功导入后临时文件可能已被自动清理，find没有结果本身不代表异常。常规命令继续使用默认目录，不为排查此错误预先增加临时目录覆盖。

保留原qdstrm，先运行同版本QdstrmImporter的 `--help`，确认该版本的输入/输出选项后另行导入到新文件。不要把qdstrm仅改扩展名当成nsys-rep，也不要将它直接传给nsys export。若原文件已消失，单凭报错无法恢复，需重新采集。

### 8.2 等待re-parented processes

`--wait=all` 表示等待重新托管的子进程，不是错误本身。检查对应server.log、worker_events和进程树，确认是否仍在正常退出。新版benchmark会先给进程组SIGTERM，持续检查整个组，最多15秒后才记录shutdown_escalation并SIGKILL；包装进程提前退出不会立即强杀worker。仍未实现显式cudaProfilerStop协议，因此最终以覆盖检查和Diagnostics为准。

### 8.3 no_eager只有Graph Creation，没有GPU节点

先看report中的实际GPU执行、cudaGraphLaunch和采集警告，再看server.log中的Graph配置。确保使用node模式及受支持的Nsight/CUDA组合，并避免提前强杀worker。Graph Creation是资源元数据，不提供回放耗时。不能据此计算Graph加速比。

### 8.4 找不到NVTX、prefill/decode或worker类

- benchmark是否传了 `--nvtx`？默认关闭，可用 `--no-nvtx` 显式关闭；无此标记的旧报告不会自动补标记。
- requirements-profiling是否安装在实际启动benchmark/vLLM的同一个venv？`NVTX_DISABLE`是否已取消？
- run.json/各组summary的worker_class是否为 `profiling_worker.ProfilingWorker`？server.log是否有worker_instrumentation_ready？
- 是否为固定的vLLM 0.14.1？自定义worker通过已有PYTHONPATH载入，不修改site-packages；换版本需重新核对hook接口。
- 只有CPU范围无GPU活动时，不要继续解释kernel性能，先处理采集覆盖问题。

### 8.5 ZMQ ipc path is longer than 107 characters

2026-10-26的日志曾在服务启动时出现该异常：旧命令把通用TMPDIR指向过深的Nsight目录，vLLM继承后拼接UUID，IPC路径达到129字节，超过该socket的107字节限制。错误发生在创建EngineCore通信socket时，尚未进入GPU模型加载和推理。

在原Bash终端执行第3.4节：清除NSYS_TMPDIR、TMPDIR、VLLM_RPC_BASE_PATH的覆盖，生成新RUN_TAG并重新启动benchmark。恢复默认目录即可，无需再创建专用IPC目录。若这些旧export已写入shell启动文件，也删除相应旧行。

只读确认新进程实际使用的目录及可能遗留的环境变量：

```bash
python -c 'import tempfile; print(tempfile.gettempdir())'
printenv | grep -E '^(NSYS_TMPDIR|TMPDIR|TEMP|TMP|VLLM_RPC_BASE_PATH)=' || true
```

Python还会考虑系统已有的TEMP/TMP，因此以实际打印结果为准；若它们也被旧排查命令改成长路径，清除对应覆盖后再启动。新的unset不会改变已运行服务的环境。

若nsys-rep已生成而benchmark标记passed=false，仍属于失败实验，不能据此分析prefill/decode性能。改变模型精度、eager或端口无法解决IPC路径长度问题。

## 9. 实现依据与进一步阅读

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：采集参数、Graph、CPU/Python追踪；本机 `--help` 决定实际可用参数。
- 本机 `nsys stats --help`、`--help-reports nvtx_gpu_proj_sum`、`--help-reports nvtx_kern_sum`：过滤窗口、同线程投影与统计列。本文stats语法用本机2025.3.2帮助核对。
- [NVTX Python API](https://nvidia.github.io/NVTX/python/reference.html)：annotate及start_range/end_range；项目使用独立domain `quantization_profiling`。
- [vLLM 0.14.1 GPU Worker源码](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/v1/worker/gpu_worker.py)、[SchedulerOutput源码](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/v1/core/sched/output.py)：worker hook和阶段判断字段。
- [八份历史报告对比](../reports/nsys_eight_report_comparison_20261023.md)、[四模型报告阅读方法](nsys_report_reading_20261023.md)：旧报告已有的数据与限制。
