# Nsight 简化版：直接运行 nsys 和 ncu

适用：Linux Bash、RTX 4090D、项目的 vLLM 0.14.1 环境。更新：2026-10-23。

流程：先用 nsys 找实际热点，再用 ncu 抽样分析热点。两个工具都直接启动现有 benchmark。

benchmark 默认读取 `capstone/quantization_profiling/data/frozen`，下方命令全部省略 `--snapshot`。请先确认目标机已有这份冻结数据；每次采集只选一组负载、1 个正式请求、1 轮，仍按快照执行预热。Profiler 下的吞吐和 HTTP 延迟不作为正式性能结果。

## 1. 准备环境和端口等待函数

在同一个终端先完整执行本节；新开终端时重新执行，后面的命令会使用这些变量和函数。

```bash
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

mkdir -p capstone/quantization_profiling/log/profiling
export VLLM_WORKER_MULTIPROC_METHOD=spawn
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)

# 每2秒检查一次7081，最多等待约90秒
wait_7081() {
  local attempt
  for attempt in {1..45}; do
    if python -c \
      'import socket; s=socket.socket(); s.bind(("0.0.0.0",7081)); s.close()' \
      2>/dev/null; then
      return 0
    fi
    sleep 2
  done

  echo "7081仍不可绑定，请检查端口状态。" >&2
  ss -tanp '( sport = :7081 )'
  return 1
}
```

## 2. nsys：采集四模型时间线（当前为 Graph 对照）

继续在同一终端执行。保留当前 `--enforce-eager false` 和逐节点 `--cuda-graph-trace=node` 配置；是否实际捕获/回放 Graph 仍要看日志与时间线。每个模型保存独立报告和 nsys 日志。临时目录只改变文件位置，不保证能解决采集器内部错误。

```bash
for MODEL_PROFILE in bf16 fp8-official; do
  wait_7081 || break

  REPORT_PREFIX="capstone/quantization_profiling/log/profiling/no_eager_${MODEL_PROFILE}_w128_o128_${RUN_TAG}"

  nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --cuda-event-trace=false \
    --trace-fork-before-exec=true \
    --wait=all \
    -o "$REPORT_PREFIX" \
    python capstone/quantization_profiling/scripts/benchmark.py \
      --snapshot capstone/quantization_profiling/data/frozen \
      --model "$MODEL_PROFILE" \
      --case w128_o128 \
      --enforce-eager false \
      --requests 1 --rounds 1 || break

  if [ ! -s "${REPORT_PREFIX}.nsys-rep" ]; then
    echo "未生成nsys-rep，停止后续实验，保留qdstrm排查。" >&2
    break
  fi
done
```

- 只采一模型：把循环模型列表改成 `fp8-local` 或 `fp8-official`，便于先定位采集错误。
- 切换负载：把命令和文件名中的 `w128_o128` 一起换成 `w1024_o128`、`w4096_o256` 或 `w1024_o512`。
- eager 对照：把 `--enforce-eager false` 改为 `true`，同时将文件名前缀 `no_eager_` 改为 `eager_`，避免误标。Graph 出错的分步定位见第 5.1 节。

## 3. nsys：查看实际 kernel 和耗时

采集成功后执行；如果分析旧文件，将 `NSYS_REPORT` 换成其实际路径。

```bash
NSYS_REPORT="$OUT/no_eager_fp8-official_w128_o128_${RUN_TAG}.nsys-rep"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "$NSYS_REPORT"
```

先看 `Name`、`Instances`、`Avg`、`Total Time`。`Time (%)` 是所统计 kernel 的累计耗时占比，不是请求 E2E 的占比。

把 `.nsys-rep` 复制到 Windows，用 Nsight Systems GUI 打开。定位 EngineCore/GPU worker，选择预热后的推理区间；本命令也采集了加载、初始化和预热，不会自动把全部记录标成 prefill/decode。

## 4. ncu：只采少量真实热点

先把 `KERNEL_RE` 改成上一步实际看到的热点名称片段，再执行本节。BF16、AWQ、FP8 的名称可能不同，应分别选取。

```bash
MODEL_PROFILE=fp8-official
KERNEL_RE='替换为nsys报告中的实际热点名称片段'
LAUNCH_SKIP=0
NCU_REPORT="$OUT/ncu_${MODEL_PROFILE}_w128_o128_${RUN_TAG}"

wait_7081 && ncu \
  --target-processes all \
  --set basic \
  --replay-mode kernel \
  --kernel-name-base demangled \
  --kernel-name "regex:$KERNEL_RE" \
  --launch-skip "$LAUNCH_SKIP" \
  --launch-count 3 \
  --export "$NCU_REPORT" \
  python capstone/quantization_profiling/scripts/benchmark.py \
    --model "$MODEL_PROFILE" \
    --case w128_o128 \
    --enforce-eager true \
    --requests 1 --rounds 1
```

输出为 `${NCU_REPORT}.ncu-rep`，复制到 Windows 后用 Nsight Compute GUI 打开。先看 Duration、计算/访存利用率和 Occupancy；需要更多指标时再查详细版。

`--launch-count 3` 是最多采 3 次匹配的 kernel 调用；`--launch-skip` 也是按匹配调用计数，都不是请求数。默认从第一个匹配调用开始，可能落在初始化或预热阶段；要分析稳定 decode，先在 nsys 中定位，再设置对应的跳过次数。本例明确使用 eager，分析其结果时记录与 Graph 模式的配置差异。

## 5. 常见报错与已有文件恢复

| 现象 | 处理 |
| --- | --- |
| `wait_7081: command not found` | 在当前 Bash 终端重新执行第 1 节。 |
| `Address already in use` | 执行 `ss -ltnp '( sport = :7081 )'` 核对监听者；若只有 TIME_WAIT，等待函数会重试。真实旧服务需正常停止。 |
| `libdw.so.1` 缺失 | 在运行工具的同一系统/容器安装：Ubuntu 22.04 用 `sudo apt-get install -y libdw1`；24.04 用 `sudo apt-get install -y libdw1t64`；安装前先执行 `sudo apt-get update`。 |
| `Generating ...qdstrm` 后报 `boost::filesystem::file_size` | 生成阶段查询文件失败，先检查原始文件是否真实存在，再按第 5.1 节排查。不能仅凭此错误断言 Graph kernel 出错。 |
| 只有 `.qdstrm`，提示 importer 错误 | 用第 5.2 节检查恢复；转换器必须匹配采集版本。 |
| ncu 显示 `No kernels were profiled` | 核对实际 kernel 名称，先把 `LAUNCH_SKIP` 设回 0；本例使用 `--target-processes all` 覆盖 worker。 |
| ncu 显示 `ERR_NVGPUCTRPERM` | 需要目标机管理员开放 GPU 性能计数器权限，处理后重新采集。 |

### 5.1 `file_size: No such file or directory`：先确定失败阶段

典型输出：

```text
Artifacts: .../benchmark_...
The target application terminated. One or more process it created re-parented.
Waiting for termination of re-parented processes.
Generating '/tmp/nsys-report-....qdstrm'
FATAL ERROR: ... boost::filesystem::file_size: No such file or directory
```

含义：benchmark 已运行到写实验产物的末尾，随后 nsys 在收集收尾/报告生成阶段发生文件查询错误。`Artifacts:` 本身不代表所有请求成功，先检查该目录的 summary.json。re-parented 是子进程被父进程退出后重新托管的提示；`--wait=all` 正在等待这些进程，不是这次 FATAL 的直接解释。`Generating` 也不保证已经生成完整 qdstrm。

目前这段错误没有给出缺失文件路径，不能确定缺失的是原始流、某个进程的中间文件，还是转换依赖；也不能直接认定是磁盘满、Graph bug 或目录权限。

**A. 先读已有结果，不立即重跑**

```bash
BENCH_DIR="/workspace/AI_system/capstone/quantization_profiling/log/benchmark_20261023T024113Z_2fea35cb"
python -m json.tool "$BENCH_DIR/summary.json"

# 查这次错误中提到的原始流。若不存在，再找输出目录和临时目录。
ls -lh /tmp/nsys-report-458c.qdstrm
find /tmp /workspace/AI_system/capstone/quantization_profiling/log \
  -maxdepth 4 -type f -name '*.qdstrm' -ls 2>/dev/null
```

有非空原始流才有转换恢复的可能。没有原始流时，换 importer 无法补回采集数据；不要删除可能还有用的 qdstrm 或本次临时目录。

**B. 检查工具、驱动、空间和临时目录**

```bash
type -a nsys
nsys --version
nvidia-smi
printf 'NSYS_TMPDIR=%s\nTMPDIR=%s\n' "${NSYS_TMPDIR:-}" "${TMPDIR:-}"
df -h /tmp /workspace
df -i /tmp /workspace

NSYS_EXE=$(readlink -f "$(command -v nsys)")
printf 'nsys executable: %s\n' "$NSYS_EXE"
find "$(dirname "$NSYS_EXE")/.." -maxdepth 4 -type f -name QdstrmImporter -print 2>/dev/null
```

`df -h` 查容量，`df -i` 查 inode。文件不存在不等于磁盘满；这些是排除项。若 nsys 是包装脚本，上述相邻目录不一定是完整安装目录，需按实际安装位置查找。

之前四份 eager 报告提示 CUDA 13.0 驱动使用 12.9 追踪库；如果仍是同一套采集环境，优先确认采集器版本支持目标驱动。版本兼容是待验证因素，不是当前错误的已证实根因。

第 1 节已设置独立 NSYS_TMPDIR，并设置 TMPDIR 作为兼容回退。重跑时检查日志中的临时路径是否改变。先只保留循环中的一个模型；第 2 节会保存完整 nsys 日志和真实退出码。不要同时启动多份 profiler 来排查。

**C. 一次只改一个追踪选项**

保持 `--enforce-eager false`，先只把 `--cuda-graph-trace=node` 改成 `graph`，并更换输出前缀。`graph` 记录整图，开销较低；它无法提供相同的逐节点 kernel 细节，所以这是定位采集问题的对照，不是对原采集配置已修复的证明。

如果 graph 成功而 node 失败，证据指向逐节点追踪/版本兼容路径，应升级或核对采集器，并重新验证 node。不能说模型本身的 Graph 执行一定有错。

若仍失败，在此基础上单独将 `--trace-fork-before-exec=true` 改成 `false`。该开关控制 fork 到 exec 之间的特殊追踪，不是“是否追踪所有 worker”的总开关；当前使用 spawn，正常 exec 后的子进程仍应核对是否采集。NVIDIA 提醒 fork-before-exec 追踪可能导致崩溃/死锁，因此适合作为独立排除项。

最后再做 eager=true 对照，并记录每次改变。若相同工具/目录下 eager 成功而 Graph 失败，只能将问题范围缩小到 Graph 相关采集、运行及退出路径。

**D. worker 退出和 CUDA buffer 刷新是另一项限制**

当前 `scripts/benchmark.py` 的 `stop_process()` 先向服务进程组发 SIGTERM；等待的是 API 父进程，父进程一退出就会尝试向剩余同组进程发 SIGKILL。因此 worker 未必获得完整的 15 秒优雅退出时间。该行为可能影响 profiler 刷新，但不能仅凭当前 file_size 错误确认因果。

可单独增加 `--cuda-flush-interval=1000` 作为数据保留对照，尝试每秒刷新 CUDA buffer；这会增加追踪开销，也不能保证修复缺文件错误或保住最后一次刷新之后的数据。要解决退出完整性，应让 worker 在采集停止前正常完成、或在服务退出前结束采集；不能只在 benchmark 主进程退出后加 sleep 期待已被杀死的 worker 刷新。

不建议为了去掉提示直接改为 `--wait=primary`；它可能更早结束等待，却不能保证修复缺失文件，并可能丢掉仍在运行的 worker 的尾部追踪。

**E. 如果仍报同样错误，查具体缺失路径**

如果系统已有 strace，可将第 2 节的单模型 `nsys profile ...` 命令作为其目标，前面加上：

```bash
strace -ff -tt -s 256 -e trace=%file -o "$OUT/nsys_filetrace" \
  nsys profile ...
# 上面的省略号必须替换为完整命令，不能直接照抄运行。
rg 'ENOENT|nsys-report|qdstrm|QdstrmImporter' "$OUT"/nsys_filetrace*
```

这是诊断性重跑，开销较大，不能作为性能数据。ENOENT 中有许多正常的依赖搜索失败，应结合报错时间和临时报告路径判断，不能将第一条 ENOENT 当作根因。

### 5.2 原始 qdstrm 存在时恢复

先确认输入文件非空，再使用与采集版本匹配的转换器。不要直接沿用另一个安装目录里的 QdstrmImporter；路径中写了 nsight-compute 不自动证明不兼容，但也不能证明与当前 nsys 配套。

```bash
RAW_QDSTRM="/替换为实际存在的报告路径.qdstrm"
# 新版本支持 nsys import 时，优先用同一 nsys 安装执行。
if [ ! -s "$RAW_QDSTRM" ]; then
  echo "原始流不存在或为空，停止转换" >&2
elif nsys import --help >/dev/null 2>&1; then
  nsys import --input-file "$RAW_QDSTRM" \
    --output-file "${RAW_QDSTRM%.qdstrm}_recovered_$(date +%Y%m%dT%H%M%S).nsys-rep"
else
  echo "当前版本无import子命令，使用下面的配套QdstrmImporter。" >&2
fi
```

旧版没有 import 子命令时，使用 B 步定位到的配套转换器；以下先改成真实路径：

```bash
IMPORTER_BIN="/替换为同版本NsightSystems安装目录/host-linux-x64/QdstrmImporter"
"$IMPORTER_BIN" --version
ldd "$IMPORTER_BIN"

if [ -s "$RAW_QDSTRM" ]; then
  "$IMPORTER_BIN" -i "$RAW_QDSTRM" \
    -o "${RAW_QDSTRM%.qdstrm}_recovered_$(date +%Y%m%dT%H%M%S).nsys-rep"
fi
```

`ldd` 的 not found 指向依赖缺失，先补依赖再转；格式版本不匹配则换配套版本。已有完整 qdstrm 转换成功才说明该文件可恢复，不保证采集覆盖完整；打开报告后继续检查 Diagnostics Summary。

参考：[NVIDIA 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)中的临时目录、wait、Graph trace、fork-before-exec 和报告恢复说明。本文没有在远端复现该 FATAL，以上是依据日志阶段与项目代码给出的定位步骤。

## 进一步阅读

- [详细分析版](nsight_fp8_performance_analysis.md)：精确采集窗口、更多 NCU 指标和原理说明。
- [vLLM 0.14.1 Profiling](https://docs.vllm.ai/en/v0.14.1/contributing/profiling/)
- [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
