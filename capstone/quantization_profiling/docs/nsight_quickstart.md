# Nsight 简化版：直接运行 nsys 和 ncu

适用：Linux Bash、RTX 4090D、项目的 vLLM 0.14.1 环境。更新：2026-10-23。

流程：先用 nsys 找实际热点，再用 ncu 抽样分析热点。两个工具都直接启动现有 benchmark。

benchmark 默认读取 `capstone/quantization_profiling/data/frozen`，下方命令全部省略 `--snapshot`。请先确认目标机已有这份冻结数据；每次采集只选一组负载、1 个正式请求、1 轮，仍按快照执行预热。Profiler 下的吞吐和 HTTP 延迟不作为正式性能结果。

## 1. 准备环境和端口等待函数

在同一个终端先完整执行本节；新开终端时重新执行，后面的命令会使用这些变量和函数。

```bash
source /workspace/venvs/vllm-0112/bin/activate
cd /workspace/AI_system

OUT="capstone/quantization_profiling/log/profiling"
mkdir -p "$OUT"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)

nsys --version
ncu --version

# 每2秒尝试一次普通 bind，最多等待约90秒；同时覆盖 TIME_WAIT 的情况。
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
  echo "7081仍不可绑定，请检查监听进程和TCP状态。" >&2
  ss -tanp '( sport = :7081 )'
  return 1
}
```

## 2. nsys：采集 BF16 和官方 FP8 时间线

继续在同一终端执行。每个模型生成单独的 `.nsys-rep`，输出文件名包含本次时间戳。

```bash
for MODEL_PROFILE in bf16 fp8-official; do
  wait_7081 || break
  REPORT_PREFIX="$OUT/${MODEL_PROFILE}_w128_o128_${RUN_TAG}"

  nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --cuda-event-trace=false \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --wait=all \
    -o "$REPORT_PREFIX" \
    python capstone/quantization_profiling/scripts/benchmark.py \
      --model "$MODEL_PROFILE" \
      --case w128_o128 \
      --enforce-eager true \
      --requests 1 --rounds 1 || break

  if [ ! -s "${REPORT_PREFIX}.nsys-rep" ]; then
    echo "未生成nsys-rep，停止后续实验，保留qdstrm排查。" >&2
    break
  fi
done
```

- 四模型都采集：把循环模型列表换成 `bf16 awq fp8-official fp8-local`。
- 切换负载：把命令和文件名中的 `w128_o128` 一起换成 `w1024_o128`、`w4096_o256` 或 `w1024_o512`。
- 对照允许 CUDA Graph 的实验：把 `--enforce-eager true` 改为 `false`，保留 `--cuda-graph-trace=node`。实际是否捕获、回放以日志和时间线为准。

## 3. nsys：查看实际 kernel 和耗时

采集成功后执行；如果分析旧文件，将 `NSYS_REPORT` 换成其实际路径。

```bash
NSYS_REPORT="$OUT/fp8-official_w128_o128_${RUN_TAG}.nsys-rep"
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
| 只有 `.qdstrm`，提示 importer 错误 | 用下面的转换器命令检查并恢复；转换器必须匹配采集版本。 |
| ncu 显示 `No kernels were profiled` | 核对实际 kernel 名称，先把 `LAUNCH_SKIP` 设回 0；本例使用 `--target-processes all` 覆盖 worker。 |
| ncu 显示 `ERR_NVGPUCTRPERM` | 需要目标机管理员开放 GPU 性能计数器权限，处理后重新采集。 |

下面沿用你已查到的转换器路径；先看 `--version` 是否正常、`ldd` 是否还有 `not found`。`REPORT_PREFIX` 改成需要恢复的原始文件前缀，保留其 `.qdstrm`。

```bash
IMPORTER_BIN="/opt/nvidia/nsight-compute/2025.2.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter"
"$IMPORTER_BIN" --version
ldd "$IMPORTER_BIN"

REPORT_PREFIX="$OUT/bf16_w128_o128_${RUN_TAG}"
"$IMPORTER_BIN" \
  -i "${REPORT_PREFIX}.qdstrm" \
  -o "${REPORT_PREFIX}_recovered_$(date +%Y%m%dT%H%M%S).nsys-rep"
```

已有 BF16 原始文件可以先转换；此前 FP8 若在端口预检查阶段退出，其 `.qdstrm` 不包含成功的 FP8 推理，仍需重采。

## 进一步阅读

- [详细分析版](nsight_fp8_performance_analysis.md)：精确采集窗口、更多 NCU 指标和原理说明。
- [vLLM 0.14.1 Profiling](https://docs.vllm.ai/en/v0.14.1/contributing/profiling/)
- [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
