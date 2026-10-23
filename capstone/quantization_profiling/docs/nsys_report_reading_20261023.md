# 四份 nsys 报告：实际结果、GUI 解读与命令行分析

分析日期：2026-10-23。对象是 `log/profile` 中四份 `w128_o128_20261023T010402Z.nsys-rep`。本文只后处理已有报告，没有运行模型、量化或 benchmark；原始报告保持不变。

## 1. 本次实际发现

官方 FP8（此前称 fp8-origin）和本地 FP8 都运行 `_w8a8_triton_block_scaled_mm`，并调用 `per_token_group_quant_8bit_kernel`。AWQ 运行 `marlin::Marlin`；BF16 的 decode 主要是 cuBLAS `internal::gemvx::kernel`。

报告内嵌 stdout 确认四组均为：5 个预热请求、1 个正式请求，`max_num_seqs=1`，`enforce_eager=true`。FP8 后端请求为 `auto`，`VLLM_TEST_FORCE_FP8_MARLIN=0`。这次采集与此前每组 80 个正式请求的性能实验不同，不能混在一起统计。

### 1.1 相同 decode 周期对照

从最后一簇 prefill 之后，利用每周期一次、相同启动形状的 cuBLAS GEMV 选取中段 20 个连续周期。每组都有 `20×36×4=2880` 个 Transformer 线性层调用；FP8 另外有 2880 个激活量化调用。阶段按 kernel 模式推定，没有 NVTX 请求边界保证。

| 模型 | 20 周期墙钟 ms | 平均每周期 ms | kernel 数 | kernel 时间并集 ms | 无已记录 kernel 的时间 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 447.971 | 22.399 | 9620 | 340.675 | 107.296 |
| AWQ | 482.579 | 24.129 | 9620 | 135.077 | 347.502 |
| 官方 FP8 | 743.677 | 37.184 | 12500 | 371.201 | 372.476 |
| 本地 FP8 | 753.925 | 37.696 | 12500 | 370.429 | 383.496 |

这是 **profiler 下局部窗口的诊断数据**，不是 HTTP TPOT 或正式吞吐。不能据此断言正常运行时 AWQ 比 BF16 慢。追踪开销、CPU 调度和采集完整性都会影响结果。

“kernel 时间并集”合并所有已记录 kernel 的重叠区间后求长度。“无已记录 kernel”可能包含传输、CPU 工作、同步、调度等待及未记录的活动，不能直接解释为 GPU 完全空闲。加入 memcpy/memset 后，记录到的 CUDA 活动时间覆盖率约为 BF16 76.22%、AWQ 28.16%、官方 FP8 50.02%、本地 FP8 49.23%。这也不是 SM 利用率。

相同窗口的 GPU kernel 累计 duration：

| 项目 | BF16 | AWQ | 官方 FP8 | 本地 FP8 |
| --- | ---: | ---: | ---: | ---: |
| Transformer 线性层累计 ms，2880 次 | 297.863 | 89.799 | 322.462 | 321.558 |
| 每周期单独一次的 BF16 GEMV，20 次，累计 ms | 25.975 | 25.982 | 25.973 | 25.978 |
| 激活量化累计 ms，2880 次 | — | — | 4.738 | 4.764 |

每周期单独一次、`GridX=18992` 的 BF16 GEMV，结合模型结构很可能是输出头；仅凭名称不能证明算子身份。BF16 第一行是所有 GEMV 减去该形状 GEMV，避免和 FP8 的线性层范围混比。

目前支持的判断：

1. 两个 FP8 的线性层时间只差约 0.28%，路径与启动形状也一致，没有显示本地模型存在明显额外 kernel 成本。
2. 当前短请求、并发 1 的 Triton FP8 路径没有体现线性层加速：约 322 ms，对照 BF16 约 298 ms。
3. 激活量化本身仅占约 1.3% 的 kernel 累计时间，不能用这 4.7 ms 解释整体降速。
4. FP8 每周期多出 144 个量化 kernel，总调用数约增加 30%；同时有更多无已记录 kernel 的时间。下一步应查看 CPU launch 与 GPU 执行的关联、间隔和调度。
5. AWQ 的线性层明显更快，但周期没有同等缩短，说明单独优化矩阵 kernel 还不足以保证整体加速。

### 1.2 先处理采集完整性限制

四份报告均有以下诊断：

- `Not all CUDA events might have been collected.`，对应实际产生 kernel 的 GPU worker。
- CUDA 13.0 驱动不受采集器当前 build 支持，追踪使用 12.9 的库。
- NVTX 可能不完整；许多未使用 CUDA 的子进程还出现 No CUDA events 等提示。

因此不能假定最后一次请求的所有 CUDA 活动完整，报告最后一个 kernel 也不能直接当成请求完成边界。本文窗口的重复 kernel 数符合预期，但这不证明其他事件类型全部完整。

重新采集时使用支持目标驱动的采集器，并确保 GPU worker 正常退出、追踪数据完成刷新。仅升级 Windows 查看器或重新导出旧报告无法补回未采集事件。

## 2. 文件、设备和时间定位

本机 CLI/GUI 为 Nsight Systems 2025.3.2。四份报告已经导出到：

```text
D:\workspace\learing\AI_system\out\quantization_profiling\profile_analysis_20261023\
  bf16.sqlite / awq.sqlite / fp8-official.sqlite / fp8-local.sqlite
  *.info.json                     采集元信息
  *_full_*.csv                    每模型 7 张全局统计表
  *_decode20_*.csv                每模型 6 张局部统计表
  fp8-local_decode20_*trace*.csv   本地 FP8 的两张逐事件表
  decode20_comparison.json        窗口与计算结果
  compare_decode.py               本次形状专用的窗口推定/计算脚本
```

四组设备为同一 UUID 的 NVIDIA GeForce RTX 4090 D，compute capability 8.9，114 个 SM。整份记录包含加载、初始化、预热、推理和服务退出。

| 模型 | CUDA worker PID | 总记录长约 s | 最后一簇 prefill 线性层区间 s | 本文 decode 窗口 s |
| --- | ---: | ---: | --- | --- |
| BF16 | 424274 | 114.655 | 111.439169–111.459665 | 111.674046447–112.122016952 |
| AWQ | 437548 | 66.201 | 62.852834–62.876981 | 63.130370244–63.612948946 |
| 官方 FP8 | 427569 | 89.457 | 84.205236–84.238811 | 84.619835327–85.363512007 |
| 本地 FP8 | 440212 | 89.164 | 84.083385–84.117264 | 84.513019558–85.266944467 |

不能用总记录长度排名推理速度。表中时间采用 SQLite / nsys stats 的会话时间。GUI 保留默认 session time，并用具体事件 Start 对齐核对；修改时间原点后先对齐事件再使用坐标。prefill 列仅为所识别线性层的首尾，不是完整 prefill 墙钟边界。

## 3. 看哪些表和参数

先确定范围，再按下列顺序查看。GUI 名称因版本可能略有不同，CLI 名称已在本机验证。

| 顺序 | 表 / CLI report | 关注列 | 目的 |
| --- | --- | --- | --- |
| 1 | CUDA GPU Kernel Summary / `cuda_gpu_kern_sum` | Name、Total Time、Instances、Avg、Med、Max | 确认实际后端与主要 GPU 热点 |
| 2 | CUDA GPU Kernel/Grid/Block Summary / `cuda_gpu_kern_gb_sum` | GridXYZ、BlockXYZ、时间列 | 拆开同名不同形状，区分 prefill/decode |
| 3 | CUDA Kernel Launch & Exec Time Summary / `cuda_kern_exec_sum` | AAvg、QAvg、KAvg、TAvg、各 Med、Count、QCount | 分清 CPU API、排队和 GPU 执行 |
| 4 | CUDA API Summary / `cuda_api_sum` | Name、Num Calls、Total Time、Avg、Max | 检查 launch 数、同步、分配和传输 API |
| 5 | CUDA GPU Trace / `cuda_gpu_trace` | Start、Duration、CorrId、Strm、Grid、Reg/Trd | 找到慢调用和调用间隔 |
| 6 | CUDA Kernel Launch & Exec Time Trace / `cuda_kern_exec_trace` | API Start/Dur、Queue Dur、Kernel Start/Dur | 将 GPU kernel 对回 CPU launch |
| 7 | `cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum` | Operation、Count、Total Time、Total Bytes | 检查 HtoD/DtoH/D2D 的时间和规模 |
| 8 | NVTX Range Summary / `nvtx_sum` | Range、Instances、Total Time | 检查是否有可信的阶段边界 |

### 3.1 Kernel Summary

- `Total Time` 是各次 kernel duration 之和；不同 stream 重叠会重复计入。
- `Time (%)` 是所选 kernel 累计时间中的占比，不是 E2E 占比，也不是 GPU 利用率。
- `Instances` 应按请求数、输出 token 数或周期数归一化，不能直接比较不同工作量的总数。
- `Avg` 容易被初始化/慢调用拉高；对照 `Med`，有差异则查看逐次 trace。
- `Max/StdDev` 帮助定位异常，不能仅凭 Max 判断整体不稳定。
- 展开完整模板名。BF16 简化名称可能只显示 `kernel`，完整名称才能看见 `internal::gemvx` 和数据类型。

### 3.2 Launch & Exec：A、Q、K、T

设 CPU launch 为 `[a0,a1]`，GPU kernel 为 `[k0,k1]`：

```text
A = a1-a0                 CPU API duration
Q = k0-a1                 API 返回到 GPU 开始，仅非负记录参与统计
K = k1-k0                 GPU kernel duration
T = max(a1,k1)-a0         从 API 开始到 API/GPU 较晚结束的跨度
```

`AAvg` 不包含两次 API 之间所有 Python/框架工作。`QAvg` 不是 HTTP 队列时间，可能包含 stream 前序依赖及其他等待。GPU 也可能在 launch 返回前启动，因此要看 `QCount` 有多少记录参与 Q 统计。

`TAvg` 不可机械等于 A+Q+K：阶段可能重叠，Q 的样本也可能不同。Q 大可以意味着 CPU 提前排好了工作；Q 小也可能是 GPU 在等 CPU，不能把降低 Q 当作独立目标。

本地 FP8 的窗口中：GEMM `KAvg≈111.652 us`、`AAvg≈6.806 us`；量化 kernel `KAvg≈1.654 us`、`AAvg≈6.803 us`。小 kernel 也有逐次发射成本，但 CPU/GPU 时间会重叠，不能直接相加还原墙钟。

### 3.3 本地 FP8 要重点看的启动形状

| GEMM GridXYZ | BlockXYZ | 次数 | Avg us | 累计 ms |
| --- | --- | ---: | ---: | ---: |
| 32×1×1 | 128×1×1 | 1440 | 98.771 | 142.230 |
| 48×1×1 | 128×1×1 | 720 | 62.552 | 45.037 |
| 192×1×1 | 128×1×1 | 720 | 186.515 | 134.291 |

Grid=32 仍混合两种线性层。Grid/Block 是启动形状，不能直接当作 GEMM 的 M/N/K。

GPU 有 114 SM，32/48 个 CTA 的一次启动无法同时覆盖全部 SM，是小 batch 并行度有限的线索。该 Triton kernel 记录每线程 238 个寄存器、动态 shared memory 25088 B。它们是后续 NCU 分析 occupancy/访存的入口，不能只凭寄存器数断言发生 spill。

## 4. GUI：step by step

### 步骤 1：打开本地 FP8

用 **NVIDIA Nsight Systems** 的 File → Open 打开：

```text
D:\workspace\learing\AI_system\capstone\quantization_profiling\log\profile\fp8-local_w128_o128_20261023T010402Z.nsys-rep
```

本机 GUI 位于：

```text
C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\host-windows-x64\nsys-ui.exe
```

不需要目标 GPU 在线。再分别打开另外三份报告；不同运行的相同绝对时间不是同一阶段。

### 步骤 2：检查采集信息与告警

在报告顶部的视图选择器切换 **Analysis Summary**，核对 GPU、命令、采集项和版本；再切换 **Diagnostics Summary**，检查 event 不完整和驱动兼容提示及对应 PID。

不要因某个非 CUDA 子进程提示 No CUDA events 就认为整个报告不可读。本次真正需要关注的是实际 CUDA worker 自身的事件不完整提示。

### 步骤 3：展开 worker、CUDA API 和 GPU stream

进入 **Timeline View**，找到 PID **440212**。它可能显示为 Python 或 vLLM worker，以 PID 和 CUDA 活动为准。

同时展开 CPU 线程的 CUDA API 行，以及 GPU context/stream 下的 Kernels、Memory 行。只看 benchmark 主进程看不到真正的计算发射。

### 步骤 4：定位推理范围并放大

先从全局看到加载和靠后的密集重复区间。本地 FP8 最后一簇 prefill 线性层约在 **84.083–84.117 s**；本文 decode 窗口为 **84.513019558–85.266944467 s**。

在时间标尺拖选范围，通过选区菜单/右键菜单缩放到选区；继续放大到一个约 38 ms 的周期，最后放大到一个 kernel。

**缩放视野不一定改变所有聚合统计的范围。** 若界面提供时间范围过滤，检查表中范围和 Instances 是否同步变化；不确定时使用已导出的 decode20 CSV 验证聚合数据，GUI 负责定位具体事件。

### 步骤 5：打开 Events View

对 GPU kernel/stream 行右键 → **Show in Events View**。在底部逐事件表中搜索 `_w8a8_triton_block_scaled_mm`。

看 Start、Duration、Grid、Block、Stream、完整名称；先按 Duration 找慢调用，再按 Start 看节奏。双击表中事件可定位到时间线；双击时间线事件也能在对应事件表中定位。缩放变化可能重置事件表范围，应重新核对。

再搜索 `per_token_group_quant_8bit_kernel`，观察量化与 GEMM 是否交替、间隔多长。Events View 是逐次事件表，不是按名称聚合的 Kernel Summary。

### 步骤 6：将一个 GEMM 关联回 CPU

点击 GPU GEMM，通过关联高亮/箭头找到 CPU CUDA API；必要时在 Timeline Options 开启 correlation arrows。

依次检查：

1. 上一个 kernel 已结束，下一个 launch API 尚未开始：说明间隔发生在提交前，需要 CPU/框架侧证据。
2. launch 已返回，GPU 过一段时间才启动：检查 stream 前序依赖、同步、其他活动和 Queue Dur。
3. API 本身长：检查是否首次调用、同步、初始化或分配。
4. GEMM 本身长且持续执行：交给 NCU 分析计算/访存效率。
5. Kernels 行空白时 Memory 行是否忙：有传输时不能称为完全空闲。

本次没有 CPU 调用栈采样和完整 OS runtime 轨迹，无法从间隔可靠追溯到具体 Python 函数。

### 步骤 7：对照官方 FP8、BF16、AWQ

- 官方 FP8：使用 PID 427569 和它自己的时间窗，比较相同 Grid/Block、相同 decode 位置的 Duration 和间隔。
- BF16：找完整名称 `internal::gemvx`；prefill 的 CUTLASS `Kernel2` 不要混入 decode 平均值。
- AWQ：找 `marlin::Marlin`。`awq_marlin_repack_kernel` 位于加载/转换阶段，不能当成每 token 的反量化成本。
- 四组都检查剩余 BF16 GEMV、attention、RMSNorm 和每周期调用数。

建议保留三张带标尺截图：阶段全貌、相同数量 decode 周期、一个量化→GEMM→下一算子的 CPU/GPU 关联。包含 PID/stream 和 kernel 名称。

## 5. Windows PowerShell 命令行流程

所有命令只分析已有报告。派生文件写入 analysisDir。

### 步骤 1：定义路径并确认工具

```powershell
Set-Location 'D:\workspace\learing\AI_system'
$nsysTool = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe'
$reportDir = 'D:\workspace\learing\AI_system\capstone\quantization_profiling\log\profile'
$analysisDir = 'D:\workspace\learing\AI_system\out\quantization_profiling\profile_analysis_20261023'
$reportTag = 'w128_o128_20261023T010402Z'
$models = @('bf16', 'awq', 'fp8-official', 'fp8-local')
New-Item -ItemType Directory -Path $analysisDir -Force | Out-Null
& $nsysTool --version
& $nsysTool stats --help-reports
& $nsysTool stats --help-reports cuda_kern_exec_sum
```

目的：确认本机可用报表；最后一条直接查 A/Q/K/T 的版本内定义。

### 步骤 2：导出 SQLite 与元信息

```powershell
foreach ($modelProfile in $models) {
    $reportPath = Join-Path $reportDir "${modelProfile}_${reportTag}.nsys-rep"
    $sqlitePath = Join-Path $analysisDir "$modelProfile.sqlite"
    $infoPath = Join-Path $analysisDir "$modelProfile.info.json"
    if (-not (Test-Path -LiteralPath $sqlitePath)) {
        & $nsysTool export --quiet=true --type=sqlite "--output=$sqlitePath" $reportPath
        if ($LASTEXITCODE -ne 0) { throw "Export failed: $modelProfile" }
    }
    if (-not (Test-Path -LiteralPath $infoPath)) {
        & $nsysTool export --quiet=true --type=info "--output=$infoPath" $reportPath
        if ($LASTEXITCODE -ne 0) { throw "Info failed: $modelProfile" }
    }
}
```

目的：之后反复统计直接读 SQLite。当前已全部导出，可直接跳过。info.json 可能包含环境信息，对外分享前查看内容。

### 步骤 3：先查看全局后端和热点

```powershell
& $nsysTool stats --report cuda_gpu_kern_sum --timeunit us "$analysisDir/fp8-local.sqlite"
& $nsysTool stats --report cuda_api_sum --timeunit us "$analysisDir/fp8-local.sqlite"
& $nsysTool stats --report nvtx_sum "$analysisDir/fp8-local.sqlite"
```

目的：确认实际 kernel、主要调用及阶段标签。本次没有有用的 request/prefill/decode NVTX，主要只有 CUB 初始化范围，不能直接按 `--filter-nvtx decode` 筛选。

批量导出七张全局表：

```powershell
$summaryReports = 'cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_api_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum'
foreach ($modelProfile in $models) {
    & $nsysTool stats --report $summaryReports --format csv --timeunit us --force-overwrite=true --output "$analysisDir/${modelProfile}_full" "$analysisDir/$modelProfile.sqlite"
    if ($LASTEXITCODE -ne 0) { throw "Stats failed: $modelProfile" }
}
```

force-overwrite 只重写同名派生 CSV。全局 BF16 HtoD 累计约 19.328 s，包含权重加载，不能推断每个请求都被 PCIe 传输拖慢。

### 步骤 4：筛选同等工作量的 decode 窗口

```powershell
$timeWindows = [ordered]@{
    'bf16'         = '111.674046447s/112.122016952s'
    'awq'          = '63.130370244s/63.612948946s'
    'fp8-official' = '84.619835327s/85.363512007s'
    'fp8-local'    = '84.513019558s/85.266944467s'
}
$decodeReports = 'cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_api_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum'
foreach ($modelProfile in $timeWindows.Keys) {
    $timeRange = $timeWindows[$modelProfile]
    & $nsysTool stats --report $decodeReports --format csv --timeunit us "--filter-time=$timeRange" --force-overwrite=true --output "$analysisDir/${modelProfile}_decode20" "$analysisDir/$modelProfile.sqlite"
    if ($LASTEXITCODE -ne 0) { throw "Decode stats failed: $modelProfile" }
}
```

目的：排除加载/初始化，比较同样 20 个周期。文件名自动带 `_start=..._end=...` 后缀。

注意：

- filter-time 对相交事件整条计入，duration 不裁剪到选区。
- timeunit us 控制输出单位；筛选字符串中的 s 明确指定输入为秒。
- 本机 2025.3.2 使用 `111674046447/112122016952` 这样的长整数会报 Time format overflow；带 s 的小数已验证可用。
- 起点比前一周期 GEMV 结束多 1 ns，避免把恰好在左边界结束的事件再计入。
- 这些时间仅适用于当前四个文件。新采集必须重新定位。

### 步骤 5：看启动形状与 A/Q/K/T

```powershell
& $nsysTool stats --report cuda_gpu_kern_gb_sum --timeunit us --filter-time=84.513019558s/85.266944467s "$analysisDir/fp8-local.sqlite"
& $nsysTool stats --report cuda_kern_exec_sum --timeunit us --filter-time=84.513019558s/85.266944467s "$analysisDir/fp8-local.sqlite"
& $nsysTool stats --report cuda_api_sum --timeunit us --filter-time=84.513019558s/85.266944467s "$analysisDir/fp8-local.sqlite"
```

目的：拆开同名 GEMM 形状，同时看 CPU API、排队、GPU 执行。FP8 GEMM 使用 cuLaunchKernelEx，很多其他 kernel 使用 cudaLaunchKernel，只统计一种会漏掉大量调用。

cudaMemcpyAsync 的 CPU API duration 也不等于 GPU 拷贝 duration。同步 API 的 CPU 时间可能与 GPU 执行重叠，不能相加还原 E2E。

### 步骤 6：导出逐事件 trace

```powershell
& $nsysTool stats --report cuda_gpu_trace,cuda_kern_exec_trace --format csv --timeunit us --filter-time=84.513019558s/85.266944467s --force-overwrite=true --output "$analysisDir/fp8-local_decode20" "$analysisDir/fp8-local.sqlite"

$traceCsv = Get-ChildItem -LiteralPath $analysisDir -Filter 'fp8-local_decode20_cuda_gpu_trace_start=*.csv' | Select-Object -First 1
Import-Csv -LiteralPath $traceCsv.FullName |
    Where-Object { $_.Name -like '*block_scaled_mm*' } |
    Sort-Object { [double]$_.'Duration (us)' } -Descending |
    Select-Object -First 10 'Start (us)', 'Duration (us)', CorrId, GrdX, BlkX, 'Reg/Trd', Name
```

目的：定位慢调用，再回 GUI 看前后事件。这里 Start 的单位是 us，除以 1000000 才是秒。

cuda_kern_exec_trace 已将 API Start/Dur、Queue Dur、Kernel Start/Dur 关联好。自己 join 时，Correlation ID 必须同时限定进程，不能跨进程只按相同 ID 关联。

### 步骤 7：检查局部传输，排除加载污染

```powershell
& $nsysTool stats --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --timeunit us --filter-time=84.513019558s/85.266944467s "$analysisDir/fp8-local.sqlite"
```

本地 FP8 此窗口 D2D 约 5.898 MB，HtoD 约 0.002 MB，没有全局加载时的大规模 HtoD。

这些表只统计 memcpy/memset，不能测量 GEMM 内部读取权重/激活的 DRAM 流量或带宽利用率，也不能用于计算显存峰值。本报告没有完整的 CUDA 内存分配追踪。

### 步骤 8：复核本文窗口计算

```powershell
python "$analysisDir/compare_decode.py"
Get-Content -LiteralPath "$analysisDir/decode20_comparison.json"
```

脚本对 SQLite 使用只读连接，仅重写自己的 JSON。识别最后一簇 prefill 后，从随后 GridX=18992 的 GEMV 序列中取两个间隔 20 次调用的结束边界，并断言线性层次数符合预期。它是本次报告的计算记录，不是任意模型通用的阶段识别工具。

## 6. SQLite 查询：深入分析时使用

可用 DB Browser for SQLite 以只读方式打开文件，或用 Python 标准库 sqlite3。原始时间单位为 ns。

### 核对设备与告警 PID

```sql
SELECT name,uuid,computeMajor,computeMinor,smCount FROM TARGET_INFO_GPU;

SELECT (globalPid >> 24) & 16777215 AS pid,
       severity,text,COUNT(*) AS occurrences
FROM DIAGNOSTIC_EVENT
WHERE severity > 1
GROUP BY globalPid,severity,text
ORDER BY pid,occurrences DESC;
```

### 按完整名称和启动形状统计

下面只用于本地 FP8 的上述窗口：

```sql
SELECT (k.globalPid >> 24) & 16777215 AS pid,
       s.value AS kernel_name,
       k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
       COUNT(*) AS calls,
       ROUND(SUM(k.end-k.start)/1e6,3) AS total_ms,
       ROUND(AVG(k.end-k.start)/1e3,3) AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id=k.demangledName
WHERE k.start >= 84513019558 AND k.end <= 85266944467
GROUP BY k.globalPid,k.demangledName,
         k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ
ORDER BY total_ms DESC;
```

这里采用完全包含于范围的事件，与 nsys filter-time 的相交规则略有不同；当前边界已核对没有切开目标 kernel。

### 找无已记录 kernel 的间隔

```sql
WITH selected AS (
  SELECT k.start,k.end,s.value AS kernel_name,
         MAX(k.end) OVER (
           ORDER BY k.start,k.end
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
         ) AS previous_max_end
  FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
  JOIN StringIds AS s ON s.id=k.shortName
  WHERE k.start >= 84513019558 AND k.end <= 85266944467
)
SELECT previous_max_end/1e9 AS gap_start_s,
       start/1e9 AS next_kernel_s,
       (start-previous_max_end)/1e3 AS gap_us,
       kernel_name AS next_kernel
FROM selected
WHERE start > previous_max_end
ORDER BY gap_us DESC LIMIT 30;
```

目的：找到 GUI 中具体空白区间。使用此前所有 kernel 的最晚结束处理跨 stream 重叠，比简单 start-LAG(end) 更稳妥。判断 CUDA 活动是否空闲还要合并 memcpy/memset，并考虑未采集事件。

## 7. Linux CLI 等价操作

只处理已有文件，不重新采集。按实际位置修改路径：

```bash
REPORT_DIR=/path/to/profile
ANALYSIS_DIR=/path/to/profile_analysis
mkdir -p "$ANALYSIS_DIR"
nsys --version
nsys stats --help-reports cuda_kern_exec_sum

for MODEL_PROFILE in bf16 awq fp8-official fp8-local; do
  nsys export --quiet=true --type=sqlite \
    --output="$ANALYSIS_DIR/$MODEL_PROFILE.sqlite" \
    "$REPORT_DIR/${MODEL_PROFILE}_w128_o128_20261023T010402Z.nsys-rep" || break
done

nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_api_sum,cuda_kern_exec_sum \
  --filter-time=84.513019558s/85.266944467s \
  --timeunit us --format csv \
  --output "$ANALYSIS_DIR/fp8-local_decode20" \
  "$ANALYSIS_DIR/fp8-local.sqlite"
```

已有同名 SQLite 时直接使用，不必 export。其他三模型替换输入文件及各自的时间窗。

## 8. 哪些结论还需要 NCU 或重新采集

| 问题 | 当前报告能否回答 | 后续证据 |
| --- | --- | --- |
| 两个 FP8 是否走 Triton，AWQ 是否走 Marlin | 能，实际 kernel 已记录 | 保留完整名称和启动日志 |
| 本地 FP8 是否有明显额外 kernel 成本 | 当前窗口未显示 | 同形状、同配置、多次运行 |
| 是否量化 kernel 本身导致全部降速 | 不支持，其 GPU duration 很小 | launch/间隔分析及受控消融 |
| DRAM 带宽是否跑满 | 不能 | NCU DRAM throughput、Memory Workload |
| Tensor Core/原生 FP8 指令是否充分使用 | 不能只凭名称判断 | NCU pipeline/指令数据，必要时 SASS |
| 硬件 occupancy 是否约 49% | 不能，本文 49% 是时间覆盖 | NCU achieved/theoretical occupancy |
| 某个 Python 函数是否导致 GPU 等待 | 当前采样不足 | CPU 采样、OSRT、服务端阶段标记 |
| CUDA Graph 能加速多少 | 不能，四组均为 eager | eager/Graph 对照并确认实际回放 |
| 能否完整还原最后一次请求耗时 | 不能，有完整性警告且缺阶段边界 | 修正采集、正常刷新并加入 worker/request NVTX |

下一步先保证采集完整；在稳定 decode 比较同形状 kernel 和发射间隔，再用 NCU 抽样 Grid=32/48/192 的 Triton GEMM，区分并行度、访存、指令与 occupancy 问题。

官方参考：[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)（报告视图、Diagnostics、Events View、关联导航）、[Post-Collection Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)（统计与后处理）。命令已用本机 2025.3.2 对四份报告验证，字段解释同时核对该版本 nsys stats --help-reports。
