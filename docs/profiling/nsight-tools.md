# Nsight Systems / Nsight Compute（跨平台）

本文档面向当前仓库的 `perf_engineering_lab`，记录：

- Windows 与 Linux 两个平台的 Nsight 使用路径
- 本机 Windows 环境下 `ncu` / `nsys` 的安装与版本
- 官方文档中的适配信息
- 当前仓库新增的最小可用 NVTX 标注
- `ncu` / `nsys` 输出该怎么看
- 当前这台 Windows 机器上已经验证成功和暂时受限的部分

更新时间：2026-04-16

## 文档定位

- Windows 部分：基于当前这台机器的实测结果整理
- Linux 部分：基于当前 CUDA Linux 容器的实际验证结果整理，并保留跨平台仓库所需的工作流说明

这里做一个明确区分：

- “Windows 实测”表示我已经在当前机器上实际跑过命令
- “Linux 工作流”表示这是面向 `linux-make-cuda-release` preset 的 Linux 实测工作流，命令和脚本已在当前容器中重新验证

## 验证状态总览

| 平台 | 构建/运行 | nsys | ncu | 备注 |
|---|---|---|---|---|
| Windows | 已验证 | 已安装，当前非管理员会话下未完成采集 | 已安装，但被 GPU counter 权限阻塞 | 当前主机实测 |
| Linux | 已验证 | 已验证，可生成 `.nsys-rep` | 已验证，可生成 `.ncu-rep` | Ubuntu 22.04 + CUDA 12.8 容器实测 |

## Windows 本机环境

- OS：Windows 10 Pro
- CUDA Toolkit：13.0.88
- GPU：NVIDIA GeForce RTX 5060
- Compute Capability：12.0
- Driver：580.88
- Nsight Compute CLI：2025.3.1.0
- Nsight Systems CLI：2025.3.2.474

本机实际命令结果：

```powershell
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\ncu.bat" --version
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" --version
```

## 官方适配判断

### Nsight Compute

根据 NVIDIA 官方 `Nsight Compute 2025.3` release notes：

- 支持 Windows 10/11、Windows Server 2019/2022 作为 host / target
- 支持的 GPU 架构覆盖 Turing、Ampere、Ada，且 2025.3 明确加入了 Blackwell 芯片 `GB10x / GB11x / GB20x`
- 2025.3 新增对 CUDA 13.0 的支持

因此：

- RTX 4090（Ada，`sm_89`）在官方支持范围内
- RTX 5060（Blackwell，`sm_120`）在官方支持范围内
- 当前本机 `CUDA 13.0 + RTX 5060 + ncu 2025.3.1` 在版本层面是匹配的

参考：

- [Nsight Compute 2025.3 Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)
- [Nsight Compute System Requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)

### Nsight Systems

根据 NVIDIA 官方 `Nsight Systems Installation Guide` 与 `2025.3 Release Notes`：

- Host app 支持 Windows 10 / Windows Server 2019
- 文档中对 x86_64 / Arm SBSA target 提到的 GPU 架构为 Turing 及以上
- 支持 CUDA 10.0 及以上的大多数平台

这里我做一个明确说明：

- “Turing 及以上”意味着 Ada 和 Blackwell 也落在支持区间内，这是从官方表述推导出的结论
- `CUDA 10.0+` 也覆盖到了本机的 CUDA 13.0，这同样是基于官方区间做的推导

但 Windows 侧是否能顺利采集，还额外受管理员权限影响。官方文档明确提到 Windows 上 CLI collection 在非 `--trace=none` 场景需要管理员权限；这和本机实测现象是一致的。

参考：

- [Nsight Systems Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
- [Nsight Systems 2025.3 Release Notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## 当前仓库里的 NVTX 接入

这次已经给 `perf_engineering_lab` 加上了最小可用的 NVTX 标注。

### 代码位置

- 公共 RAII 封装：[nvtx.hpp](/E:/learning/AI_system/include/ai_system/profiling/nvtx.hpp)
- benchmark 自动 range：[benchmark_runner.cpp](/E:/learning/AI_system/src/benchmark/benchmark_runner.cpp:12)
- perf lab case / phase range：[perf_engineering_lab.cpp](/E:/learning/AI_system/labs/perf_engineering/perf_engineering_lab.cpp:124)

### 标注层级

当前时间线里会看到这些层级：

- `profiled_workload`
- `vector_add`
- `reduction`
- `naive_gemm`
- `prepare_inputs`
- `cpu_benchmark`
- `cuda_correctness`
- `cuda_benchmark`
- 以及 `run_benchmark()` 自动打进去的 `vector_add/cpu`、`vector_add/cuda`、`reduction/cpu`、`naive_gemm/cuda` 这类 benchmark range

### 设计意图

- 先用 `profiled_workload` 包住真正要看的实验阶段
- 每个算子一层 case range，时间线一眼能分区
- benchmark 名称作为更细粒度子区间，方便和 `ncu` / `nsys stats` 对上

如果编译环境里没有可用的 NVTX 头，这层封装会自动退化成 no-op，不影响 CPU-only 构建。

## Windows 本机验证结果

### 1. 工程构建与运行

以下验证已经通过：

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release
ctest --preset windows-vs2022-cuda-release
& "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 1024 --reduction-size 1024 `
  --gemm-m 32 --gemm-n 32 --gemm-k 32 `
  --warmup 1 --iters 2
```

说明：

- NVTX 接入没有破坏现有 build/test
- `perf_engineering_lab` 的 CPU/CUDA correctness 仍然通过

### 2. Nsight Systems

版本检查通过：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" --version
```

环境检查结果：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" status -e
```

本机输出要点：

- `Administrator privileges: No`
- `Sampling Environment: Fail`

进一步实测：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" profile `
  --sample=none --cpuctxsw=none --trace=cuda,nvtx `
  -o "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\nsight\perf_lab_nvtx" `
  "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 1024 --reduction-size 1024 `
  --gemm-m 32 --gemm-n 32 --gemm-k 32 `
  --warmup 1 --iters 1
```

当前非管理员会话下，`nsys profile` 在本机超时退出，未成功生成 `.nsys-rep`。

另外一个平台差异也已经确认：

- Linux 文档里常见的 `--trace=osrt`
- 在当前 Windows 安装的 `nsys` CLI 中不是合法 trace 值
- 本机可用值包含 `cuda`、`nvtx`、`wddm` 等

结论：

- `nsys` 已安装，版本检查通过
- 但当前会话不满足实际采集条件
- 如果要在 Windows 上真正采到 CUDA timeline，建议使用“管理员 PowerShell / CMD”重新运行

### 3. Nsight Compute

版本检查通过：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\ncu.bat" --version
```

设备识别检查：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\ncu.bat" `
  --query-metrics-mode suffix --metrics launch__waves_per_multiprocessor
```

本机输出要点：

- 能识别 `NVIDIA GeForce RTX 5060 (GB206)`
- 返回 `ERR_NVGPUCTRPERM`

这表示当前会话缺少访问 GPU performance counters 的权限。官方错误说明见：

- [ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

进一步对 `perf_engineering_lab` 做直接 profile 时，本机还遇到了 target launch 失败；但在真正开始采集之前，权限问题已经是明确阻塞项。

结论：

- `ncu` 已安装，版本检查通过
- 工具和硬件/Toolkit 的版本匹配没有问题
- 当前 Windows 会话还不能做真正的 kernel-level profiling
- 下一步应先解决 GPU counter 权限，再谈 profile 结果分析

## 推荐的 Windows 使用方式

### 一键脚本

仓库里已经提供：

- [profile_nsys.ps1](/E:/learning/AI_system/scripts/profile_nsys.ps1)
- [profile_ncu.ps1](/E:/learning/AI_system/scripts/profile_ncu.ps1)

最小示例：

```powershell
./scripts/profile_nsys.ps1 -DryRun
./scripts/profile_ncu.ps1 -DryRun
```

实际运行示例：

```powershell
./scripts/profile_nsys.ps1 -EnableWddm
./scripts/profile_ncu.ps1 -KernelRegex naive_gemm_kernel
```

说明：

- 两个脚本都会自动发现最新安装的 `nsys` / `ncu`
- 默认目标是 `windows-vs2022-cuda-release`
- 默认输出目录是 `out/build/<preset>/nsight/`
- `-DryRun` 会只打印命令，不真正执行

### Nsight Systems

先进入管理员 PowerShell，然后运行：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" profile `
  --sample=none --cpuctxsw=none --trace=cuda,nvtx `
  --force-overwrite=true `
  -o "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\nsight\perf_lab_nvtx" `
  "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 1048576 --reduction-size 1048576 `
  --gemm-m 256 --gemm-n 256 --gemm-k 256 `
  --warmup 2 --iters 5
```

说明：

- 先只用 `cuda,nvtx`
- `wddm` 需要更高权限时再加
- `profiled_workload` / `vector_add` / `cuda_benchmark` 这些 range 会直接出现在时间线上

### Nsight Compute

先确保已启用 GPU performance counters，然后再运行：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\ncu.bat" `
  --set basic `
  --target-processes all `
  --kernel-name regex:vector_add_kernel `
  --launch-count 1 `
  "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 1048576 --reduction-size 1024 `
  --gemm-m 32 --gemm-n 32 --gemm-k 32 `
  --warmup 1 --iters 1
```

再逐步换成：

- `reduce_sum_kernel`
- `naive_gemm_kernel`

## Linux 工作流

这一节对应项目的 Linux 路径，和 `linux-make-cuda-release` preset 对齐；下面的命令和脚本已在当前容器中实际验证。

### 一键脚本

仓库里已经提供：

- [profile_nsys.sh](/E:/learning/AI_system/scripts/profile_nsys.sh)
- [profile_ncu.sh](/E:/learning/AI_system/scripts/profile_ncu.sh)

最小示例：

```bash
./scripts/profile_nsys.sh --dry-run
./scripts/profile_ncu.sh --dry-run
```

实际运行示例：

```bash
./scripts/profile_nsys.sh --preset linux-make-cuda-release
./scripts/profile_ncu.sh --kernel-regex naive_gemm_kernel
```

说明：

- 两个脚本都会优先使用 `PATH` 里的 `nsys` / `ncu`
- 找不到时会继续回退到 `/usr/local/cuda/bin/` 或 `/usr/local/cuda-*/bin/`
- 默认输出目录是 `out/build/<preset>/nsight/`
- `--dry-run` 会只打印命令，不真正执行

### 1. 构建与测试

```bash
./scripts/configure.sh linux-make-cuda-release native
./scripts/build.sh linux-make-cuda-release
ctest --preset linux-make-cuda-release
```

### 2. 运行 `perf_engineering_lab`

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 \
  --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

### 3. 检查 Nsight 工具

```bash
command -v nsys
command -v ncu
nsys --version
ncu --version
```

### 4. Linux 上的 `nsys`

Linux 侧最常用的最小命令是：

```bash
nsys profile --sample=none --trace=cuda,nvtx,osrt -o /tmp/nsys-perf-lab \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1048576 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --warmup 2 --iters 5
```

生成报告后再看摘要：

```bash
nsys stats /tmp/nsys-perf-lab.nsys-rep
```

当前容器中的实测结果：

- `nsys --version` 可用
- `nsys status -e` 通过
- 上述最小命令可成功生成 `.nsys-rep`

说明：

- Linux 侧常见的 trace 组合是 `cuda,nvtx,osrt`
- `osrt` 这类 OS runtime trace 在 Linux 工作流里很常见
- 如果你要采 system-wide sampling、GPU metrics、syscall trace 等更重的特性，通常要按官方文档用 `sudo`

### 5. Linux 上的 `ncu`

先抓最小 kernel：

```bash
ncu --set basic --target-processes all \
  --kernel-name regex:vector_add_kernel \
  --launch-count 1 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

然后再逐步切换到：

- `reduce_sum_kernel`
- `naive_gemm_kernel`

当前容器中的实测结果：

- `ncu --version` 可用
- `vector_add_kernel` 最小命令可成功生成 `.ncu-rep`
- 可正常输出 `Launch Statistics`、`Occupancy` 等指标

### 6. Linux 侧权限提示

根据 Nsight Systems 官方用户指南：

- 一般的应用级 profiling 可以直接用 `nsys profile <app>`
- 某些功能，比如 system-wide CPU sampling，需要 `sudo`
- GPU metrics 也属于更高权限的场景

所以在 Linux 上建议分两步走：

1. 先用 `--sample=none --trace=cuda,nvtx,osrt` 跑通最小链路
2. 再按需要加 `sudo` 去做更重的系统级采集

参考：

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## 平台差异

这部分是跨平台仓库里最容易踩坑的地方。

### `nsys` trace 选项

- Linux 工作流里常见 `--trace=cuda,nvtx,osrt`
- 当前 Windows 安装的 `nsys` CLI 不接受 `osrt`
- Windows 侧更常见的是 `cuda`、`nvtx`、`wddm`

### 权限模型

- Windows：当前机器实测表明，非管理员会话会直接影响 `nsys profile`，`ncu` 还会被 GPU counter 权限卡住
- Linux：普通 CUDA/NVTX trace 往往更容易先跑通，但 system-wide / sampling / metrics 仍然经常需要 `sudo`

### 可执行文件路径

- Windows：通常是 `C:\Program Files\NVIDIA Corporation\...`
- Linux：更常见的是 `PATH` 里直接有 `nsys` / `ncu`，也可能位于 `/usr/local/cuda/bin/` 或 `/usr/local/cuda-*/bin/`

### 报告文件位置

- Windows：建议写到项目的 `out/build/.../nsight/`
- Linux：更常见的是 `/tmp/` 或当前工作目录

## `nsys` 输出怎么看

`nsys` 的核心价值是看“端到端时间线”和“谁在等谁”。

### GUI / `.nsys-rep`

最重要的几类轨道：

- CPU 线程轨道：看 host 侧是不是卡住了 launch 或同步
- CUDA API 轨道：看 `cudaMalloc/cudaMemcpy/cudaLaunchKernel/cudaDeviceSynchronize`
- GPU 轨道：看 kernel 和 memcpy 的时间分布
- NVTX 轨道：看你自己定义的阶段边界

在这个仓库里，最值得看的不是单个 kernel 的绝对时长，而是：

- `vector_add` / `reduction` / `naive_gemm` 之间谁更重
- `cpu_benchmark` 和 `cuda_benchmark` 的时间占比
- correctness 检查是否把 benchmark 结果“污染”了
- launch / sync 是否过密

### `nsys stats`

官方用户指南列出的常用 report 里，对这个仓库最实用的是：

- `cuda_api_sum`
- `cuda_gpu_trace`
- `cuda_gpu_sum[:nvtx-name]`
- `nvtx_sum`

参考：

- [Nsight Systems User Guide: report scripts](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

一个实用心法：

- `cuda_api_sum`：看 CPU 侧 API 时间花在哪里
- `cuda_gpu_trace`：看时间顺序和单次 kernel 细节
- `cuda_gpu_sum`：看同名 kernel 聚合后谁最重
- `nvtx_sum`：看自己划分的阶段累计耗时

如果你已经加了 NVTX，那么 `cuda_gpu_sum[:nvtx-name]` 很适合把 kernel 按“处于哪个 NVTX phase”重新归组。

## `ncu` 输出怎么看

`ncu` 的核心价值是看“单个 kernel 为什么快/慢”。

在 `--set basic` 下，最常见、最值得先看懂的是下面几组。

### GPU Speed Of Light Throughput

这是一个高层摘要。

关注点：

- Compute throughput 接近不接近峰值
- Memory throughput 接近不接近峰值

理解方式：

- Compute 很低、Memory 也很低：通常是 launch 太小，GPU 根本没吃满
- Memory 很高、Compute 一般：更像 memory-bound
- Compute 很高、Memory 一般：更像 compute-bound

### Launch Statistics

这是 launch 配置层面的信息。

关注点：

- Grid / Block 配置
- Waves Per SM
- 每次 launch 到底给 GPU 多少工作

对这个仓库最常见的意义：

- 小尺寸 `vector_add` / `reduction` 常常不是 kernel 写得差，而是 grid 太小
- `Waves Per SM` 很低时，GPU 利用率低通常很正常

### Occupancy

这是“理论上能塞多少 warps”和“实际上达到了多少”的对比。

关注点：

- Theoretical Occupancy
- Achieved Occupancy
- 限制来源是不是寄存器、shared memory 或 block size

常见判断：

- Theoretical 很低：先看 block / register / smem 配置
- Theoretical 很高但 Achieved 很低：通常是工作规模不够，或者运行时行为没有把 SM 喂满

### Memory Workload Analysis

这是看内存系统压力的主入口。

关注点：

- DRAM / L2 / L1/TEX 利用率
- load/store 分布
- 是否出现明显的 memory bottleneck

对 `sgemm_v2` 这类后续优化尤其重要，因为：

- vectorized load
- padding
- bank conflict 处理

最后都需要在这里找证据，而不是只看 GFLOPS。

### Compute Workload Analysis

这是看算力管线利用率的入口。

关注点：

- 指令类型
- 算术吞吐
- pipeline 利用率

对后面的 `wmma_demo`、Tensor Core、CUTLASS/Triton 对比最有帮助。

参考：

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

## 结合这个仓库，怎么读结果

### 先看 `perf_engineering_lab` 自己的表

先确认：

- `Validation` 全是 `PASS`
- benchmark 表里 `avg_ms / min_ms / max_ms / perf` 合理

这一步不通过，就不要继续看 Nsight。

### 再用 `nsys`

先回答这类问题：

- 真正慢的是 GPU kernel、CUDA API，还是 host 同步？
- 当前 workload 是不是被很多很碎的调用切断了？
- correctness 路径有没有混进 profiling 区间？

### 最后才用 `ncu`

只在下面这些问题出现时再进 `ncu`：

- 为什么 occupancy 低？
- 为什么 GEMM 吞吐上不去？
- 是 DRAM/L2/shared memory 还是算力成为瓶颈？

## 当前结论

截至 2026-04-16，这台 Windows 机器上的状态是：

- `perf_engineering_lab` 已接入最小可用 NVTX 标注
- 工程构建、测试、lab 运行全部正常
- `ncu 2025.3.1` 与 `CUDA 13.0 + RTX 5060` 在版本层面匹配
- `nsys 2025.3.2` 与当前环境在版本层面没有明显冲突
- 但当前非管理员会话下，`nsys profile` 未成功完成，`ncu` 明确被 GPU performance counter 权限阻塞

如果你要在这台机器上真正开始 profiling，建议按这个顺序处理：

1. 用管理员 PowerShell 重新运行 `nsys`
2. 处理 GPU performance counter 权限，再运行 `ncu`
3. 先抓 `vector_add`，再抓 `reduction`，最后抓 `naive_gemm`
