# Nsight Systems / Nsight Compute Guide

本文档面向当前仓库的 `perf_engineering_lab`，目标是给出一份可直接落地的 Nsight 使用手册，包括：

- Windows 与 Linux 的常用命令
- `nsys` / `ncu` 的使用顺序
- `ncu` 单独抓一个 kernel 与一次抓多个/全部 kernel 的方法
- 常见输出文件、GUI 视图和结果解读方式
- 当前仓库里 NVTX 的接入方式与推荐实践

更新时间：2026-04-17

## 1. 先说结论

对这个仓库，最推荐的 profiling 流程是：

1. 先跑 `perf_engineering_lab`，确认 `Validation` 全部 `PASS`
2. 先用 `nsys` 看端到端时间线
3. 再用 `ncu` 抓热点 kernel
4. 默认优先单 kernel 抓取；只有在“先摸底都有哪些 kernel”时，才建议一次抓全部

一句话区分两者：

- `nsys` 回答：时间花在了哪里，谁在等谁
- `ncu` 回答：某一个 kernel 为什么快/慢

## 2. 输出文件是什么

### 2.1 `nsys`

`nsys profile` 最主要的输出是：

- `*.nsys-rep`

这是原始报告文件，GUI 主要直接打开它。

后续执行 `nsys stats xxx.nsys-rep` 时，如果旁边没有 SQLite 导出，通常会自动生成：

- `*.sqlite`

所以常见情况是最终看到两个文件：

- `xxx.nsys-rep`
- `xxx.sqlite`

但它们不是同一阶段同时生成的：

- `nsys-rep` 来自 `nsys profile`
- `sqlite` 常常来自 `nsys stats`

### 2.2 `ncu`

`ncu` 导出的主文件是：

- `*.ncu-rep`

这个文件既可以在 GUI 中打开，也可以配合 `ncu --import` 再查看。

## 3. 当前仓库里的 NVTX

当前仓库已经给 `perf_engineering_lab` 接上了最小可用的 NVTX：

- 公共封装：[nvtx.hpp](/E:/learning/AI_system/include/ai_system/profiling/nvtx.hpp:1)
- benchmark 公共入口：[benchmark_runner.cpp](/E:/learning/AI_system/src/benchmark/benchmark_runner.cpp:10)
- 实验主逻辑：[perf_engineering_lab.cpp](/E:/learning/AI_system/labs/perf_engineering/perf_engineering_lab.cpp:1)

当前时间线上常见的 NVTX 层级大致是：

```text
profiled_workload
|- vector_add
|- reduction
`- gemm
   |- prepare_inputs
   |- cuda_correctness
   |- cuda_benchmark
   |  |- gemm/e2e/cuda_naive
   |  |- gemm/e2e/cublas_sgemm
   |  |- gemm/e2e/cublas_hgemm
   |  `- gemm/e2e/cublas_tensor_core
   |- kernel_only_correctness
   `- kernel_only_benchmark
      |- gemm/kernel_only/cuda_naive
      |- gemm/kernel_only/cublas_sgemm
      |- gemm/kernel_only/cublas_hgemm
      `- gemm/kernel_only/cublas_tensor_core
```

注意：

- 真正的层级来自作用域里的 NVTX 嵌套
- 名字里的 `/` 只是命名习惯，不是 NVTX 自动识别出的层级

## 4. 当前环境与验证状态

### Windows：已实测

当前机器实测环境：

- OS：Windows 10 Pro
- CUDA Toolkit：13.0.88
- GPU：NVIDIA GeForce RTX 5060
- Compute Capability：12.0
- Driver：580.88
- Nsight Compute CLI：2025.3.1.0
- Nsight Systems CLI：2025.3.2.474

已经实测通过的内容：

- `cmake --build --preset windows-vs2022-cuda-release --config Release`
- `ctest --preset windows-vs2022-cuda-release`
- `perf_engineering_lab`
- 提权后的 `nsys`
- 提权后的 `ncu`

### Linux：只提供命令，不做本次验证

本节会给出 Linux 工作流和命令模板，方便跨平台项目使用。  
但这里要明确说明：

- **本次没有在当前会话里验证 Linux 的 `nsys` / `ncu`**
- Linux 内容按通用 CUDA/Nsight 工作流整理，作为落地命令模板使用

## 5. 推荐工作流

### 5.1 第一步：先跑 benchmark

先确认实验本身是正确的：

```powershell
& "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 2048 --reduction-size 2048 `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --warmup 1 --iters 1
```

只有在：

- 程序能正常退出
- `Validation` 全是 `PASS` 或预期的 `SKIP`

时，才值得继续做 Nsight。

### 5.2 第二步：先用 `nsys`

先回答：

- 时间到底花在哪
- 是 host 侧同步、拷贝、分配更重，还是 kernel 更重
- NVTX 各阶段的时间比例是什么

### 5.3 第三步：再用 `ncu`

再回答：

- 某一个 kernel 为什么慢
- occupancy、throughput、launch 配置有没有问题
- GEMM 是 compute-bound 还是 memory-bound

## 6. Windows：`nsys`

### 6.1 最小可用命令

建议在管理员 PowerShell 中运行：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" profile `
  --sample=none `
  --cpuctxsw=none `
  --trace=cuda,nvtx `
  --force-overwrite=true `
  -o ".\nsys_2048_admin" `
  "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 2048 --reduction-size 2048 `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --warmup 1 --iters 1
```

### 6.2 常用 `stats` 命令

只看最有用的几张表：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" stats `
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum `
  .\nsys_2048_admin.nsys-rep
```

只看 `gemm` 这段：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" stats `
  --filter-nvtx gemm `
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum `
  .\nsys_2048_admin.nsys-rep
```

### 6.3 脚本入口

仓库里已有：

- [profile_nsys.ps1](/E:/learning/AI_system/scripts/profile_nsys.ps1:1)

例如：

```powershell
./scripts/profile_nsys.ps1 -DryRun
./scripts/profile_nsys.ps1 -VectorSize 2048 -ReductionSize 2048 -GemmM 2048 -GemmN 2048 -GemmK 2048 -Warmup 1 -Iters 1
```

## 7. Linux：`nsys`

下面是 Linux 命令模板，本次未验证。

### 7.1 最小可用命令

```bash
nsys profile \
  --sample=none \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  -o ./nsys_2048_linux \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 2048 --reduction-size 2048 \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --warmup 1 --iters 1
```

### 7.2 常用 `stats` 命令

```bash
nsys stats \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  ./nsys_2048_linux.nsys-rep
```

只看 `gemm`：

```bash
nsys stats \
  --filter-nvtx gemm \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  ./nsys_2048_linux.nsys-rep
```

### 7.3 脚本入口

仓库里已有：

- [profile_nsys.sh](/E:/learning/AI_system/scripts/profile_nsys.sh:1)

例如：

```bash
./scripts/profile_nsys.sh --dry-run
./scripts/profile_nsys.sh --vector-size 2048 --reduction-size 2048 --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 --warmup 1 --iters 1
```

## 8. `nsys` GUI 常用视图

### 8.1 Timeline View

最核心的主界面。常看这三层：

- `NVTX`
- `Threads -> CUDA API`
- `CUDA HW -> Kernels / Memory`

它适合回答：

- 哪个阶段最慢
- memcpy 和 kernel 是否交错
- host 什么时候在等 GPU

### 8.2 Stats System View

这是 GUI 里的统计报表中心，作用等价于 GUI 版 `nsys stats`。

常用场景：

- 快速看 `CUDA API Summary`
- 看 `CUDA GPU Kernel Summary`
- 看 `NVTX Range Summary`
- 抄数字、做汇报

它是聚合视图，不看时序，只看：

- 总时间
- 调用次数
- 平均耗时

### 8.3 Expert System View

这是自动规则诊断视图。

常用场景：

- 快速检查是不是有明显的同步问题
- 看有没有 GPU gaps
- 看 memcpy 用法是不是命中了常见坏味道

它更像性能巡检，不等于最终结论：

- 有提示时，要回到 timeline 和 stats 验证
- 没提示，不代表程序已经最优

## 9. Windows：`ncu`

### 9.1 单独抓一个 kernel，并导出 `.ncu-rep`

这也是最推荐的默认用法。

示例：只抓 `naive_gemm_kernel`

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe" `
  --set basic `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name regex:naive_gemm_kernel `
  --launch-count 1 `
  --export .\ncu_naive_gemm_2048 `
  "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 2048 --reduction-size 2048 `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --warmup 1 --iters 1
```

同理可以分别抓：

- `regex:reduce_sum_kernel`
- `regex:vector_add_kernel`
- `regex:cutlass_80_simt_sgemm`
- `regex:tensorop_h`
- `regex:tensorop_s`

### 9.2 一次抓多个热点 kernel，导出一个 `.ncu-rep`

如果你已经知道主要关注 GEMM 这几类 kernel，可以用一个 regex 抓一组：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe" `
  --set basic `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name "regex:naive_gemm_kernel|cutlass_80_simt_sgemm|tensorop_h|tensorop_s" `
  --launch-count 8 `
  --export .\docs\reports\ncu_gemm_group_2048 `
  "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 2048 --reduction-size 2048 `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --warmup 1 --iters 1
```

这种方式适合：

- 想把 GEMM 相关 kernel 放到一个报告里统一浏览
- 先整体摸底，再决定后面单独抓谁

### 9.3 一次抓这次运行中的所有 GPU kernel，导出一个 `.ncu-rep`

可以，做法就是：

- 不传 `--kernel-name`
- 不用 `--launch-count 1` 把结果限制死

命令示例：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe" `
  --set basic `
  --target-processes all `
  --kernel-name-base demangled `
  --export .\ncu_all_2048 `
  "E:\learning\AI_system\out\build\windows-vs2022-cuda-release\labs\perf_engineering\Release\perf_engineering_lab.exe" `
  --vector-size 2048 --reduction-size 2048 `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --warmup 1 --iters 1
```

说明：

- 这样会把这次运行中被 profile 到的多个 kernel 放进一个 `.ncu-rep`
- 它适合做“全量摸底”
- 但不适合作为日常默认方法，因为报告会变大，分析也会更杂

### 9.4 脚本入口

仓库里已有：

- [profile_ncu.ps1](/E:/learning/AI_system/scripts/profile_ncu.ps1:1)

例如：

```powershell
./scripts/profile_ncu.ps1 -DryRun
./scripts/profile_ncu.ps1 -KernelRegex naive_gemm_kernel -GemmM 2048 -GemmN 2048 -GemmK 2048 -Warmup 1 -Iters 1
```

## 10. Linux：`ncu`

下面是 Linux 命令模板，本次未验证。

### 10.1 单独抓一个 kernel

```bash
ncu \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name regex:naive_gemm_kernel \
  --launch-count 1 \
  --export ./ncu_naive_gemm_2048 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 2048 --reduction-size 2048 \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --warmup 1 --iters 1
```

### 10.2 一次抓多个热点 kernel

```bash
ncu \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:naive_gemm_kernel|cutlass_80_simt_sgemm|tensorop_h|tensorop_s' \
  --launch-count 8 \
  --export ./ncu_gemm_group_2048 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 2048 --reduction-size 2048 \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --warmup 1 --iters 1
```

### 10.3 一次抓全部 GPU kernel

```bash
ncu \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  --export ./ncu_all_2048 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 2048 --reduction-size 2048 \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --warmup 1 --iters 1
```

### 10.4 脚本入口

仓库里已有：

- [profile_ncu.sh](/E:/learning/AI_system/scripts/profile_ncu.sh:1)

例如：

```bash
./scripts/profile_ncu.sh --dry-run
./scripts/profile_ncu.sh --kernel-regex naive_gemm_kernel --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 --warmup 1 --iters 1
```

## 11. `ncu` 的最佳实践

### 11.1 默认优先单 kernel 抓取

最佳实践通常不是“一把抓全部”，而是：

1. 先用 `nsys` 找热点
2. 再用 `ncu` 单独抓热点 kernel

原因：

- `ncu` 为了拿指标会做 replay
- 抓的 kernel 越多，开销越大
- 一个报告里混太多 kernel，不利于对比

### 11.2 只有在摸底时才一次抓全部

一次抓全部适合：

- 第一次看一份 workload
- 想确认这次运行到底涉及哪些 kernel
- 为后续精确过滤准备样本

不适合：

- 作为日常默认命令
- 用来做最终性能对比报告

### 11.3 `--set basic` 作为第一步

推荐先从：

```text
--set basic
```

开始。先把：

- Speed Of Light
- Launch Statistics
- Occupancy

看懂，再逐渐加更重的 section。

### 11.4 profiling workload 要尽量缩小

建议：

- `--warmup 1 --iters 1`
- 用能复现问题的最小 shape
- 不要把 benchmark 的所有 sweep 一次都塞进 `ncu`

否则：

- 报告太大
- replay 太慢
- 干扰更重

### 11.5 `--kernel-name-base demangled`

推荐固定加上：

```text
--kernel-name-base demangled
```

这样：

- 过滤时更容易写 regex
- 输出名也更容易读

### 11.6 合理使用 `--launch-count` / `--launch-skip`

常见做法：

- `--launch-count 1`
  - 抓一个代表性 launch
- `--launch-skip N`
  - 跳过前面的 warmup 或不关心的 launch

如果 workload 中同一个 kernel 会重复很多次，这两个参数非常重要。

## 12. `ncu` 输出怎么看

先重点看这几组：

- `GPU Speed Of Light Throughput`
- `Launch Statistics`
- `Occupancy`
- `Memory Workload Analysis`
- `Compute Workload Analysis`

最常见的解读方式：

- `Compute Throughput` 高、`Memory Throughput` 一般：更偏 compute-bound
- `Memory Throughput` 高、`Compute Throughput` 一般：更偏 memory-bound
- `Waves Per SM` 很低：工作规模可能太小
- `Achieved Occupancy` 明显低于 `Theoretical Occupancy`：通常还有调度或负载不均衡问题

## 13. `nsys` / `ncu` 常见分工

### 用 `nsys` 回答

- 为什么 `e2e` 慢
- 是不是 memcpy / malloc / synchronize 很重
- NVTX 哪个阶段最重
- CPU 和 GPU 有没有明显等待关系

### 用 `ncu` 回答

- 为什么某个 GEMM kernel 慢
- occupancy 是否受寄存器/shared memory 限制
- throughput 是否接近设备能力
- 某个 Tensor Core kernel 到底有没有吃满

## 14. 当前仓库的落地建议

对 `perf_engineering_lab`，最推荐的实际操作是：

1. `nsys` 先抓一份 `2048` 或 `1024` 的 `gemm` workload
2. 在 GUI 或 `stats` 里确认热点 kernel
3. 用 `ncu` 依次单独抓：
   - `naive_gemm_kernel`
   - `cutlass_80_simt_sgemm`
   - `tensorop_h`
   - `tensorop_s`
4. 如果只是想摸底，再补一份 `ncu_all_2048.ncu-rep`

对后续 `labs/gemm` 的 `sgemm_v1/v2/wmma_demo` 也建议继续沿用：

- `nsys` 看端到端
- `ncu` 看单 kernel
- `e2e` / `kernel_only` 两套 benchmark 同时保留

