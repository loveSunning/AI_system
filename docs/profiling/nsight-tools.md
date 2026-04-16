# Nsight Systems 与 Nsight Compute 使用文档

本文档整理了本仓库在当前 CUDA 开发容器中的 Nsight 工具检查、安装、验证、基础使用和测试方法，重点面向 `perf_engineering_lab` 的 CUDA profiling。

## 适用环境

- 容器系统：Ubuntu 22.04
- CUDA 工具链：12.8
- 已验证 GPU：RTX 4090 D
- 已验证构建 preset：`linux-make-cuda-release`

## 工具位置与版本

当前容器中两个工具已经安装完成，无需额外安装。

### Nsight Systems

- 可执行文件：`/usr/local/cuda-12.8/bin/nsys`
- 版本：`2024.6.2.225`
- 所属包：`cuda-nsight-systems-12-8`

### Nsight Compute

- 可执行文件：`/usr/local/cuda-12.8/bin/ncu`
- 版本：`2025.1.1.0`
- 所属包：`cuda-nsight-compute-12-8`

## 检查是否已安装

先检查命令是否存在：

```bash
command -v nsys
command -v ncu
```

如果安装成功，应该输出可执行文件路径，例如：

```text
/usr/local/cuda-12.8/bin/nsys
/usr/local/cuda-12.8/bin/ncu
```

再检查版本：

```bash
nsys --version
ncu --version
```

本容器中的验证结果：

```text
NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0

NVIDIA (R) Nsight Compute Command Line Profiler
Version 2025.1.1.0
```

## 如果容器里没有这两个工具，如何安装

本容器不需要执行这一步，但如果你在其他 Ubuntu + CUDA 12.8 环境中缺少工具，可以安装：

```bash
sudo apt-get update
sudo apt-get install -y cuda-nsight-systems-12-8 cuda-nsight-compute-12-8
```

如果你使用的是其他 CUDA 版本，先搜索可用包名：

```bash
apt-cache search nsight
```

常见规则：

- CUDA 12.8 对应：`cuda-nsight-systems-12-8`、`cuda-nsight-compute-12-8`
- CUDA 12.7 对应：`cuda-nsight-systems-12-7`、`cuda-nsight-compute-12-7`
- CUDA 12.6 对应：`cuda-nsight-systems-12-6`、`cuda-nsight-compute-12-6`

安装后建议重新执行版本检查。

## 使用前准备

确保项目已经使用 CUDA preset 配置并构建完成：

```bash
./scripts/configure.sh linux-make-cuda-release native
./scripts/build.sh linux-make-cuda-release
ctest --preset linux-make-cuda-release --output-on-failure
```

本文档中的 profiling 示例默认使用下面这个程序：

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab
```

它支持以下关键参数：

- `--vector-size N`
- `--reduction-size N`
- `--gemm-m M`
- `--gemm-n N`
- `--gemm-k K`
- `--warmup I`
- `--iters I`

注意：

- `--warmup` 和 `--iters` 必须是正整数，不能传 `0`
- 太小的输入尺寸会导致 GPU 利用率非常低，profiling 更适合看机制，不适合下性能结论
- 太大的 GEMM 可能受显存容量限制

## 最小化运行验证

先确认 benchmark 本身可运行：

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 \
  --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

如果程序运行成功，输出中应包含两张表：

- `Benchmark Results`
- `Validation`

并且 `Validation` 中三个 CUDA correctness 检查应为 `PASS`。

## Nsight Systems 使用

`nsys` 用来观察整体时间线、CUDA API 调用、CPU/GPU 协同关系和各阶段耗时分布。

### 1. 查看版本

```bash
nsys --version
```

### 2. 最小 profiling 命令

```bash
nsys profile --sample=none --trace=cuda,nvtx,osrt -o /tmp/nsys-ai-system-test \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

该命令会生成报告文件：

```text
/tmp/nsys-ai-system-test.nsys-rep
```

本仓库已验证该命令可以成功执行并生成 `.nsys-rep` 报告。

### 3. 查看报告摘要

```bash
nsys stats /tmp/nsys-ai-system-test.nsys-rep
```

### 4. 常用命令模板

分析更大尺寸输入：

```bash
nsys profile --sample=none --trace=cuda,nvtx,osrt -o /tmp/nsys-gemm-256 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1048576 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --warmup 2 --iters 5
```

更关注 CUDA 时间线时，可以缩小 trace 范围：

```bash
nsys profile --sample=none --trace=cuda,osrt -o /tmp/nsys-cuda-only \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1048576 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --warmup 2 --iters 5
```

### 5. 什么时候用 Nsight Systems

适合看：

- 整体端到端时间线
- kernel launch 是否频繁过碎
- CUDA API 耗时
- Host 端是否阻塞 GPU
- 多段 workload 之间是否存在空洞

不适合直接回答：

- 单个 kernel 为何 occupancy 低
- 某个 kernel 的 DRAM throughput 是否被打满
- 某个 kernel 的寄存器或共享内存配置是否不合理

这些问题更适合 `ncu`。

## Nsight Compute 使用

`ncu` 用来分析单个 kernel 的微观性能指标，例如 occupancy、throughput、launch 配置、访存与算力利用率。

### 1. 查看版本

```bash
ncu --version
```

### 2. 最小 profiling 命令

```bash
ncu --set basic --launch-count 1 --target-processes all \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

本仓库已验证该命令可以成功抓取 `vector_add_kernel` 的 profiling 信息，并输出：

- `GPU Speed Of Light Throughput`
- `Launch Statistics`
- `Occupancy`
- `GPU and Memory Workload Distribution`

### 3. 只分析指定 kernel

```bash
ncu --set basic --kernel-name regex:vector_add_kernel --launch-count 1 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

分析 reduction kernel：

```bash
ncu --set basic --kernel-name regex:reduce_sum_kernel --launch-count 1 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1048576 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

分析 GEMM kernel：

```bash
ncu --set basic --kernel-name regex:naive_gemm_kernel --launch-count 1 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1024 \
  --gemm-m 512 --gemm-n 512 --gemm-k 512 \
  --warmup 1 --iters 1
```

### 4. 导出报告

```bash
ncu --set basic --export /tmp/ncu-gemm-256 --launch-count 1 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1048576 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --warmup 2 --iters 3
```

### 5. 什么时候用 Nsight Compute

适合看：

- occupancy 是否偏低
- grid / block 配置是否导致 SM 利用率过低
- DRAM、L2、L1/TEX throughput 是否逼近瓶颈
- 某个 kernel 是否受 launch 配置限制

例如本仓库对小尺寸 `vector_add_kernel` 的验证中，`ncu` 给出了如下结论：

- grid 太小，无法填满 4090 D 的全部 SM
- Waves Per SM 很低
- Achieved Occupancy 明显低于理论 occupancy

这说明小尺寸测试适合验证功能和工具链，但不适合评价 GPU 峰值性能。

## 建议的测试流程

### 阶段 1：先做功能验证

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

通过标准：

- 程序返回码为 `0`
- `Validation` 表中各项 `status=PASS`

### 阶段 2：再做系统级 profiling

```bash
nsys profile --sample=none --trace=cuda,nvtx,osrt -o /tmp/nsys-validation \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1048576 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --warmup 2 --iters 5
```

目标：

- 观察整体时间线
- 判断 CPU / GPU 是否存在明显空转
- 判断 launch 是否过于碎片化

### 阶段 3：最后做单 kernel 诊断

```bash
ncu --set basic --kernel-name regex:naive_gemm_kernel --launch-count 1 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1024 \
  --gemm-m 512 --gemm-n 512 --gemm-k 512 \
  --warmup 1 --iters 1
```

目标：

- 看 GEMM kernel 的 occupancy
- 看吞吐和 launch 配置
- 判断是否是 block 配置、访存模式或计算密度的问题

## 如何判断 profiling 结果可靠

只有在下列条件同时满足时，profiling 结果才适合拿来做性能分析：

1. benchmark 程序自身运行成功
2. `Validation` 表中的 correctness 检查全部为 `PASS`
3. 采集时使用的输入尺寸足够大，能够让 GPU 有稳定负载
4. `warmup` 与 `iters` 设置合理，避免只看单次偶然值

不建议直接用下面这类场景的结果做最终结论：

- `N=1024` 的 vector add
- `32x32x32` 的 naive GEMM
- correctness 失败的样本
- 显存不足导致 fallback 或运行失败的样本

## 故障排查

### 1. `command not found: nsys` 或 `ncu`

说明工具未安装，按上文的 apt 命令安装即可。

### 2. `ncu` 或 `nsys` 能运行，但程序 profiling 失败

先单独运行 benchmark，确认程序本身可执行：

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab --help
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1024 \
  --gemm-m 32 --gemm-n 32 --gemm-k 32 \
  --warmup 1 --iters 1
```

### 3. 传了 `--warmup 0` 或 `--iters 0`

当前程序会直接报错，必须传正整数。

### 4. 小尺寸下 GPU 利用率很差

这是正常现象。先增大输入规模，再看 Nsight 指标是否改善。

### 5. 大尺寸 GEMM 失败

优先减小 `m`、`n`、`k`，并确认 GPU 显存是否足够。

## 推荐命令速查

### 检查工具

```bash
command -v nsys
command -v ncu
nsys --version
ncu --version
```

### 跑 benchmark

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1048576 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --warmup 2 --iters 5
```

### 跑 Nsight Systems

```bash
nsys profile --sample=none --trace=cuda,nvtx,osrt -o /tmp/nsys-run \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 --reduction-size 1048576 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --warmup 2 --iters 5
```

### 跑 Nsight Compute

```bash
ncu --set basic --kernel-name regex:naive_gemm_kernel --launch-count 1 \
  ./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1024 --reduction-size 1024 \
  --gemm-m 512 --gemm-n 512 --gemm-k 512 \
  --warmup 1 --iters 1
```