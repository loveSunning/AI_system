# AI_system

`AI_system` 是一个围绕你这份 12 个月学习排期表搭出来的跨平台实验仓库：用一个统一的 CMake 工程，把 CUDA kernel、GEMM/Triton/CUTLASS/CuTe、FlashAttention、PyTorch custom op、TVM/MLIR、RK3588 和 TPU-MLIR 的学习路径串成一个可逐月扩展的项目。

## 当前落地内容

- 跨 Windows / Linux 的 CMake 工程骨架。
- 4090 / 5060 两套显卡架构配置：
  - `RTX 4090 -> Ada Lovelace -> sm_89`
  - `RTX 5060 -> Blackwell -> sm_120`
- 首批性能工程样例：
  - `vector add`
  - `reduction`
  - `naive GEMM`
- 一个 CLI 工具，用于查看学习阶段和当前 GPU。
- 一个 smoke test，用于验证 CPU 与 CUDA 基础路径。
- 对应表格主题的目录占位和模块说明。

## 目录结构

```text
AI_system/
|-- accelerators/
|-- capstone/
|-- cmake/
|-- compilers/
|-- docs/
|-- edge/
|-- include/
|-- integrations/
|-- labs/
|-- scripts/
|-- src/
`-- tests/
```

## 快速开始

### Windows

```powershell
./scripts/configure.ps1 -Preset windows-vs2022-cuda-release -GpuProfile native
./scripts/build.ps1 -Preset windows-vs2022-cuda-release -Configuration Release
ctest --preset windows-vs2022-cuda-release
```

### Linux

```bash
./scripts/configure.sh linux-make-cuda-release native
./scripts/build.sh linux-make-cuda-release
ctest --preset linux-make-cuda-release
```

## 运行生成产物

Linux 下完成构建后，主要可执行文件位于 `out/build/linux-make-cuda-release/`。

### 使用 ai_system_cli

用于查看当前构建配置、学习阶段和本机 GPU 信息。

```bash
./out/build/linux-make-cuda-release/ai_system_cli --summary
./out/build/linux-make-cuda-release/ai_system_cli --list-plan
./out/build/linux-make-cuda-release/ai_system_cli --print-gpus
./out/build/linux-make-cuda-release/ai_system_cli --help
```

- `--summary`：输出工程版本、是否启用 CUDA、当前 GPU profile、编译时目标架构。
- `--list-plan`：输出 12 个月学习计划及对应目录。
- `--print-gpus`：输出当前进程可见的 CUDA GPU 和显存信息。
- `--help`：查看命令行帮助。

### 使用 perf_engineering_lab

用于对 `vector add`、`reduction`、`naive GEMM` 的 CPU / CUDA 路径做基准测试，并校验 GPU 结果是否和 CPU 基准一致。

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab --help
```

支持的参数：

- `--vector-size N`：设置 vector add 的元素个数。
- `--reduction-size N`：设置 reduction 的输入长度。
- `--gemm-m M`：设置 GEMM 输出矩阵行数。
- `--gemm-n N`：设置 GEMM 输出矩阵列数。
- `--gemm-k K`：设置 GEMM 的共享维度。
- `--warmup I`：设置每个 case 的预热轮数。
- `--iters I`：设置每个 case 的正式测量轮数。

示例：

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 1048576 \
  --reduction-size 1048576 \
  --gemm-m 128 --gemm-n 128 --gemm-k 128 \
  --warmup 2 --iters 6

./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab \
  --vector-size 16777216 \
  --reduction-size 16777216 \
  --gemm-m 1024 --gemm-n 1024 --gemm-k 1024 \
  --warmup 3 --iters 10
```

输出会分成两张表：

- `Benchmark Results`：包含 `op`、`impl`、`shape`、`avg_ms`、`min_ms`、`max_ms`、`perf`、`unit`、`warmup`、`iters`，便于横向对比 CPU / CUDA 的性能。
- `Validation`：包含 correctness 或 runtime 检查结果，避免把错误样本误当成性能结论。

### 如何测试不同尺寸输入的性能

推荐按下面的方式做尺寸扫描：

1. 先固定 `warmup` 和 `iters`，避免不同轮数影响结果对比。
2. 对 vector add 和 reduction 按 `2^k` 逐步放大输入长度，例如 `2^20`、`2^22`、`2^24`。
3. 对 GEMM 分别测试 `128x128x128`、`256x256x256`、`512x512x512`、`1024x1024x1024`。
4. 每次记录输出里的 `avg/min/max`，优先比较 `avg`，同时观察 `min/max` 是否抖动过大。

例如：

```bash
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab --vector-size 1048576 --reduction-size 1048576 --gemm-m 256 --gemm-n 256 --gemm-k 256 --warmup 2 --iters 8
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab --vector-size 4194304 --reduction-size 4194304 --gemm-m 512 --gemm-n 512 --gemm-k 512 --warmup 2 --iters 8
./out/build/linux-make-cuda-release/labs/perf_engineering/perf_engineering_lab --vector-size 16777216 --reduction-size 16777216 --gemm-m 1024 --gemm-n 1024 --gemm-k 1024 --warmup 2 --iters 8
```

### 如何判断结果是否准确

`perf_engineering_lab` 会先运行 CPU 版本作为基准，再运行 CUDA 版本，并在 `Validation` 表中汇总正确性检查结果：

- `status=PASS`：表示该算子的 CUDA 结果通过了 correctness 校验。
- `status=FAIL`：表示 correctness 校验失败，或 CUDA 路径运行失败。
- `detail`：会展示误差详情、CPU/GPU 结果，或 runtime 错误原因。

如果 `Validation` 表里出现 `status=FAIL`，或者程序返回非零退出码，说明当前实现或环境有问题，不应继续使用该次 benchmark 数据做性能结论。

建议的判断标准：

1. 只有在三个算子的 correctness 都是 `status=PASS` 时，才比较性能数据。
2. 如果某个 case 的 CUDA path 直接失败，先检查驱动、CUDA toolkit、目标架构和显存容量。
3. 对大尺寸 GEMM，如果显存不足，优先缩小 `m/n/k`，不要直接把失败结果当成性能退化。

## GPU 架构选择

项目通过 `AI_SYSTEM_GPU_PROFILE` 统一管理 CUDA 架构：

- `native`：根据本机 `nvidia-smi` 自动探测。
- `4090`：固定 `sm_89`。
- `5060`：固定 `sm_120`。
- `all`：同时编译 `sm_89;sm_120`。

也可以手动传入：

```text
-DAI_SYSTEM_GPU_PROFILE=89;120
```

## 推荐学习节奏

- 第 1 个月：`labs/perf_engineering`
- 第 2 个月：`labs/gemm`
- 第 3-4 个月：`labs/triton`
- 第 5 个月：`labs/cutlass`
- 第 6 个月：`labs/cute`
- 第 7 个月：`labs/flash_attention`
- 第 8 个月：`integrations/pytorch`
- 第 9-10 个月：`compilers/tvm` + `edge/rk3588`
- 第 11 个月：`compilers/mlir`
- 第 12 个月：`accelerators/tpu_mlir` + `capstone`

完整路线见 [docs/learning-roadmap.md](docs/learning-roadmap.md)。
