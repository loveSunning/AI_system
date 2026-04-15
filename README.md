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
