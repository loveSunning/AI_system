# W21 CUTLASS Profiler 参数扫描：Step by Step

本文给出一套可以直接执行的 W21 实验流程，覆盖：

- Windows 10/11 + RTX 5060 + `sm_120`；
- Linux + RTX 4090D + `sm_89`；
- `M=N=K=4096`；
- FP16 A/B、FP32 accumulator、FP32 C/D；
- CTA tile、warp shape/count、stages、alignment、split-K；
- 可选的 swizzle/raster order。

本文使用仓库中的 CUTLASS 4.5.2。Profiler 是 CUTLASS 自己的独立 CMake
工程，不是本仓库 `out/build/...` 中的学习示例目标。

## 0. 先理解 Profiler 的工作方式

`--cta_m`、`--stages`、`--warps_m` 等参数主要用于筛选已经编译进
`cutlass_profiler` 的 kernel，不会在运行时临时生成一个新 kernel。因此正确顺序是：

```text
配置 CUTLASS operation library
  -> 编译一组候选 kernel
  -> enumerate/dry_run 确认 kernel 存在
  -> 正确性验证
  -> 性能扫描
  -> 汇总 CSV
```

如果 CMake 没有设置 `CUTLASS_LIBRARY_KERNELS`，CUTLASS 默认可能只生成最大的
tile，无法完成至少六种 shape 的扫描。本文编译下面这个公共 kernel 家族：

```text
cutlass_tensorop_s16816gemm_f16_*
```

名字中的关键信息：

```text
s16816       FP16 Tensor Core MMA m16n8k16，FP32 accumulator
gemm_f16     FP16 A/B GEMM
128x128      CTA M/N tile
32x3         CTA K tile=32，stages=3
tn           A=row-major，B=column-major
align8       A/B 每次访问 8 个 FP16，即 128 bit
```

例如：

```text
cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8
```

## 1. 固定实验条件

主扫描固定以下条件，避免同时改变太多变量：

| 项目 | 固定值 |
| --- | --- |
| Problem | `M=N=K=4096` |
| A | `f16:row` |
| B | `f16:column` |
| C/D | `f32:column` |
| Accumulator | `f32` |
| Op class | `tensorop` |
| Instruction | `16x8x16` |
| Alpha/Beta | `1/0` |
| Batch | 1 |
| Warmup | 10 次 |
| Profiling | 每个 kernel 50 次 |

主实验保持 `m16n8k16` 不变。把 instruction shape 写入 CSV，但不要为了得到其他
instruction shape 而把 SIMT kernel 混入 Tensor Core 排名。

W20 手写示例使用 row-major C/D；这里改用 column-major C/D，是因为 CUTLASS 4.5.2
的 `GenerateSM80_TensorOp_16816` 官方 Profiler family 固定实例化 column-major C/D。
两组实验的 A/B layout 和 Tensor Core mainloop 相同，但 epilogue layout 不同。报告中
必须记录这个差异，不能把 W20 可执行文件与 W21 Profiler 的绝对时间当成完全同配置对比。

## 2. Step 1：检查环境

### 2.1 Windows / RTX 5060

在 PowerShell 中执行：

```powershell
cd D:\workspace\learing\AI_system

nvidia-smi
nvcc --version
cmake --version
python --version
nvcc --list-gpu-code | Select-String sm_120
```

必须确认：

- GPU 是 RTX 5060；
- CUDA 工具链能够生成 `sm_120`；
- 已安装 Visual Studio 2022 C++ workload；
- CUTLASS 位于 `D:\workspace\learing\AI_system\3rdparty\cutlass`。

RTX 5060 使用 `120`，不是 `120a`。`120a` 包含 architecture-accelerated feature
约束，不是本实验公共 FP16 `mma.sync` 路径所需的目标。

### 2.2 Linux / RTX 4090D

```bash
cd /path/to/AI_system

nvidia-smi
nvcc --version
cmake --version
python3 --version
nvcc --list-gpu-code | grep sm_89
```

必须确认 GPU 是 RTX 4090D，CUDA 工具链能够生成 `sm_89`。

## 3. Step 2：配置并编译 CUTLASS Profiler

### 3.1 Windows / RTX 5060 / SM120

#### 推荐：使用仓库脚本

```powershell
cd D:\workspace\learing\AI_system

.\labs\cutlass\scripts\configure_official_cutlass.ps1
.\labs\cutlass\scripts\build_official_cutlass.ps1
```

默认构建目录：

```text
3rdparty/cutlass/build/windows-vs2022-5060
```

Profiler 路径：

```text
3rdparty/cutlass/build/windows-vs2022-5060/tools/profiler/Release/cutlass_profiler.exe
```

#### 等价的完整 CMake 指令

```powershell
cd D:\workspace\learing\AI_system

$cutlass = Resolve-Path ".\3rdparty\cutlass"
$build = Join-Path $cutlass "build\windows-vs2022-5060"

cmake `
  -S $cutlass `
  -B $build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCUTLASS_NVCC_ARCHS=120 `
  -DCUTLASS_ENABLE_TESTS=OFF `
  -DCUTLASS_ENABLE_CUBLAS=ON `
  -DCUTLASS_LIBRARY_OPERATIONS=gemm `
  "-DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_f16_*" `
  "-DCUTLASS_LIBRARY_IGNORE_KERNELS=cutlass_tensorop_s16816gemm_f16_s8_*,cutlass_tensorop_s16816gemm_f16_u8_*" `
  -DCUTLASS_PROFILER_DISABLE_REFERENCE=ON `
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

cmake --build $build `
  --config Release `
  --target cutlass_profiler `
  --parallel
```

`CUTLASS_PROFILER_DISABLE_REFERENCE=ON` 关闭 CUTLASS 自带的 host/device reference
kernel，避免为本实验编译不需要的 INT4 reference 路径。它不会关闭被测 CUTLASS
kernel。命令仍请求 cuBLAS 验证，但必须以最终 `Disposition` 判断 cuBLAS 是否真的运行。

### 3.2 Linux / RTX 4090D / SM89

#### 推荐：使用仓库脚本

```bash
cd /path/to/AI_system

bash ./labs/cutlass/scripts/configure_official_cutlass.sh
bash ./labs/cutlass/scripts/build_official_cutlass.sh
```

默认构建目录：

```text
3rdparty/cutlass/build/linux-4090d
```

Profiler 路径：

```text
3rdparty/cutlass/build/linux-4090d/tools/profiler/cutlass_profiler
```

#### 等价的完整 CMake 指令

```bash
cd /path/to/AI_system

CUTLASS_ROOT="$PWD/3rdparty/cutlass"
BUILD_DIR="$CUTLASS_ROOT/build/linux-4090d"

cmake \
  -S "$CUTLASS_ROOT" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=89 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_CUBLAS=ON \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  '-DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_f16_*' \
  '-DCUTLASS_LIBRARY_IGNORE_KERNELS=cutlass_tensorop_s16816gemm_f16_s8_*,cutlass_tensorop_s16816gemm_f16_u8_*' \
  -DCUTLASS_PROFILER_DISABLE_REFERENCE=ON \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

cmake --build "$BUILD_DIR" \
  --target cutlass_profiler \
  --parallel "$(nproc)"
```

## 4. Step 3：设备、帮助与 kernel inventory

### 4.1 Windows

```powershell
cd D:\workspace\learing\AI_system

$profiler = Resolve-Path ".\3rdparty\cutlass\build\windows-vs2022-5060\tools\profiler\Release\cutlass_profiler.exe"
$resultDir = ".\out\cutlass\w21-sm120"
New-Item -ItemType Directory -Force $resultDir | Out-Null

# Visual Studio 不会把 cutlass.dll 自动复制到 profiler.exe 旁边。
$cutlassDllDir = Resolve-Path ".\3rdparty\cutlass\build\windows-vs2022-5060\tools\library\Release"
$cudaDllDir = Join-Path $env:CUDA_PATH "bin\x64"
$env:PATH = "$cutlassDllDir;$cudaDllDir;$env:PATH"

& $profiler --version
& $profiler --device-info |
  Tee-Object -FilePath "$resultDir\device_info.txt"
& $profiler --operation=Gemm --help |
  Out-File -Encoding utf8 "$resultDir\gemm_help.txt"
& $profiler `
  --mode=enumerate `
  --operation=Gemm `
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_*' |
  Tee-Object -FilePath "$resultDir\kernel_inventory.txt"
```

### 4.2 Linux

```bash
cd /path/to/AI_system

PROFILER="$PWD/3rdparty/cutlass/build/linux-4090d/tools/profiler/cutlass_profiler"
RESULT_DIR="$PWD/out/cutlass/w21-sm89"
mkdir -p "$RESULT_DIR"

"$PROFILER" --version
"$PROFILER" --device-info | tee "$RESULT_DIR/device_info.txt"
"$PROFILER" --operation=Gemm --help > "$RESULT_DIR/gemm_help.txt"
"$PROFILER" \
  --mode=enumerate \
  --operation=Gemm \
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_*' \
  | tee "$RESULT_DIR/kernel_inventory.txt"
```

检查 `kernel_inventory.txt`。如果只有一种 CTA tile，说明之前配置 build tree 时
没有启用本文的 `CUTLASS_LIBRARY_KERNELS`，需要重新执行 Step 2 的 configure 和 build。
当前配置应生成 264 个完整 family 实例；限定 `tn` layout 后应看到 66 个候选，覆盖
22 个 tile/stages 组合和 `align8/align4/align2`。具体数量以当前 CUTLASS 版本为准。

## 5. Step 4：建立公共参数并做 smoke test

### 5.1 Windows

```powershell
$common = @(
  '--operation=Gemm'
  '--providers=cutlass'
  '--m=4096'
  '--n=4096'
  '--k=4096'
  '--A=f16:row'
  '--B=f16:column'
  '--C=f32:column'
  '--D=f32:column'
  '--accum=f32'
  '--op_class=tensorop'
  '--inst_m=16'
  '--inst_n=8'
  '--inst_k=16'
  '--alpha=1'
  '--beta=0'
  '--verification-enabled=true'
  '--verification-providers=cublas'
  '--warmup-iterations=10'
  '--profiling-iterations=20'
)

& $profiler @common `
  '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' `
  --cta_m=128 --cta_n=128 --cta_k=32 `
  --warps_m=2 --warps_n=2 --warps_k=1 `
  --stages=3 `
  "--output=$resultDir\smoke"

if ($LASTEXITCODE -ne 0) { throw "CUTLASS Profiler smoke test failed" }
```

也可以使用仓库包装脚本完成基础 smoke：

```powershell
.\labs\cutlass\scripts\run_profiler.ps1 `
  -M 4096 -N 4096 -K 4096 `
  -WarmupIterations 10 `
  -ProfilingIterations 20
```

### 5.2 Linux

```bash
COMMON=(
  --operation=Gemm
  --providers=cutlass
  --m=4096 --n=4096 --k=4096
  --A=f16:row --B=f16:column
  --C=f32:column --D=f32:column
  --accum=f32
  --op_class=tensorop
  --inst_m=16 --inst_n=8 --inst_k=16
  --alpha=1 --beta=0
  --verification-enabled=true
  --verification-providers=cublas
  --warmup-iterations=10
  --profiling-iterations=20
)

"$PROFILER" "${COMMON[@]}" \
  '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' \
  --cta_m=128 --cta_n=128 --cta_k=32 \
  --warps_m=2 --warps_n=2 --warps_k=1 \
  --stages=3 \
  --output="$RESULT_DIR/smoke"
```

包装脚本：

```bash
bash ./labs/cutlass/scripts/run_profiler.sh \
  --m 4096 --n 4096 --k 4096 \
  --warmup-iterations 10 \
  --profiling-iterations 20
```

`Status=Success` 只表示 CUTLASS kernel 成功执行；只有 `Disposition=Passed` 才表示
reference 校验真正执行并通过。若显示 `Disposition=Not verified` 和 `cuBLAS: Not run`，
该行可以用于性能测量，但不能作为正确性通过的证据。需要严格校验时增加：

```text
--verification-required=true
```

这样没有任何 reference provider 实际运行时，Profiler 会把它视为错误。CUDA 13 +
CUTLASS 4.5.2 的 cuBLASLt default algorithm 在部分 SM120 环境可能返回 `Not run`；
此时参照 `windows-linux-build.md` 的 Full reference profiler build，启用 CUTLASS
host/device reference，再使用 `--verification-providers=device`。

Profiler 会在 `--output` 给定的前缀后追加 operation kind 和 `.csv`；实际文件通常是
`smoke.gemm.csv`，不要因为没有看到恰好名为 `smoke` 的文件而误判失败。

## 6. Step 5：先 dry-run，再正式 profile

在任何大扫描前，将相同命令临时加上：

```text
--mode=dry_run
```

dry-run 不启动 kernel、不分配 workspace，用于确认筛选条件能匹配 kernel。例如：

```powershell
& $profiler @common --mode=dry_run `
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_align8' `
  --cta_m=128 --cta_n=128 --cta_k=32 `
  --warps_m=2 --warps_n=2 --warps_k=1 --stages=4
```

```bash
"$PROFILER" "${COMMON[@]}" --mode=dry_run \
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_align8' \
  --cta_m=128 --cta_n=128 --cta_k=32 \
  --warps_m=2 --warps_n=2 --warps_k=1 --stages=4
```

匹配 0 个 kernel 不是性能结果，应标记为 `NotBuilt/Unsupported`，不能记成 0
TFLOP/s。Profiler 在某些 0-match 情况下仍可能返回退出码 0；如果输出 CSV 只有表头
而没有数据行，也必须判定为失败。

## 7. Step 6：扫描 CTA tile 与 warp count/shape

当前 CUTLASS 4.5.2 的 SM80 FP16 generator 提供以下典型候选。warp shape 由
`CTA shape / warp count` 推导：

| ID | CTA tile | Stages | Warp count | 推导出的 warp shape |
| --- | --- | ---: | --- | --- |
| S1 | 256x128x32 | 3 | 4x2x1 | 64x64x32 |
| S2 | 128x256x32 | 3 | 2x4x1 | 64x64x32 |
| S3 | 256x64x32 | 3 | 4x1x1 | 64x64x32 |
| S4 | 64x256x32 | 4 | 1x4x1 | 64x64x32 |
| S5 | 128x128x32 | 3 | 2x2x1 | 64x64x32 |
| S6 | 128x64x32 | 6 | 2x2x1 | 64x32x32 |
| S7 | 64x128x32 | 6 | 2x2x1 | 32x64x32 |
| S8 | 64x64x32 | 10 | 2x2x1 | 32x32x32 |

这里 stages 不能对所有 CTA shape 固定为 3，因为官方 generator 并没有为每个
shape 生成相同 stage 数。这一阶段比较的是“CUTLASS 真实提供的完整配置”，不是纯粹
隔离 stages 变量；纯 stages 对比放在下一步。

### 7.1 Windows

```powershell
$configs = @(
  @{ Id='S1'; M=256; N=128; K=32; Stages=3;  WM=4; WN=2; WK=1 }
  @{ Id='S2'; M=128; N=256; K=32; Stages=3;  WM=2; WN=4; WK=1 }
  @{ Id='S3'; M=256; N=64;  K=32; Stages=3;  WM=4; WN=1; WK=1 }
  @{ Id='S4'; M=64;  N=256; K=32; Stages=4;  WM=1; WN=4; WK=1 }
  @{ Id='S5'; M=128; N=128; K=32; Stages=3;  WM=2; WN=2; WK=1 }
  @{ Id='S6'; M=128; N=64;  K=32; Stages=6;  WM=2; WN=2; WK=1 }
  @{ Id='S7'; M=64;  N=128; K=32; Stages=6;  WM=2; WN=2; WK=1 }
  @{ Id='S8'; M=64;  N=64;  K=32; Stages=10; WM=2; WN=2; WK=1 }
)

foreach ($cfg in $configs) {
  & $profiler @common `
    '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_align8' `
    "--cta_m=$($cfg.M)" "--cta_n=$($cfg.N)" "--cta_k=$($cfg.K)" `
    "--warps_m=$($cfg.WM)" "--warps_n=$($cfg.WN)" "--warps_k=$($cfg.WK)" `
    "--stages=$($cfg.Stages)" `
    "--tags=experiment:cta_warp,config_id:$($cfg.Id),gpu:sm120" `
    "--output=$resultDir\cta_warp_$($cfg.Id)"
  if ($LASTEXITCODE -ne 0) { throw "CTA/warp scan failed: $($cfg.Id)" }
}
```

### 7.2 Linux

```bash
CONFIGS=(
  'S1 256 128 32 3 4 2 1'
  'S2 128 256 32 3 2 4 1'
  'S3 256 64  32 3 4 1 1'
  'S4 64  256 32 4 1 4 1'
  'S5 128 128 32 3 2 2 1'
  'S6 128 64  32 6 2 2 1'
  'S7 64  128 32 6 2 2 1'
  'S8 64  64  32 10 2 2 1'
)

for row in "${CONFIGS[@]}"; do
  read -r id cta_m cta_n cta_k stages warps_m warps_n warps_k <<< "$row"
  "$PROFILER" "${COMMON[@]}" \
    '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_align8' \
    --cta_m="$cta_m" --cta_n="$cta_n" --cta_k="$cta_k" \
    --warps_m="$warps_m" --warps_n="$warps_n" --warps_k="$warps_k" \
    --stages="$stages" \
    --tags="experiment:cta_warp,config_id:$id,gpu:sm89" \
    --output="$RESULT_DIR/cta_warp_$id"
done
```

## 8. Step 7：隔离扫描 stages

`128x128x32 / warp count 2x2x1` 同时存在 stages 3、4、5，适合做控制变量实验。

### Windows

```powershell
foreach ($stages in 3, 4, 5) {
  & $profiler @common `
    '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_align8' `
    --cta_m=128 --cta_n=128 --cta_k=32 `
    --warps_m=2 --warps_n=2 --warps_k=1 `
    "--stages=$stages" `
    "--tags=experiment:stages,stages:$stages,gpu:sm120" `
    "--output=$resultDir\stages_$stages"
}
```

### Linux

```bash
for stages in 3 4 5; do
  "$PROFILER" "${COMMON[@]}" \
    '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_align8' \
    --cta_m=128 --cta_n=128 --cta_k=32 \
    --warps_m=2 --warps_n=2 --warps_k=1 \
    --stages="$stages" \
    --tags="experiment:stages,stages:$stages,gpu:sm89" \
    --output="$RESULT_DIR/stages_$stages"
done
```

重点分析：更多 stages 是否改善 `cp.async` 与 MMA 重叠，以及 shared memory 增长是否
降低可驻留 CTA 数。不要预设 stages 越多越快。

## 9. Step 8：扫描 alignment

这个 kernel generator 提供 `align8/align4/align2`，不提供同一家族的 `align1`。
Profiler 没有独立 `--alignment` 参数，alignment 通过 kernel 名字筛选。

### Windows

```powershell
foreach ($alignment in 8, 4, 2) {
  & $profiler @common `
    "--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_align$alignment" `
    --cta_m=128 --cta_n=128 --cta_k=32 `
    --warps_m=2 --warps_n=2 --warps_k=1 `
    --stages=3 `
    "--tags=experiment:alignment,alignment:$alignment,gpu:sm120" `
    "--output=$resultDir\alignment_$alignment"
}
```

### Linux

```bash
for alignment in 8 4 2; do
  "$PROFILER" "${COMMON[@]}" \
    --kernels="cutlass_tensorop_s16816gemm_f16_*_tn_align$alignment" \
    --cta_m=128 --cta_n=128 --cta_k=32 \
    --warps_m=2 --warps_n=2 --warps_k=1 \
    --stages=3 \
    --tags="experiment:alignment,alignment:$alignment,gpu:sm89" \
    --output="$RESULT_DIR/alignment_$alignment"
done
```

4096 和 CUDA 分配地址满足 `align8`。如果 `align8` 更快，重点检查向量化 load、
内存指令数量和 `cp.async` 搬运宽度；不能只写“对齐更好”。

## 10. Step 9：扫描 split-K

先在基准 kernel 上比较 serial split-K：

### Windows

```powershell
foreach ($slices in 1, 2, 4, 8) {
  & $profiler @common `
    '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' `
    --split_k_mode=serial "--split_k_slices=$slices" `
    "--tags=experiment:split_k_serial,slices:$slices,gpu:sm120" `
    "--output=$resultDir\split_k_serial_$slices"
}

foreach ($slices in 2, 4, 8) {
  & $profiler @common `
    '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' `
    --split_k_mode=parallel "--split_k_slices=$slices" `
    "--tags=experiment:split_k_parallel,slices:$slices,gpu:sm120" `
    "--output=$resultDir\split_k_parallel_$slices"
}
```

### Linux

```bash
for slices in 1 2 4 8; do
  "$PROFILER" "${COMMON[@]}" \
    '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' \
    --split_k_mode=serial --split_k_slices="$slices" \
    --tags="experiment:split_k_serial,slices:$slices,gpu:sm89" \
    --output="$RESULT_DIR/split_k_serial_$slices"
done

for slices in 2 4 8; do
  "$PROFILER" "${COMMON[@]}" \
    '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' \
    --split_k_mode=parallel --split_k_slices="$slices" \
    --tags="experiment:split_k_parallel,slices:$slices,gpu:sm89" \
    --output="$RESULT_DIR/split_k_parallel_$slices"
done
```

`4096x4096` 输出和 `128x128` CTA 已产生 `32x32=1024` 个输出 CTA，通常不缺
并行度。预期 split-K 可能因为 workspace、同步和额外 reduction 而变慢。这个负面
结果也是本周需要解释的结论。

## 11. Step 10：可选 swizzle/raster order

先执行下面的 dry-run：

```powershell
& $profiler @common --mode=dry_run `
  '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' `
  --raster_order=along_m --swizzle_size=2
```

```bash
"$PROFILER" "${COMMON[@]}" --mode=dry_run \
  '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' \
  --raster_order=along_m --swizzle_size=2
```

如果目标 kernel 支持运行时 tile remapping，再扫描：

```text
raster_order = along_m, along_n
swizzle_size = 1, 2, 4, 8
```

Windows 示例：

```powershell
foreach ($order in 'along_m', 'along_n') {
  foreach ($swizzle in 1, 2, 4, 8) {
    & $profiler @common `
      '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' `
      "--raster_order=$order" "--swizzle_size=$swizzle" `
      "--tags=experiment:raster,raster:$order,swizzle:$swizzle,gpu:sm120" `
      "--output=$resultDir\raster_${order}_$swizzle"
  }
}
```

Linux 示例：

```bash
for order in along_m along_n; do
  for swizzle in 1 2 4 8; do
    "$PROFILER" "${COMMON[@]}" \
      '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8' \
      --raster_order="$order" --swizzle_size="$swizzle" \
      --tags="experiment:raster,raster:$order,swizzle:$swizzle,gpu:sm89" \
      --output="$RESULT_DIR/raster_${order}_$swizzle"
  done
done
```

本实验的 `cutlass_tensorop_s16816...` 是 SM80-compatible legacy kernel 家族。
Profiler 接受参数不等于 kernel 一定实际使用动态 raster/swizzle。若所有结果完全一致，
或者 operation descriptor 不支持该能力，应在报告中标记为“不适用于该 kernel 家族”，
不要制造优化结论。该扩展更适合支持 runtime tile scheduler 的 CUTLASS 3.x kernel。

## 12. Step 11：正式复测协议

初筛完成后，对每张 GPU 的前五名配置执行：

```text
warmup-iterations=10
profiling-iterations=50
独立重复 5 次
```

建议每次重复写入不同 CSV，通过 `--tags=repeat_id:<n>` 标记。最终比较中位数，至少
计算：

```text
median runtime
median GFLOP/s
min/max
标准差
相对基准百分比
```

SM89 和 SM120 分开排名。最终可以合并 CSV，但必须保留：

```text
gpu_name, compute_capability, cuda_version, cutlass_version
```

不要把两张卡的绝对 TFLOP/s 放在同一个无分组排名中。

## 13. CSV 与报告要求

Profiler 的每个 raw CSV 至少保留以下原始字段：

```text
Operation, Status, Disposition
m, n, k
cta_m, cta_n, cta_k
warps_m, warps_n, warps_k
inst_m, inst_n, inst_k
stages
split_k_mode, split_k_slices
Runtime, GFLOPs
```

汇总 `sweep_results.csv` 时增加：

```text
experiment_id, config_id, repeat_id
gpu_name, compute_capability
alignment_a, alignment_b
warp_shape_m = cta_m / warps_m
warp_shape_n = cta_n / warps_n
warp_shape_k = cta_k / warps_k
total_warps = warps_m * warps_n * warps_k
threads_per_cta = total_warps * 32
grid_ctas = ceil(M/cta_m) * ceil(N/cta_n) * split_k_slices
relative_to_baseline
```

报告中必须区分：

- Profiler 实测事实；
- 从 tile 参数计算出的派生值；
- 对寄存器、shared memory、occupancy、cache 的机制假设；
- 需要 Nsight Compute 才能验证的结论。

## 14. 最终验收清单

- [ ] Windows Profiler 使用 `CUTLASS_NVCC_ARCHS=120` 编译成功；
- [ ] Linux Profiler 使用 `CUTLASS_NVCC_ARCHS=89` 编译成功；
- [ ] 两个平台都保存 `device_info.txt` 和 `kernel_inventory.txt`；
- [ ] smoke test 的 `Status=Success`；严格正确性验收必须 `Disposition=Passed`；
- [ ] 至少扫描六种真实存在的 CTA/warp 配置；
- [ ] 对 `128x128x32` 隔离比较 stages 3/4/5；
- [ ] 比较 alignment 8/4/2；
- [ ] 比较 serial/parallel split-K 和 slices 1/2/4/8；
- [ ] 可选 raster/swizzle 明确记录 kernel 是否真正支持；
- [ ] 最终候选独立重复五次并使用中位数；
- [ ] SM89、SM120 分开排名；
- [ ] 输出 `sweep_results.csv` 和配置选择报告。

## 15. 常见错误

### 找不到任何 kernel

原因通常是 build tree 没有按 Step 2 重新 configure，或过滤条件组合在 generator 中
不存在。先用 `--mode=enumerate` 和 `--mode=dry_run`，不要直接扩大扫描范围。

### Windows 编译时间很长

Profiler 会实例化大量模板。本文已经限制为 GEMM 和一个 FP16 TensorOp kernel
家族，并启用 unity build。不要使用 `CUTLASS_LIBRARY_KERNELS=all`。

### Windows 启动返回 `0xC0000135`

这是 DLL 搜索失败。`cutlass_profiler.exe` 依赖构建目录中的 `cutlass.dll` 和 CUDA
目录中的 cuBLAS DLL。按 Step 3 把下面两个目录加入当前 PowerShell 的 `PATH`：

```text
3rdparty/cutlass/build/windows-vs2022-5060/tools/library/Release
$env:CUDA_PATH/bin/x64
```

仓库的 `run_profiler.ps1` 会自动设置这两个运行时目录。

### cuBLAS verification 不可用

检查 CMake configure 输出是否找到 cuBLAS，并确认使用了：

```text
-DCUTLASS_ENABLE_CUBLAS=ON
```

如果输出仍是 `cuBLAS: Not run`，不要把它记录成验证通过。先加
`--verification-required=true` 让流水线失败；需要严格校验时，按
`windows-linux-build.md` 的 Full reference profiler build 启用 CUTLASS reference。

### alignment 过滤不起作用

Profiler 没有 `--alignment=8`。必须用：

```text
--kernels=*align8
```

### 把 warp count 当成 warp shape

`warps_m=2, warps_n=2` 表示 CTA 中有 4 个 warp，不表示 warp tile 是 `2x2`。
warp tile 必须用 CTA tile 除以 warp count。
