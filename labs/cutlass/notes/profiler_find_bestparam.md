# RTX 5060：用 CUTLASS Profiler 为 8192x8192x4096 GEMM 找参数

本文针对一个固定任务：

```text
GPU:         NVIDIA GeForce RTX 5060
OS:          Windows 10/11
Architecture: sm_120
M/N/K:       8192 / 8192 / 4096
A:           FP16 row-major
B:           FP16 column-major
C/D:         FP32 column-major
Accumulator: FP32
Formula:     D = alpha * A * B + beta * C
alpha/beta:  1 / 0
```

目标不是让 Profiler 猜一个参数，而是：生成候选 kernel、筛选、首轮测量、重复确认、
扫描 split-K，最后把胜出配置映射到 `cutlass_3x_gemm_best.cu`。

## 1. 本次实测结论

当前机器使用 CUTLASS 4.5.2、CUDA 13.0、RTX 5060 / SM120。最终选择：

| 参数 | 选择值 |
| --- | --- |
| Profiler kernel | `cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8` |
| CTA tile | `128x128x32` |
| Warp count | `2x2x1`，共 4 warps / 128 threads |
| 推导 warp shape | `64x64x32` |
| MMA instruction | `16x8x16` |
| Pipeline stages | 3 |
| A/B alignment | 8 个 FP16，即 128 bit |
| Split-K | serial, slices=1，也就是普通 GEMM |
| Raster/swizzle | 保持默认 heuristic / 1 |

五次确认测试的中位数：

```text
Runtime: 13.0291 ms
Math:    42.205 TFLOP/s
Range:   13.0045 ~ 13.0374 ms
```

这些数值只适用于当前 GPU、驱动、CUDA、CUTLASS 版本、数据类型和 layout。换机器或
升级工具链后应重新运行本文流程。

## 2. Profiler 与最终 3.x 代码的关系

Profiler 中胜出的 operation 是 CUTLASS 官方生成的 SM80-compatible legacy kernel。
最终代码使用 CUTLASS 3.x `GemmUniversalAdapter + CollectiveMma + CuTe` 表达同一套：

```text
CTA tile
warp topology
MMA atom
pipeline stages
128-bit A/B copy
split-K=1
```

这叫做“参数拓扑映射”，不是两个二进制逐指令完全相同。尤其 epilogue 组织、模板
实例和编译器资源分配可能不同，所以最终 3.x 代码必须单独编译、校验和计时。

## 3. Step 1：检查环境

```powershell
cd D:\workspace\learing\AI_system

nvidia-smi
nvcc --version
cmake --version
python --version
nvcc --list-gpu-code | Select-String sm_120
```

确认：

- GPU 是 RTX 5060；
- CUDA 能生成 `sm_120`；
- 安装了 Visual Studio 2022 C++ workload；
- `3rdparty/cutlass` 是 CUTLASS 4.5.2 或你明确记录的其他版本。

## 4. Step 2：配置 Profiler kernel 库

Profiler 参数是对“已经编译的 kernel”的筛选条件，不会运行时生成新 kernel。因此
必须先生成包含 tile/stages/alignment 变化的 operation library。

推荐使用仓库脚本：

```powershell
.\labs\cutlass\scripts\configure_official_cutlass.ps1
.\labs\cutlass\scripts\build_official_cutlass.ps1
```

等价的完整命令：

```powershell
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

不要使用 `CUTLASS_LIBRARY_KERNELS=all`。本次受限 family 已经生成 264 个实例；全量
构建会大幅增加编译时间和磁盘占用。

## 5. Step 3：设置运行环境

Visual Studio 把 `cutlass.dll` 放在 library 目录，不会自动复制到 Profiler 旁边。
CUDA 13 的 cuBLAS DLL 位于 `bin/x64`。在当前 PowerShell 设置：

```powershell
$profiler = Resolve-Path ".\3rdparty\cutlass\build\windows-vs2022-5060\tools\profiler\Release\cutlass_profiler.exe"
$cutlassDllDir = Resolve-Path ".\3rdparty\cutlass\build\windows-vs2022-5060\tools\library\Release"
$cudaDllDir = Join-Path $env:CUDA_PATH "bin\x64"
$env:PATH = "$cutlassDllDir;$cudaDllDir;$env:PATH"

$resultDir = ".\out\cutlass\find_best_8192x8192x4096_sm120"
New-Item -ItemType Directory -Force $resultDir | Out-Null

& $profiler --version
& $profiler --device-info |
  Tee-Object -FilePath "$resultDir\device_info.txt"
```

启动返回 `0xC0000135` 表示 DLL 搜索失败，先检查上面的两个 PATH 目录。

## 6. Step 4：枚举候选 kernel

```powershell
& $profiler `
  --mode=enumerate `
  --operation=Gemm `
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_*' |
  Tee-Object -FilePath "$resultDir\kernel_inventory.txt"
```

`tn` 在这个 operation name 中表示 A row-major、B column-major。当前 generator 的
这个 family 固定使用 FP32 column-major C/D。

确认 inventory 至少包含：

```text
128x128_32x3_tn_align8
128x128_32x3_tn_align4
128x128_32x3_tn_align2
128x128_32x4_tn_align8
128x128_32x5_tn_align8
256x128_32x3_tn_align8
128x256_32x3_tn_align8
256x64_32x4_tn_align8
64x256_32x4_tn_align8
```

## 7. Step 5：先做单 kernel smoke

```powershell
& $profiler `
  --operation=Gemm `
  --providers=cutlass `
  --m=8192 --n=8192 --k=4096 `
  '--A=f16:row' `
  '--B=f16:column' `
  '--C=f32:column' `
  '--D=f32:column' `
  --accum=f32 `
  --alpha=1 --beta=0 `
  --warmup-iterations=2 `
  --profiling-iterations=5 `
  '--kernels=cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8'
```

`Status=Success` 表示 kernel 能运行。正确性必须查看 `Disposition`；当前 CUDA 13 +
SM120 环境中 cuBLASLt reference 可能显示 `Not run`。严格检查时加入：

```text
--verification-enabled=true
--verification-providers=cublas
--verification-required=true
```

如果 cuBLAS 没有真正运行，严格模式会失败，不能把 `Not verified` 写成 `Passed`。
最终 `cutlass_3x_gemm_best` 自带独立 GPU 全输出校验。

## 8. Step 6：首轮扫描全部候选

性能扫描阶段关闭 reference，避免把 reference 时间和可用性混入 kernel 排名：

```powershell
& $profiler `
  --operation=Gemm `
  --providers=cutlass `
  --m=8192 --n=8192 --k=4096 `
  '--A=f16:row' `
  '--B=f16:column' `
  '--C=f32:column' `
  '--D=f32:column' `
  --accum=f32 `
  --op_class=tensorop `
  --inst_m=16 --inst_n=8 --inst_k=16 `
  --alpha=1 --beta=0 `
  --verification-enabled=false `
  --warmup-iterations=5 `
  --profiling-iterations=20 `
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_*' `
  '--tags=gpu:rtx5060_sm120,shape:8192x8192x4096,phase:kernel_sweep' `
  "--output=$resultDir\kernel_sweep" `
  --verbose=false
```

实际 CSV：

```text
out/cutlass/find_best_8192x8192x4096_sm120/kernel_sweep.gemm.csv
```

当前构建中有 66 个 `tn` 候选，45 个在本次过滤和设备约束下产生成功测量。0-match、
`not_supported` 和只有表头的 CSV 都不能当作 0 TFLOP/s；它们是不适用配置。

## 9. Step 7：排序首轮结果

```powershell
$csv = "$resultDir\kernel_sweep.gemm.csv"
$rows = @(Import-Csv $csv)

$rows |
  Where-Object { $_.Status -eq 'success' -and $_.Runtime } |
  Sort-Object { [double]$_.GFLOPs } -Descending |
  Select-Object -First 15 `
    Operation,cta_m,cta_n,cta_k,stages,warps_m,warps_n,warps_k,`
    inst_m,inst_n,inst_k,Runtime,GFLOPs |
  Format-Table -AutoSize
```

不要直接采用首轮第一名。多个 kernel 连续执行时，GPU boost、温度和执行顺序会改变
1% 左右的排名。本次首轮中 align4 暂时领先，但独立重复后 align8 才是稳定第一。

## 10. Step 8：独立确认前六名

确认集合：

```powershell
$kernels = @(
  'cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8'
  'cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align4'
  'cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align2'
  'cutlass_tensorop_s16816gemm_f16_256x128_32x3_tn_align8'
  'cutlass_tensorop_s16816gemm_f16_128x128_32x5_tn_align8'
  'cutlass_tensorop_s16816gemm_f16_256x64_32x4_tn_align8'
)

$confirmDir = Join-Path $resultDir 'confirm'
New-Item -ItemType Directory -Force $confirmDir | Out-Null

foreach ($kernel in $kernels) {
  foreach ($repeat in 1..5) {
    & $profiler `
      --operation=Gemm --providers=cutlass `
      --m=8192 --n=8192 --k=4096 `
      '--A=f16:row' '--B=f16:column' `
      '--C=f32:column' '--D=f32:column' `
      --accum=f32 --alpha=1 --beta=0 `
      --verification-enabled=false `
      --warmup-iterations=10 `
      --profiling-iterations=50 `
      "--kernels=$kernel" `
      "--tags=repeat:$repeat,phase:confirm,gpu:rtx5060_sm120" `
      "--output=$confirmDir\${kernel}_r$repeat" `
      --verbose=false
    if ($LASTEXITCODE -ne 0) { throw "confirmation failed: $kernel" }
  }
}
```

实测中位数：

| Rank | Kernel 核心参数 | Median ms | Median TFLOP/s |
| ---: | --- | ---: | ---: |
| 1 | 128x128x32, stages=3, align8 | 13.0291 | 42.205 |
| 2 | 128x128x32, stages=3, align4 | 13.1252 | 41.896 |
| 3 | 128x128x32, stages=3, align2 | 13.2905 | 41.375 |
| 4 | 128x128x32, stages=5, align8 | 13.3051 | 41.329 |
| 5 | 256x128x32, stages=3, align8 | 13.3462 | 41.202 |
| 6 | 256x64x32, stages=4, align8 | 13.3485 | 41.195 |

结论：128-bit A/B 搬运、128x128 CTA、3-stage pipeline 的组合最稳定。更深 pipeline
没有弥补额外 shared-memory/resource 成本；更大的 CTA 也没有带来净收益。

## 11. Step 9：确认 split-K

基准 CTA 网格是：

```text
8192 / 128 = 64 CTA along M
8192 / 128 = 64 CTA along N
64 * 64 = 4096 output CTAs
```

已经有充足并行度，理论上不需要 split-K。仍应实测：

```powershell
$bestKernel = 'cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8'
$splitDir = Join-Path $resultDir 'splitk'
New-Item -ItemType Directory -Force $splitDir | Out-Null

$cases = @(
  @('serial',1), @('serial',2), @('serial',4), @('serial',8),
  @('parallel',2), @('parallel',4), @('parallel',8)
)

foreach ($case in $cases) {
  $mode = $case[0]
  $slices = $case[1]
  foreach ($repeat in 1..3) {
    & $profiler `
      --operation=Gemm --providers=cutlass `
      --m=8192 --n=8192 --k=4096 `
      '--A=f16:row' '--B=f16:column' `
      '--C=f32:column' '--D=f32:column' `
      --accum=f32 --alpha=1 --beta=0 `
      --verification-enabled=false `
      --warmup-iterations=5 --profiling-iterations=30 `
      "--kernels=$bestKernel" `
      "--split_k_mode=$mode" `
      "--split_k_slices=$slices" `
      "--tags=repeat:$repeat,phase:splitk" `
      "--output=$splitDir\${mode}_${slices}_r$repeat" `
      --verbose=false
  }
}
```

结果：

| Mode | Slices | Median ms |
| --- | ---: | ---: |
| serial | 1 | 13.0460 |
| serial | 2 | 13.0466 |
| serial | 4 | 13.9640 |
| serial | 8 | 14.1933 |
| parallel | 2 | 15.9104 |
| parallel | 4 | 17.3790 |
| parallel | 8 | 20.3768 |

serial 1 和 2 的差距小于噪声，但 slices=1 更简单、无额外分割依赖，因此最终选择 1。
parallel split-K 的额外 workspace/reduction 明显变慢。

## 12. Step 10：可选 raster/swizzle

可以扫描：

```text
raster_order = along_m, along_n
swizzle_size = 1, 2, 4, 8
```

但本次胜出的是 legacy SM80-compatible family。Profiler 接受这些参数，不代表 kernel
一定真正实现运行时 tile scheduler。实测差异不足以形成稳定结论，因此最终代码保持
默认 heuristic/identity 行为，不把 raster/swizzle 宣称为已优化参数。

## 13. Step 11：参数映射到 CUTLASS 3.x

Profiler 到代码的映射：

| Profiler | CUTLASS 3.x/CuTe |
| --- | --- |
| `cta=128x128x32` | `Shape<_128,_128,_32>` |
| `stages=3` | `MainloopSm80CpAsync<3>` |
| `warps=2x2x1` | `Layout<Shape<_2,_2,_1>>` |
| `inst=16x8x16` | `MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>` |
| `align8` | `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` + 8-value vector layout |
| `split_k_slices=1` | `GemmUniversalMode::kGemm` |
| C/D column-major | `LayoutC = cutlass::layout::ColumnMajor` |

实现文件：

```text
labs/cutlass/examples/cutlass_3x_gemm_best.cu
```

## 14. Step 12：编译最终 3.x 代码

```powershell
cd D:\workspace\learing\AI_system

cmake -S . --preset windows-vs2022-cuda-release `
  -DAI_SYSTEM_CUTLASS_ROOT="D:\workspace\learing\AI_system\3rdparty\cutlass"

cmake --build --preset windows-vs2022-cuda-release `
  --config Release `
  --target cutlass_3x_gemm_best
```

## 15. Step 13：运行与校验

程序默认就是目标 shape：

```powershell
$bin = ".\out\build\windows-vs2022-cuda-release\labs\cutlass\Release"

& "$bin\cutlass_3x_gemm_best.exe"
```

显式参数运行：

```powershell
& "$bin\cutlass_3x_gemm_best.exe" `
  --m=8192 --n=8192 --k=4096 `
  --warmup=5 --iterations=20
```

快速 smoke：

```powershell
& "$bin\cutlass_3x_gemm_best.exe" `
  --m=256 --n=256 --k=256 `
  --warmup=1 --iterations=1
```

验收输出必须包含：

```text
Logical problem    : 8192 x 8192 x 4096
Output layout      : column-major
CTA tile           : 128x128x32
Pipeline           : MainloopSm80CpAsync<3>
A/B alignment      : 8 FP16 elements = 128 bits
Split-K            : 1
Verification       : PASSED
```

本机最终程序实测（5 次 warmup、20 次计时）：

```text
Average runtime    : 13.272 ms
Tensor throughput  : 41.423 TFLOP/s
Verification       : PASSED (expected 4096.000, tolerance 0.410, mismatches 0)
```

Profiler 胜出 kernel 的五次复测中位数是 13.0291 ms，最终 3.x 显式实现为
13.272 ms，差约 1.9%。这与前文的“拓扑映射而非二进制克隆”预期一致；验收最终代码时，
应以这里的独立正确性和计时结果为准。

## 16. 何时必须重新调参

出现任一条件都应重新跑 Profiler：

- M/N/K 改变；
- A/B/C/D layout 改变；
- FP16 改成 BF16、FP8、NVFP4；
- beta 从 0 改成非零，导致 epilogue 读取 C；
- GPU 从 RTX 5060 换成其他型号；
- CUDA、驱动或 CUTLASS 版本变化；
- 目标从单次 GEMM 变成连续 GEMM、CUDA Graph 或融合 epilogue。

“最佳参数”是特定 workload 和软件栈下的测量结论，不是 CUTLASS 的全局常量。
