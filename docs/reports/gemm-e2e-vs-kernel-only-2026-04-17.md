# GEMM e2e vs kernel_only Report

日期: 2026-04-17  
平台: Windows 10 + CUDA 13.0 + RTX 5060 (sm_120)  
构建: `windows-vs2022-cuda-release`

## 1. 目的

这次把 `perf_engineering_lab` 里的 GEMM 对比分成了两套口径:

- `gemm_e2e`: 统计整条调用链, 包括 host 侧准备、设备端分配、H2D/D2H 拷贝、cuBLAS handle / 库加载成本、以及半精度转换成本。
- `gemm_kernel_only`: 输入和输出缓冲区、cuBLAS handle 都提前准备好, measured 阶段只用 CUDA event 统计设备端调用本身。

这两套口径需要同时存在, 因为它们回答的是两个不同问题:

- `e2e` 回答 “真实业务路径里, 这条调用到底要花多少时间”
- `kernel_only` 回答 “如果把准备成本拿掉, 纯设备端 GEMM 谁更快”

## 2. 代码变更

- 公共 timed benchmark 接口: [benchmark_runner.hpp](/E:/learning/AI_system/include/ai_system/benchmark/benchmark_runner.hpp:18), [benchmark_runner.cpp](/E:/learning/AI_system/src/benchmark/benchmark_runner.cpp:54)
- GEMM 设备端 prepared runner: [basic_kernels.hpp](/E:/learning/AI_system/include/ai_system/kernels/basic_kernels.hpp:12), [basic_kernels.cu](/E:/learning/AI_system/src/kernels/basic_kernels.cu:395)
- `perf_engineering_lab` 现在同时输出 `gemm_e2e` 和 `gemm_kernel_only`: [perf_engineering_lab.cpp](/E:/learning/AI_system/labs/perf_engineering/perf_engineering_lab.cpp:279)

## 3. 运行命令

构建与测试:

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release
ctest --preset windows-vs2022-cuda-release
```

尺寸 sweep:

```powershell
./out/build/windows-vs2022-cuda-release/labs/perf_engineering/Release/perf_engineering_lab.exe --vector-size 1048576 --reduction-size 1048576 --gemm-m 256 --gemm-n 256 --gemm-k 256 --warmup 1 --iters 3
./out/build/windows-vs2022-cuda-release/labs/perf_engineering/Release/perf_engineering_lab.exe --vector-size 1048576 --reduction-size 1048576 --gemm-m 512 --gemm-n 512 --gemm-k 512 --warmup 1 --iters 3
./out/build/windows-vs2022-cuda-release/labs/perf_engineering/Release/perf_engineering_lab.exe --vector-size 1048576 --reduction-size 1048576 --gemm-m 1024 --gemm-n 1024 --gemm-k 1024 --warmup 1 --iters 3
```

Nsight Systems:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe" profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --force-overwrite=true -o docs/reports/raw/nsys_256_admin out/build/windows-vs2022-cuda-release/labs/perf_engineering/Release/perf_engineering_lab.exe --vector-size 1024 --reduction-size 1024 --gemm-m 256 --gemm-n 256 --gemm-k 256 --warmup 1 --iters 1
```

Nsight Compute:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe" --set basic --target-processes all --launch-count 1 --kernel-name-base demangled --kernel-name regex:naive_gemm_kernel out/build/windows-vs2022-cuda-release/labs/perf_engineering/Release/perf_engineering_lab.exe --vector-size 1024 --reduction-size 1024 --gemm-m 256 --gemm-n 256 --gemm-k 256 --warmup 1 --iters 1
```

同样的方法还抓了:

- `regex:cutlass_80_simt_sgemm`
- `regex:wmma_tensorop_h161616gemm`
- `regex:wmma_tensorop_s161616gemm_f16`

原始输出保存在 [docs/reports/raw](/E:/learning/AI_system/docs/reports/raw)。

## 4. 尺寸对比结果

### 4.1 `gemm_e2e`

| shape | cuda_naive GFLOPS | cublas_sgemm GFLOPS | cublas_hgemm GFLOPS | cublas_tensor_core GFLOPS |
|---|---:|---:|---:|---:|
| 256x256x256 | 58.936 | 47.857 | 21.944 | 27.706 |
| 512x512x512 | 232.324 | 208.359 | 82.369 | 81.984 |
| 1024x1024x1024 | 167.749 | 152.645 | 119.017 | 116.779 |

### 4.2 `gemm_kernel_only`

| shape | cuda_naive GFLOPS | cublas_sgemm GFLOPS | cublas_hgemm GFLOPS | cublas_tensor_core GFLOPS |
|---|---:|---:|---:|---:|
| 256x256x256 | 1040.942 | 2255.002 | 6765.007 | 6959.575 |
| 512x512x512 | 1245.895 | 7871.700 | 17561.636 | 16100.975 |
| 1024x1024x1024 | 1198.487 | 11043.091 | 45849.827 | 24160.158 |

### 4.3 直接结论

1. `e2e` 口径下, `cuda_naive` 仍然比 `cublas_sgemm` 略快, 尤其在 `256` 和 `512` 这两个尺寸上更明显。
2. `kernel_only` 口径下, 结论完全反过来: `cublas_sgemm` 明显快于 `cuda_naive`, `cublas_hgemm` / `cublas_tensor_core` 又明显快于 `cublas_sgemm`。
3. 这说明之前 “naive 比 cuBLAS 快” 的现象主要不是 kernel 本身更强, 而是 `e2e` 路径里的 host 侧管理成本掩盖了设备端优势。

1024 这个尺寸最典型:

- `gemm_e2e`: `cuda_naive 167.749 GFLOPS` vs `cublas_sgemm 152.645 GFLOPS`
- `gemm_kernel_only`: `cuda_naive 1198.487 GFLOPS` vs `cublas_sgemm 11043.091 GFLOPS`

也就是说, 一旦把准备成本移出 measured 区间, `cublas_sgemm` 立刻拉开了接近 `9.2x` 的纯设备端差距。

## 5. Nsight Systems 结论

本次 `nsys` 用 `256x256x256` 做代表性分析, 因为问题规模足够小, 更容易看清 `e2e` 路径里的固定成本。

`gemm` NVTX range 下的 CUDA API 汇总:

- `cuLibraryLoadData`: `70.9%`
- `cudaMemcpy`: `8.9%`
- `cudaLaunchKernelExC_v11060`: `5.0%`
- `cudaFree`: `4.2%`
- `cuKernelGetFunction`: `4.0%`
- `cudaMalloc`: `2.6%`
- `cudaDeviceSynchronize`: `2.6%`

`gemm` NVTX range 下的 GPU kernel 汇总:

- `naive_gemm_kernel`: `63.5%`
- `cutlass_80_simt_sgemm...`: `16.6%`
- `wmma_tensorop_h161616gemm...`: `7.1%`
- `wmma_tensorop_s161616gemm_f16...`: `6.9%`
- `cublasLt::splitKreduce_kernel...`: `5.8%`

`gemm` NVTX range 下的 mem op 汇总:

- Host-to-Device: `67.0%`
- Device-to-Host: `33.0%`

结论:

1. 这条 `e2e` 路径里, 真正的 GEMM kernel 时间并不是唯一主角, host 侧库加载、分配、拷贝和同步占了非常可观的比例。
2. 对 `cublas_hgemm` 和 `cublas_tensor_core` 来说, 当前封装里还包含 `float -> half` 的 host 侧转换, 这也是它们 `e2e` 口径下没有优势的直接原因之一。
3. 因为 `nsys` 本身有观测开销, 所以这里的绝对时间不拿来替代 benchmark 表里的 `avg_ms`; `nsys` 主要用来识别时间花在哪里。

## 6. Nsight Compute 结论

### 6.1 `naive_gemm_kernel`

主要指标:

- Duration: `38.11 us`
- Memory Throughput: `81.66%`
- Compute Throughput: `81.66%`
- Achieved Occupancy: `75.33%`
- Registers / thread: `38`
- Waves / SM: `1.42`

解读:

- 这个 kernel 在 256 尺寸上已经比较“吃满”了 SM 和 memory pipe。
- `ncu` 还提示有明显的 tail effect, 原因是 `256` 个 block 只形成 `1` 个完整 wave 加一个 partial wave。

### 6.2 `cublas_sgemm`

主要指标:

- Duration: `10.50 us`
- Memory Throughput: `46.61%`
- Compute Throughput: `30.36%`
- Achieved Occupancy: `13.86%`
- Registers / thread: `78`
- Dynamic shared memory / block: `20.48 KB`
- Waves / SM: `0.40`

解读:

- 这个 kernel 比 naive 短很多, 但在 `256` 这么小的 shape 上 grid 很小, GPU 根本没被铺满。
- 也就是说, 这里 `cublas_sgemm` 的瓶颈不是“算得慢”, 而是“小问题规模下启动后很快就做完了, 没有足够工作去填满整个 GPU”。

### 6.3 `cublas_hgemm`

主要指标:

- Duration: `5.28 us`
- Memory Throughput: `28.65%`
- Compute Throughput: `12.51%`
- Achieved Occupancy: `17.50%`
- Registers / thread: `96`
- Dynamic shared memory / block: `18.94 KB`

解读:

- 设备端调用已经非常快, 但小尺寸下同样受 grid 过小影响。
- 所以它的 `kernel_only` 性能很高, 但 `e2e` 仍然会被 host 转换和拷贝成本拖住。

### 6.4 `cublas_tensor_core`

主要指标:

- Duration: `5.09 us`
- Memory Throughput: `31.61%`
- Compute Throughput: `13.52%`
- Achieved Occupancy: `17.59%`
- Registers / thread: `64`
- Dynamic shared memory / block: `18.94 KB`

解读:

- 设备端同样非常快, 和 `hgemm` 在 `256` 尺寸上几乎一个量级。
- 当前这条路径使用的是 `FP16` 输入 + `FP32` 输出/累加, 所以在 `kernel_only` 里仍很强, 但 `e2e` 口径会被 host 侧准备成本吞掉不少收益。

## 7. 为什么 `e2e` 里 `cuda_naive` 还会更快

根因已经比较明确:

1. `cuda_naive` 的封装更轻, 没有 cuBLAS handle 创建和库调度路径。
2. `cublas_hgemm` / `cublas_tensor_core` 当前每次调用都在 host 上做 `float -> half` 转换。
3. 现在的 `e2e` 还包含 `cudaMalloc / cudaMemcpy / cudaFree / synchronize`。
4. 小尺寸和中等尺寸下, 固定管理成本对总时间的影响很大。

所以当前结论应该表述成:

- “当前实现的 `e2e` 封装路径里, `cuda_naive` 更轻”
- 不是
- “naive kernel 比 cuBLAS kernel 更强”

## 8. 下一步建议

1. 把 `cublas` 的 handle 和 device buffer 做成可复用对象, 尤其是后面 `labs/gemm` 的 autotune 场景。
2. 把 `half` 输入的 host 转换提前到 benchmark 外, 避免它污染 `e2e` 对比。
3. 后续 `sgemm_v1 / sgemm_v2 / wmma_demo` 统一同时给出 `e2e` 和 `kernel_only` 两张表。
4. 如果要做更公平的库对比, 再加第三套口径: `device_resident`, 只允许一次 H2D, 多次复用 device 输入。

