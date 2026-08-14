# Linux RTX 4090 D：用 CUTLASS Profiler 为 8192x8192x4096 GEMM 找参数

本文是 `profiler_find_bestparam.md` 的 Linux / RTX 4090 D 版本，针对以下固定任务：

```text
GPU:          NVIDIA GeForce RTX 4090 D
OS:           Linux
Architecture: Ada / sm_89
M/N/K:        8192 / 8192 / 4096
A:            FP16 row-major
B:            FP16 column-major
C/D:          FP32 column-major
Accumulator:  FP32
Formula:      D = alpha * A * B + beta * C
alpha/beta:   1 / 0
```

目标是生成候选 kernel，扫描 CTA tile、warp topology、pipeline stages 和 alignment，
对领先候选做独立重复测量，再扫描 split-K 和可选 raster/swizzle，最终得到只适用于
当前 RTX 4090 D 软件与硬件环境的配置。

> 重要：RTX 5060 / SM120 的实测冠军不能直接称为 RTX 4090 D / SM89 的最优配置。
> 本文可把它用作 smoke test 的起点，但所有排名和最终选择都必须在 4090 D 上重测。

## 1. 最终需要产出什么

执行完本文后，应保留以下内容：

```text
out/cutlass/find_best_8192x8192x4096_sm89/
├── environment.txt
├── device_info.txt
├── kernel_inventory.txt
├── kernel_sweep.gemm.csv
├── top6.txt
├── confirm/
├── confirm_ranking.csv
├── best_kernel.txt
├── splitk/
├── splitk_ranking.csv
└── raster/                         # 可选
```

最终报告至少记录：

| 参数 | RTX 4090 D 实测值 |
| --- | --- |
| Profiler kernel | 待测 |
| CTA tile | 待测 |
| Warp count / warp shape | 待测 |
| MMA instruction | 待测 |
| Pipeline stages | 待测 |
| A/B alignment | 待测 |
| Split-K mode / slices | 待测 |
| Raster/swizzle | 待测或保持默认 |
| 五次测量中位数 | 待测 |
| Runtime 范围 | 待测 |
| Median TFLOP/s | 待测 |

## 2. Profiler 参数与 CUTLASS 3.x 代码的关系

本次 Profiler 构建使用 CUTLASS 生成的
`cutlass_tensorop_s16816gemm_f16_*` kernel family。Profiler 找到的结果可映射为：

```text
Profiler CTA tile       -> CuTe TileShape
Profiler warp count     -> CuTe TiledMMA warp layout
Profiler instruction    -> MMA atom
Profiler stages         -> mainloop dispatch policy
Profiler alignment      -> global-memory copy atom/vector width
Profiler split-K        -> GemmUniversal mode和切分策略
```

这是算法拓扑映射，不保证生成逐指令相同的二进制。CUTLASS 3.x 的 epilogue、模板实例、
寄存器分配和调度仍可能不同，所以映射后的 3.x 内核必须单独做正确性验证和计时。

## 3. Step 1：检查 Linux 和 GPU 环境

从仓库根目录开始：

```bash
cd /path/to/AI_system

uname -a
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,pstate,temperature.gpu,power.draw \
  --format=csv,noheader
nvcc --version
cmake --version
python3 --version
nvcc --list-gpu-code | grep sm_89
```

确认：

- `nvidia-smi` 显示 NVIDIA GeForce RTX 4090 D；
- CUDA 编译器支持 `sm_89`；
- 当前用户能正常运行 CUDA 程序；
- `3rdparty/cutlass` 是 CUTLASS 4.5.2，或记录了实际使用的其他版本；
- 编译、扫描和最终复测期间没有其他进程长期占用 GPU。

记录完整环境，之后才能解释不同机器之间的结果差异：

```bash
RESULT_DIR="${PWD}/out/cutlass/find_best_8192x8192x4096_sm89"
mkdir -p "${RESULT_DIR}"

{
  date --iso-8601=seconds
  uname -a
  nvidia-smi
  nvcc --version
  cmake --version
  git -C ./3rdparty/cutlass rev-parse HEAD
} | tee "${RESULT_DIR}/environment.txt"
```

## 4. Step 2：配置并编译 CUTLASS Profiler

Profiler 只能筛选编译时已经生成的 kernel，不会根据运行时参数临时生成新的 tile 或
stage 组合。因此必须先构建包含候选变化的 operation library。

### 4.1 使用仓库脚本

```bash
bash ./labs/cutlass/scripts/configure_official_cutlass.sh
bash ./labs/cutlass/scripts/build_official_cutlass.sh
```

脚本默认构建目录是：

```text
3rdparty/cutlass/build/linux-4090d
```

### 4.2 等价的完整命令

需要修改候选 family 时，直接使用下面的完整命令更清楚：

```bash
CUTLASS_ROOT="${PWD}/3rdparty/cutlass"
BUILD_DIR="${CUTLASS_ROOT}/build/linux-4090d"

cmake \
  -S "${CUTLASS_ROOT}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=89 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_CUBLAS=ON \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  '-DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_f16_*' \
  '-DCUTLASS_LIBRARY_IGNORE_KERNELS=cutlass_tensorop_s16816gemm_f16_s8_*,cutlass_tensorop_s16816gemm_f16_u8_*' \
  -DCUTLASS_PROFILER_DISABLE_REFERENCE=ON \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

cmake --build "${BUILD_DIR}" --target cutlass_profiler -j"$(nproc)"
```

不要一开始设置 `CUTLASS_LIBRARY_KERNELS=all`。全量 operation library 会显著增加编译
时间和磁盘占用，本任务只需要 FP16 Tensor Core GEMM family。

## 5. Step 3：准备 Profiler 运行环境

```bash
CUTLASS_ROOT="${PWD}/3rdparty/cutlass"
BUILD_DIR="${CUTLASS_ROOT}/build/linux-4090d"
PROFILER="${BUILD_DIR}/tools/profiler/cutlass_profiler"
RESULT_DIR="${PWD}/out/cutlass/find_best_8192x8192x4096_sm89"

mkdir -p "${RESULT_DIR}"
test -x "${PROFILER}" || {
  echo "cutlass_profiler does not exist or is not executable: ${PROFILER}" >&2
  exit 1
}

# 一般不需要手工设置；如果 CUTLASS 动态库不在系统搜索路径，可加入这一项。
export LD_LIBRARY_PATH="${BUILD_DIR}/tools/library:${LD_LIBRARY_PATH:-}"

ldd "${PROFILER}" | tee "${RESULT_DIR}/profiler_ldd.txt"
if ldd "${PROFILER}" | grep -q 'not found'; then
  echo 'Profiler has unresolved shared-library dependencies.' >&2
  exit 1
fi

"${PROFILER}" --version
"${PROFILER}" --device-info | tee "${RESULT_DIR}/device_info.txt"
```

如果找不到 `libcublas.so`，先确认 CUDA Toolkit 的安装位置。只有在系统没有正确配置
动态库搜索路径时，才把 CUDA 的 `lib64` 加入 `LD_LIBRARY_PATH`：

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
```

## 6. Step 4：枚举候选 kernel

```bash
"${PROFILER}" \
  --mode=enumerate \
  --operation=Gemm \
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_*' \
  | tee "${RESULT_DIR}/kernel_inventory.txt"
```

这里的 `tn` 表示 A row-major、B column-major。当前 generator 的这个 family 使用
FP32 column-major C/D。候选名称通常会包含：

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

实际 inventory 由 CUTLASS 版本和 generator 决定。某个名字不存在时，不要假设它已经
以 0 TFLOP/s 运行；它只是没有被当前构建生成。

## 7. Step 5：先做单 kernel smoke 和严格正确性检查

先用 Windows/RTX 5060 的胜出配置作为起点，只验证 Profiler 链路是否完整：

```bash
SEED_KERNEL='cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8'

"${PROFILER}" \
  --operation=Gemm \
  --providers=cutlass \
  --m=8192 --n=8192 --k=4096 \
  --A=f16:row \
  --B=f16:column \
  --C=f32:column \
  --D=f32:column \
  --accum=f32 \
  --alpha=1 --beta=0 \
  --verification-enabled=true \
  --verification-providers=cublas \
  --verification-required=true \
  --warmup-iterations=2 \
  --profiling-iterations=5 \
  --kernels="${SEED_KERNEL}"
```

验收时同时检查：

- `Status=Success`：CUTLASS kernel 成功运行；
- `Disposition=Passed`：reference provider 真正运行且结果一致。

如果结果是 `Not run` 或 `Not verified`，不能写成通过。可以先用
`--verification-required=false` 诊断 reference provider，但最终代码仍必须有独立正确性
校验。

## 8. Step 6：首轮扫描所有候选

性能排名阶段关闭 reference，避免 reference 可用性和执行时间干扰排名：

```bash
"${PROFILER}" \
  --operation=Gemm \
  --providers=cutlass \
  --m=8192 --n=8192 --k=4096 \
  --A=f16:row \
  --B=f16:column \
  --C=f32:column \
  --D=f32:column \
  --accum=f32 \
  --op_class=tensorop \
  --inst_m=16 --inst_n=8 --inst_k=16 \
  --alpha=1 --beta=0 \
  --verification-enabled=false \
  --warmup-iterations=5 \
  --profiling-iterations=20 \
  '--kernels=cutlass_tensorop_s16816gemm_f16_*_tn_*' \
  '--tags=gpu:rtx4090d_sm89,shape:8192x8192x4096,phase:kernel_sweep' \
  --output="${RESULT_DIR}/kernel_sweep" \
  --verbose=false
```

预期 CSV：

```text
out/cutlass/find_best_8192x8192x4096_sm89/kernel_sweep.gemm.csv
```

检查至少有一行成功数据：

```bash
test -s "${RESULT_DIR}/kernel_sweep.gemm.csv"
head -n 2 "${RESULT_DIR}/kernel_sweep.gemm.csv"
```

`not_supported`、0-match 和只有 CSV 表头都不是慢 kernel，而是不适用或没有匹配的
配置，应从性能排名中排除。

## 9. Step 7：排序首轮结果并生成 Top-6

下面只依赖 Python 标准库，不需要 pandas：

```bash
export RESULT_DIR
python3 - <<'PY'
import csv
import os

result_dir = os.environ['RESULT_DIR']
source = os.path.join(result_dir, 'kernel_sweep.gemm.csv')
target = os.path.join(result_dir, 'top6.txt')

with open(source, newline='', encoding='utf-8-sig') as handle:
    rows = [
        row for row in csv.DictReader(handle)
        if row.get('Status', '').lower() == 'success' and row.get('Runtime')
    ]

rows.sort(key=lambda row: float(row['GFLOPs']), reverse=True)
top = rows[:15]
columns = (
    'Operation', 'cta_m', 'cta_n', 'cta_k', 'stages',
    'warps_m', 'warps_n', 'warps_k',
    'inst_m', 'inst_n', 'inst_k', 'Runtime', 'GFLOPs'
)
print('\t'.join(columns))
for row in top:
    print('\t'.join(row.get(column, '') for column in columns))

seen = set()
top6 = []
for row in rows:
    operation = row['Operation']
    if operation not in seen:
        seen.add(operation)
        top6.append(operation)
    if len(top6) == 6:
        break

if len(top6) < 6:
    raise SystemExit(f'Only {len(top6)} successful unique kernels were found')

with open(target, 'w', encoding='utf-8') as handle:
    handle.write('\n'.join(top6) + '\n')
print(f'Wrote {target}')
PY

cat "${RESULT_DIR}/top6.txt"
```

不要直接采用首轮第一名。GPU boost、温度、功耗限制和候选执行顺序可能改变接近配置的
排名。首轮用于缩小范围，最终选择应看独立重复测试的中位数和波动范围。

## 10. Step 8：对 Top-6 做五次独立确认

```bash
CONFIRM_DIR="${RESULT_DIR}/confirm"
mkdir -p "${CONFIRM_DIR}"
mapfile -t KERNELS < "${RESULT_DIR}/top6.txt"

for kernel in "${KERNELS[@]}"; do
  for repeat in 1 2 3 4 5; do
    "${PROFILER}" \
      --operation=Gemm --providers=cutlass \
      --m=8192 --n=8192 --k=4096 \
      --A=f16:row --B=f16:column \
      --C=f32:column --D=f32:column \
      --accum=f32 --alpha=1 --beta=0 \
      --verification-enabled=false \
      --warmup-iterations=10 \
      --profiling-iterations=50 \
      --kernels="${kernel}" \
      --tags="repeat:${repeat},phase:confirm,gpu:rtx4090d_sm89" \
      --output="${CONFIRM_DIR}/${kernel}_r${repeat}" \
      --verbose=false
  done
done
```

汇总五次结果，同时生成最终候选文件：

```bash
export RESULT_DIR
python3 - <<'PY'
import csv
import glob
import os
import statistics

result_dir = os.environ['RESULT_DIR']
groups = {}
for path in glob.glob(os.path.join(result_dir, 'confirm', '*.gemm.csv')):
    with open(path, newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row.get('Status', '').lower() != 'success' or not row.get('Runtime'):
                continue
            item = groups.setdefault(row['Operation'], {'ms': [], 'gflops': []})
            item['ms'].append(float(row['Runtime']))
            item['gflops'].append(float(row['GFLOPs']))

ranking = []
for operation, values in groups.items():
    if len(values['ms']) != 5:
        raise SystemExit(f'{operation}: expected 5 results, got {len(values["ms"])}')
    ranking.append({
        'Operation': operation,
        'Runs': len(values['ms']),
        'Median_ms': statistics.median(values['ms']),
        'Min_ms': min(values['ms']),
        'Max_ms': max(values['ms']),
        'Median_GFLOPs': statistics.median(values['gflops']),
    })

ranking.sort(key=lambda row: row['Median_ms'])
if not ranking:
    raise SystemExit('No successful confirmation results found')

output = os.path.join(result_dir, 'confirm_ranking.csv')
with open(output, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=ranking[0].keys())
    writer.writeheader()
    writer.writerows(ranking)

with open(os.path.join(result_dir, 'best_kernel.txt'), 'w', encoding='utf-8') as handle:
    handle.write(ranking[0]['Operation'] + '\n')

for rank, row in enumerate(ranking, 1):
    print(
        f'{rank}: {row["Operation"]}  '
        f'median={row["Median_ms"]:.4f} ms  '
        f'range={row["Min_ms"]:.4f}..{row["Max_ms"]:.4f} ms  '
        f'{row["Median_GFLOPs"] / 1000:.3f} TFLOP/s'
    )
PY
```

选择规则：

1. 先排除运行失败或测量次数不足的配置；
2. 按五次独立运行的 `Median_ms` 排名；
3. 如果前两名差距小于约 1%，同时比较 min/max 范围，避免选择波动明显更大的配置；
4. 仍近似持平时，优先选择更自然的 128-bit alignment、更少资源或更简单的配置；
5. 保存原始 CSV，不只保存最终表格。

## 11. Step 9：扫描 split-K

对本问题来说，假设胜出 CTA 是 128x128，输出网格已有：

```text
8192 / 128 * 8192 / 128 = 4096 CTAs
```

通常不缺并行度，因此 split-K 很可能没有收益，但仍应用目标 GPU 实测：

```bash
BEST_KERNEL="$(<"${RESULT_DIR}/best_kernel.txt")"
SPLIT_DIR="${RESULT_DIR}/splitk"
mkdir -p "${SPLIT_DIR}"

cases=(serial:1 serial:2 serial:4 serial:8 parallel:2 parallel:4 parallel:8)

for item in "${cases[@]}"; do
  mode="${item%%:*}"
  slices="${item##*:}"
  for repeat in 1 2 3; do
    "${PROFILER}" \
      --operation=Gemm --providers=cutlass \
      --m=8192 --n=8192 --k=4096 \
      --A=f16:row --B=f16:column \
      --C=f32:column --D=f32:column \
      --accum=f32 --alpha=1 --beta=0 \
      --verification-enabled=false \
      --warmup-iterations=5 \
      --profiling-iterations=30 \
      --kernels="${BEST_KERNEL}" \
      --split_k_mode="${mode}" \
      --split_k_slices="${slices}" \
      --tags="repeat:${repeat},phase:splitk,gpu:rtx4090d_sm89" \
      --output="${SPLIT_DIR}/${mode}_${slices}_r${repeat}" \
      --verbose=false
  done
done
```

汇总 split-K 中位数：

```bash
export RESULT_DIR
python3 - <<'PY'
import csv
import glob
import os
import statistics

result_dir = os.environ['RESULT_DIR']
groups = {}
for path in glob.glob(os.path.join(result_dir, 'splitk', '*.gemm.csv')):
    with open(path, newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row.get('Status', '').lower() != 'success' or not row.get('Runtime'):
                continue
            key = (row['split_k_mode'], int(row['split_k_slices']))
            groups.setdefault(key, []).append(float(row['Runtime']))

ranking = []
for (mode, slices), runtimes in groups.items():
    ranking.append({
        'Mode': mode,
        'Slices': slices,
        'Runs': len(runtimes),
        'Median_ms': statistics.median(runtimes),
        'Min_ms': min(runtimes),
        'Max_ms': max(runtimes),
    })
ranking.sort(key=lambda row: row['Median_ms'])
if not ranking:
    raise SystemExit('No successful split-K results found')

output = os.path.join(result_dir, 'splitk_ranking.csv')
with open(output, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=ranking[0].keys())
    writer.writeheader()
    writer.writerows(ranking)

for row in ranking:
    print(
        f'{row["Mode"]:8} slices={row["Slices"]:<2} '
        f'median={row["Median_ms"]:.4f} ms '
        f'range={row["Min_ms"]:.4f}..{row["Max_ms"]:.4f} ms'
    )
PY
```

如果 slices=1 与其他配置差距处于噪声范围，选择 slices=1。它没有额外 reduction、
workspace 或跨 CTA 累加依赖。

## 12. Step 10：可选扫描 raster order 和 swizzle

只有 Profiler 的胜出 kernel 确实实现对应 scheduler 参数时，这些扫描才有意义：

```bash
RASTER_DIR="${RESULT_DIR}/raster"
mkdir -p "${RASTER_DIR}"

for raster in heuristic along_m along_n; do
  for swizzle in 1 2 4 8; do
    for repeat in 1 2 3; do
      "${PROFILER}" \
        --operation=Gemm --providers=cutlass \
        --m=8192 --n=8192 --k=4096 \
        --A=f16:row --B=f16:column \
        --C=f32:column --D=f32:column \
        --accum=f32 --alpha=1 --beta=0 \
        --verification-enabled=false \
        --warmup-iterations=5 --profiling-iterations=30 \
        --kernels="${BEST_KERNEL}" \
        --split_k_mode=serial --split_k_slices=1 \
        --raster_order="${raster}" \
        --swizzle_size="${swizzle}" \
        --tags="repeat:${repeat},phase:raster,gpu:rtx4090d_sm89" \
        --output="${RASTER_DIR}/${raster}_${swizzle}_r${repeat}" \
        --verbose=false
    done
  done
done
```

legacy SM80-compatible kernel family 可能接受参数但不真正改变底层 tile scheduler。若差异
没有超过测量波动，应记录“无稳定结论”，并保持默认 heuristic / swizzle=1。

## 13. Step 11：填写 RTX 4090 D 最终结果

从 `confirm_ranking.csv`、`splitk_ranking.csv` 和可选 raster 结果中填写：

| Rank | Kernel 核心参数 | Median ms | Min..Max ms | Median TFLOP/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | 待测 | 待测 | 待测 | 待测 |
| 2 | 待测 | 待测 | 待测 | 待测 |
| 3 | 待测 | 待测 | 待测 | 待测 |
| 4 | 待测 | 待测 | 待测 | 待测 |
| 5 | 待测 | 待测 | 待测 | 待测 |
| 6 | 待测 | 待测 | 待测 | 待测 |

然后填写配置结论：

```text
Profiler kernel:  <best_kernel.txt 中的值>
CTA tile:         <cta_m>x<cta_n>x<cta_k>
Warp count:       <warps_m>x<warps_n>x<warps_k>
Warp shape:       CTA shape / warp count
MMA instruction:  <inst_m>x<inst_n>x<inst_k>
Pipeline stages:  <stages>
A/B alignment:    <从 operation name 和 inventory 确认>
Split-K:          <mode>, slices=<n>
Raster/swizzle:   <实测值或默认>
Median runtime:   <ms>
Median math:      <TFLOP/s>
```

## 14. Step 12：映射到 CUTLASS 3.x

如果 RTX 4090 D 的胜出结果恰好仍是示例配置，可映射为：

| Profiler | CUTLASS 3.x / CuTe |
| --- | --- |
| `cta=128x128x32` | `Shape<_128,_128,_32>` |
| `stages=3` | `MainloopSm80CpAsync<3>` |
| `warps=2x2x1` | `Layout<Shape<_2,_2,_1>>` |
| `inst=16x8x16` | `MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>` |
| `align8` | 128-bit copy atom + 8-value FP16 vector layout |
| `split_k_slices=1` | `GemmUniversalMode::kGemm` |
| C/D column-major | `LayoutC = cutlass::layout::ColumnMajor` |

如果实际 winner 不同，必须按 CSV 中的 CTA、warp、stage、instruction 和 alignment 修改
CuTe 类型，不能照抄 RTX 5060 的参数。现有
`labs/cutlass/examples/cutlass_3x_gemm_best.cu` 可以作为结构参考，但其中配置与性能注释
来自 RTX 5060，不能在完成 SM89 实测前称为 4090 D 最优内核。

## 15. Step 13：在 Linux 编译映射后的 3.x 内核

仓库 Linux preset 已固定 RTX 4090 D / `sm_89`：

```bash
cmake -S . --preset linux-make-cuda-release \
  -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"

cmake --build --preset linux-make-cuda-release \
  --target cutlass_3x_gemm_best -j"$(nproc)"
```

编译产物：

```text
out/build/linux-make-cuda-release/labs/cutlass/cutlass_3x_gemm_best
```

运行默认目标 shape：

```bash
BIN=./out/build/linux-make-cuda-release/labs/cutlass
"${BIN}/cutlass_3x_gemm_best"
```

显式指定参数：

```bash
"${BIN}/cutlass_3x_gemm_best" \
  --m=8192 --n=8192 --k=4096 \
  --warmup=5 --iterations=20
```

先用非整 tile 尺寸检查 padding 和边界：

```bash
"${BIN}/cutlass_3x_gemm_best" \
  --m=130 --n=130 --k=130 \
  --warmup=1 --iterations=1
```

运行输出必须包含 `Verification: PASSED`。映射后的 3.x 内核性能应单独记录，不能直接
引用 Profiler operation 的计时。

## 16. 让性能结果可复现

每轮确认测试前执行：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem \
  --format=csv,noheader
```

建议：

- 关闭其他 GPU 工作负载；
- 先让 GPU 温度进入相对稳定状态；
- 不在一轮扫描中修改功耗上限、锁频或驱动设置；
- 同一比较组使用相同 warmup 和 profiling iterations；
- 用多次独立进程的中位数，而不是单次最佳值；
- 保留驱动、CUDA、CUTLASS commit 和完整 CSV。

## 17. 何时必须重新调参

出现任一条件都应重新跑 Profiler：

- M/N/K 改变；
- A/B/C/D 的 layout 改变；
- FP16 改成 BF16、TF32 或 FP8；
- beta 从 0 改成非零，导致 epilogue 读取 C；
- GPU、驱动、CUDA 或 CUTLASS 版本变化；
- 功耗限制或散热状态明显变化；
- 从单次 GEMM 改成连续 GEMM、CUDA Graph 或融合 epilogue；
- 最终应用的前后算子改变了 cache 状态或并发行为。

“最佳参数”是特定 workload、数据布局、GPU 和软件栈下的测量结论，不是 CUTLASS 的
全局常量。
