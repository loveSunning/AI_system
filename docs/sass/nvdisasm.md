# nvdisasm 使用文档

本文记录在本工程中使用 `nvcc` 先生成 cubin，再用 `nvdisasm` 生成 GEMM SASS 的流程。

## 适用场景

`nvdisasm` 直接反汇编 cubin/ELF image。它不能像 `cuobjdump` 那样直接遍历 `.lib` archive；通常有两种输入来源：

1. 用 `nvcc -cubin` 从 CUDA 源文件生成 cubin。
2. 用 `cuobjdump --extract-elf` 从已有二进制或静态库中抽取 cubin。

本次任务按第一种流程执行：从 `labs\gemm\*.cu` 生成 cubin，再生成 SASS。所有 cubin 和 SASS 都保存到：

```powershell
D:\workspace\learing\AI_system\out\sass
```

## Windows 环境准备

普通 PowerShell 里可能找不到 MSVC 的 `cl.exe`，此时 `nvcc` 会报错：

```text
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
```

可选做法：

1. 使用 Visual Studio Developer PowerShell / Developer Command Prompt。
2. 在调用 `nvcc` 的同一个命令里先运行 `vcvars64.bat`。

本次使用第二种方式：

```powershell
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
```

## 生成 cubin

当前工程 Release 构建目标是 `sm_120`，所以 `nvcc` 使用 `-arch=sm_120`。为了让 `nvdisasm` 尽量输出 line info，本次额外带上 `-lineinfo`。

在工程根目录执行：

```powershell
New-Item -ItemType Directory -Force out\sass

$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$kernels = @(
  'tiled_gemm_block',
  'tiled_gemm_register',
  'gemm_dbuffer_vload',
  'gemm_wrap_tile',
  'sgemm_v1',
  'sgemm_v3'
)

foreach ($kernel in $kernels) {
  $src = "labs\gemm\$kernel.cu"
  $out = "out\sass\nvcc_${kernel}_sm120.cubin"
  cmd /c "`"$vcvars`" && nvcc -std=c++20 --expt-relaxed-constexpr -O3 -lineinfo -arch=sm_120 -Iinclude -Ilabs\gemm -Iout\build\windows-vs2022-cuda-release\generated\include -cubin $src -o $out"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

关键参数说明：

| 参数 | 作用 |
| --- | --- |
| `-std=c++20` | 匹配工程 CUDA/C++ 标准 |
| `--expt-relaxed-constexpr` | 匹配 CMake 中 CUDA 编译选项 |
| `-O3` | 生成优化后的 SASS，适合性能分析 |
| `-lineinfo` | 让 cubin 携带行号信息，便于 `nvdisasm` 标注 |
| `-arch=sm_120` | 生成 Blackwell `sm_120` cubin |
| `-Iinclude` | 工程公共头文件 |
| `-Ilabs\gemm` | GEMM lab 头文件 |
| `-Iout\build\windows-vs2022-cuda-release\generated\include` | CMake 生成的 `ai_system/config.hpp` |
| `-cubin` | 只生成 device cubin，不生成 host object/exe |

## 生成 SASS

对每个 cubin 调用 `nvdisasm`：

```powershell
$kernels = @(
  'tiled_gemm_block',
  'tiled_gemm_register',
  'gemm_dbuffer_vload',
  'gemm_wrap_tile',
  'sgemm_v1',
  'sgemm_v3'
)

foreach ($kernel in $kernels) {
  $cubin = "out\sass\nvcc_${kernel}_sm120.cubin"
  $sass = "out\sass\nvdisasm_${kernel}_sm120.sass"
  nvdisasm --print-code --separate-functions --print-line-info-ptx --print-instruction-encoding $cubin |
      Out-File -FilePath $sass -Encoding ascii
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

本次使用的 `nvdisasm` 参数：

| 参数 | 作用 |
| --- | --- |
| `--print-code` | 主要输出 code section，减少非代码段噪音 |
| `--separate-functions` | 在函数之间增加分隔，方便扫描各模板实例 |
| `--print-line-info-ptx` | 使用 cubin 中的 line info 标注反汇编 |
| `--print-instruction-encoding` | 在每条 SASS 后保留机器码 encoding |

## 已生成文件

| kernel family | cubin | SASS | 函数数量 |
| --- | --- | --- | ---: |
| `tiled_gemm_block_kernel` | `out\sass\nvcc_tiled_gemm_block_sm120.cubin` | `out\sass\nvdisasm_tiled_gemm_block_sm120.sass` | 45 |
| `tiled_gemm_register_kernel` | `out\sass\nvcc_tiled_gemm_register_sm120.cubin` | `out\sass\nvdisasm_tiled_gemm_register_sm120.sass` | 366 |
| `gemm_dbuffer_vload_kernel` | `out\sass\nvcc_gemm_dbuffer_vload_sm120.cubin` | `out\sass\nvdisasm_gemm_dbuffer_vload_sm120.sass` | 54 |
| `gemm_wrap_tile_kernel` | `out\sass\nvcc_gemm_wrap_tile_sm120.cubin` | `out\sass\nvdisasm_gemm_wrap_tile_sm120.sass` | 54 |
| `sgemm_v1_kernel` | `out\sass\nvcc_sgemm_v1_sm120.cubin` | `out\sass\nvdisasm_sgemm_v1_sm120.sass` | 54 |
| `sgemm_v3_kernel` | `out\sass\nvcc_sgemm_v3_sm120.cubin` | `out\sass\nvdisasm_sgemm_v3_sm120.sass` | 54 |

统计 nvdisasm 输出中的函数数量：

```powershell
Select-String -Path out\sass\nvdisasm_*_sm120.sass -Pattern '^//--------------------- \.text' |
    Group-Object Path |
    Select-Object Name,Count
```

## 常用变体

只看 SASS，不显示机器码：

```powershell
nvdisasm --print-code --separate-functions out\sass\nvcc_sgemm_v1_sm120.cubin
```

输出控制流图，供 Graphviz 使用：

```powershell
nvdisasm --output-control-flow-graph --print-instr-offsets-cfg out\sass\nvcc_sgemm_v1_sm120.cubin |
    Out-File -FilePath out\sass\nvdisasm_sgemm_v1_sm120.cfg.dot -Encoding ascii
```

输出 JSON，便于脚本处理：

```powershell
nvdisasm --emit-json out\sass\nvcc_sgemm_v1_sm120.cubin |
    Out-File -FilePath out\sass\nvdisasm_sgemm_v1_sm120.json -Encoding ascii
```

## 和 cuobjdump 的区别

| 工具 | 输入 | 优势 | 本工程用法 |
| --- | --- | --- | --- |
| `cuobjdump` | `.lib`、`.obj`、`.exe`、fatbin、cubin | 能遍历二进制容器，列 ELF/PTX/符号/资源占用 | 从 `ai_system_gemm_lab.lib` 直接提取信息并 dump SASS |
| `nvdisasm` | cubin/ELF image | 反汇编格式更适合函数级阅读，支持 CFG/JSON/line info 等视图 | 用 `nvcc -cubin` 先生成 cubin，再反汇编 |

若目标是“看已有 Release 库里真实嵌入了什么”，优先用 `cuobjdump`。若目标是“从某个 `.cu` 独立生成 cubin 并做更细的反汇编视图”，优先用 `nvdisasm`。

