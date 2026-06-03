# cuobjdump 使用文档

本文记录在本工程中使用 `cuobjdump` 从 GEMM 静态库提取 CUDA device code 信息并生成 SASS 的流程。

## 适用场景

`cuobjdump` 适合直接分析已经构建出来的 CUDA 二进制产物，例如 `.exe`、`.dll`、`.lib`、`.obj` 或 fatbin。它可以列出嵌入的 ELF/cubin/PTX，导出资源占用、符号表，也可以直接从库里 dump SASS。

本次分析对象：

```powershell
D:\workspace\learing\AI_system\out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib
```

当前 Release 构建目标架构来自 `CMakeCache.txt`：

```text
CMAKE_CUDA_ARCHITECTURES=120
```

因此本文命令统一使用 `sm_120`。

## 常用命令

在工程根目录执行：

```powershell
New-Item -ItemType Directory -Force out\sass
```

列出 `.lib` 中包含的 cubin/ELF：

```powershell
cuobjdump --list-elf out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib |
    Out-File -FilePath out\sass\cuobjdump_ai_system_gemm_lab_elf_list.txt -Encoding ascii
```

导出 ELF 符号表，后续可以用符号名定位具体 kernel：

```powershell
cuobjdump --dump-elf-symbols out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib |
    Out-File -FilePath out\sass\cuobjdump_ai_system_gemm_lab_elf_symbols.txt -Encoding ascii
```

导出每个 kernel 的寄存器、栈、shared memory、constant memory 等资源占用：

```powershell
cuobjdump --dump-resource-usage out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib |
    Out-File -FilePath out\sass\cuobjdump_ai_system_gemm_lab_resource_usage.txt -Encoding ascii
```

导出完整 SASS：

```powershell
cuobjdump --dump-sass --sort-functions --gpu-architecture sm_120 out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib |
    Out-File -FilePath out\sass\cuobjdump_ai_system_gemm_lab_sm120.sass -Encoding ascii
```

抽取 `.lib` 内嵌 cubin。注意 `cuobjdump --extract-elf` 会把文件写到当前工作目录，所以这里先进入 `out\sass`：

```powershell
Push-Location out\sass
cuobjdump --extract-elf all D:\workspace\learing\AI_system\out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib
Pop-Location
```

## 本次 ELF/cubin 映射

`cuobjdump --list-elf` 得到的映射如下：

| cubin | 来源 member | 说明 |
| --- | --- | --- |
| `ai_system_gemm_lab.1.sm_120.cubin` | `ai_system_gemm_lab.device-link.obj` | device-link 后的整体 device image |
| `ai_system_gemm_lab.2.sm_120.cubin` | `sgemm_v3.obj` | `sgemm_v3_kernel` 模板实例 |
| `ai_system_gemm_lab.3.sm_120.cubin` | `sgemm_v1.obj` | `sgemm_v1_kernel` 模板实例 |
| `ai_system_gemm_lab.4.sm_120.cubin` | `gemm_wrap_tile.obj` | `gemm_wrap_tile_kernel` 模板实例 |
| `ai_system_gemm_lab.5.sm_120.cubin` | `gemm_dbuffer_vload.obj` | `gemm_dbuffer_vload_kernel` 模板实例 |
| `ai_system_gemm_lab.6.sm_120.cubin` | `tiled_gemm_register.obj` | `tiled_gemm_register_kernel` 模板实例 |
| `ai_system_gemm_lab.7.sm_120.cubin` | `tiled_gemm_block.obj` | `tiled_gemm_block_kernel` 模板实例 |
| `ai_system_gemm_lab.8.sm_120.cubin` | `gemm_lab.cu.obj` | GEMM lab CUDA translation unit，无主要计算 kernel |

## 已生成的 SASS 文件

完整 dump 文件：

| 文件 | 说明 |
| --- | --- |
| `out\sass\cuobjdump_ai_system_gemm_lab_sm120.sass` | 从整个 `.lib` dump 出来的完整 SASS，包含 device-link image 和各 `.obj` member |

按 GEMM kernel family 拆出的文件：

| 文件 | kernel family | 函数数量 |
| --- | --- | ---: |
| `out\sass\cuobjdump_tiled_gemm_block_sm120.sass` | `tiled_gemm_block_kernel` | 45 |
| `out\sass\cuobjdump_tiled_gemm_register_sm120.sass` | `tiled_gemm_register_kernel` | 366 |
| `out\sass\cuobjdump_gemm_dbuffer_vload_sm120.sass` | `gemm_dbuffer_vload_kernel` | 54 |
| `out\sass\cuobjdump_gemm_wrap_tile_sm120.sass` | `gemm_wrap_tile_kernel` | 54 |
| `out\sass\cuobjdump_sgemm_v1_sm120.sass` | `sgemm_v1_kernel` | 54 |
| `out\sass\cuobjdump_sgemm_v3_sm120.sass` | `sgemm_v3_kernel` | 54 |

完整 `.lib` dump 会包含重复视角：device-link image 中已经有一份整体 device code，后面各 `.obj` member 中还有各自的 cubin。日常分析某一类 GEMM kernel 时，优先打开上表中按 family 拆出的文件。

## 抽取单个 kernel family 的 SASS

`out\sass\cuobjdump_tiled_gemm_block_sm120.sass` 这类文件不是只包含一个模板实例，而是包含一个 CUDA 源文件/member 对应的 kernel family。例如 `tiled_gemm_block.obj` 中有 45 个 `tiled_gemm_block_kernel<...>` 模板实例，所以这个文件里会有 45 个 `Function : ...tiled_gemm_block_kernel...`。

推荐的可复现流程有两种。

### 方法一：先抽 cubin，再 dump SASS

先根据 `.lib` 的 ELF 列表找到目标 member 对应的 cubin：

```powershell
cuobjdump --list-elf out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib
```

本次映射中，`tiled_gemm_block.obj` 对应：

```text
ELF file    7: ai_system_gemm_lab.7.sm_120.cubin
```

然后进入 `out\sass` 抽取这个 cubin。`--extract-elf` 接收 ELF 文件名的部分匹配，所以可以用 `ai_system_gemm_lab.7`：

```powershell
Push-Location out\sass
cuobjdump --extract-elf ai_system_gemm_lab.7 D:\workspace\learing\AI_system\out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib
Pop-Location
```

最后只对这个 cubin dump SASS：

```powershell
cuobjdump --dump-sass --sort-functions --gpu-architecture sm_120 out\sass\ai_system_gemm_lab.7.sm_120.cubin |
    Out-File -FilePath out\sass\cuobjdump_tiled_gemm_block_sm120.sass -Encoding ascii
```

其他 kernel family 的对应关系如下：

| kernel family | cubin | 输出文件 |
| --- | --- | --- |
| `sgemm_v3_kernel` | `ai_system_gemm_lab.2.sm_120.cubin` | `cuobjdump_sgemm_v3_sm120.sass` |
| `sgemm_v1_kernel` | `ai_system_gemm_lab.3.sm_120.cubin` | `cuobjdump_sgemm_v1_sm120.sass` |
| `gemm_wrap_tile_kernel` | `ai_system_gemm_lab.4.sm_120.cubin` | `cuobjdump_gemm_wrap_tile_sm120.sass` |
| `gemm_dbuffer_vload_kernel` | `ai_system_gemm_lab.5.sm_120.cubin` | `cuobjdump_gemm_dbuffer_vload_sm120.sass` |
| `tiled_gemm_register_kernel` | `ai_system_gemm_lab.6.sm_120.cubin` | `cuobjdump_tiled_gemm_register_sm120.sass` |
| `tiled_gemm_block_kernel` | `ai_system_gemm_lab.7.sm_120.cubin` | `cuobjdump_tiled_gemm_block_sm120.sass` |

### 方法二：dump 整个 `.lib` 后按 member 切分

本次已经生成的 `out\sass\cuobjdump_tiled_gemm_block_sm120.sass` 就是用这种方式得到的：先把整个 `.lib` dump 成 `cuobjdump_ai_system_gemm_lab_sm120.sass`，再按 `member ... tiled_gemm_block.obj:` 这一段切出来。

这种方式的好处是一次完整 dump 后可以批量拆出所有 family；缺点是完整 dump 文件很大。

## 抽取真正单个模板实例

如果只想看一个具体模板实例，例如 `tiled_gemm_block_kernel<128, 8, 32>`，需要使用完整 mangled symbol name。先从符号表中找出目标实例：

```powershell
$mangled = (
    Select-String -Path out\sass\cuobjdump_ai_system_gemm_lab_elf_symbols.txt -Pattern 'tiled_gemm_block_kernelILi128ELi8ELi32' |
        Select-Object -First 1
).Line -replace '^.*STO_ENTRY\s+', ''
```

再用 `--function` 只 dump 这个 kernel 实例：

```powershell
cuobjdump --dump-sass --gpu-architecture sm_120 --function $mangled out\build\windows-vs2022-cuda-release\labs\gemm\Release\ai_system_gemm_lab.lib |
    Out-File -FilePath out\sass\cuobjdump_tiled_gemm_block_128x8x32_sm120.sass -Encoding ascii
```

如果 `--function` 没有输出，通常是 symbol name 没有完全匹配。先确认 `$mangled` 不为空，并且来自 `STO_ENTRY` 行。

## 拆分完整 SASS 的 PowerShell 脚本

完整 dump 中每个 archive member 以 `member ... .obj:` 开头。下面脚本按 member 名切出当前 GEMM kernel family：

```powershell
$inputPath = 'out\sass\cuobjdump_ai_system_gemm_lab_sm120.sass'
$outDir = 'out\sass'
$targets = @{
  'sgemm_v3.obj' = 'cuobjdump_sgemm_v3_sm120.sass'
  'sgemm_v1.obj' = 'cuobjdump_sgemm_v1_sm120.sass'
  'gemm_wrap_tile.obj' = 'cuobjdump_gemm_wrap_tile_sm120.sass'
  'gemm_dbuffer_vload.obj' = 'cuobjdump_gemm_dbuffer_vload_sm120.sass'
  'tiled_gemm_register.obj' = 'cuobjdump_tiled_gemm_register_sm120.sass'
  'tiled_gemm_block.obj' = 'cuobjdump_tiled_gemm_block_sm120.sass'
}

$reader = [System.IO.StreamReader]::new($inputPath)
$writer = $null
try {
  while (($line = $reader.ReadLine()) -ne $null) {
    if ($line.StartsWith('member ')) {
      if ($writer -ne $null) {
        $writer.Dispose()
        $writer = $null
      }
      foreach ($key in $targets.Keys) {
        if ($line.Contains($key)) {
          $outputPath = Join-Path $outDir $targets[$key]
          $writer = [System.IO.StreamWriter]::new($outputPath, $false, [System.Text.Encoding]::ASCII)
          break
        }
      }
    }
    if ($writer -ne $null) {
      $writer.WriteLine($line)
    }
  }
}
finally {
  if ($writer -ne $null) { $writer.Dispose() }
  $reader.Dispose()
}
```

## 分析建议

用 `Select-String` 查某一类指令：

```powershell
Select-String -Path out\sass\cuobjdump_tiled_gemm_register_sm120.sass -Pattern 'FFMA|LDG|STS|BAR|BRA'
```

统计 kernel 实例数量：

```powershell
Select-String -Path out\sass\cuobjdump_*_sm120.sass -Pattern 'Function :' |
    Group-Object Path |
    Select-Object Name,Count
```

查看资源占用中的某个 kernel family：

```powershell
Select-String -Path out\sass\cuobjdump_ai_system_gemm_lab_resource_usage.txt -Pattern 'tiled_gemm_register_kernel'
```

如果只想 dump 某个具体 kernel，需要先从 `cuobjdump_ai_system_gemm_lab_elf_symbols.txt` 中找到完整 mangled name，再使用：

```powershell
cuobjdump --dump-sass --function <mangled-kernel-name> <binary-or-library>
```
