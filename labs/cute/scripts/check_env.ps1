param(
    [string]$CutlassRoot = $env:CUTLASS_ROOT
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not $CutlassRoot) {
    $CutlassRoot = Join-Path $repoRoot "3rdparty\cutlass"
}

Write-Host "[CuTe] environment check"
Write-Host "OS: $([System.Environment]::OSVersion.VersionString)"

foreach ($tool in @("cmake", "nvcc", "nvidia-smi", "git")) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host "${tool}: $($cmd.Source)"
    } else {
        Write-Host "${tool}: not found"
    }
}

$cudaRoot = $env:CUDA_PATH
$nvccCommand = Get-Command nvcc -ErrorAction SilentlyContinue
if ((-not $cudaRoot) -and $nvccCommand) {
    $cudaRoot = Split-Path -Parent (Split-Path -Parent $nvccCommand.Source)
}

if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    nvcc --version | Select-Object -First 4
    $gpuCode = nvcc --list-gpu-code 2>$null
    if ($LASTEXITCODE -eq 0) {
        if ($gpuCode -match "120") {
            Write-Host "nvcc sm_120 support: yes"
        } else {
            Write-Host "nvcc sm_120 support: not listed"
        }
    }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
}

$cudaRootDisplay = "<unset>"
if ($cudaRoot) {
    $cudaRootDisplay = $cudaRoot
}
Write-Host "CUDA Toolkit root: $cudaRootDisplay"
$cudaCutlassHeader = if ($cudaRoot) { Join-Path $cudaRoot "include/cutlass/cutlass.h" } else { "" }
$cudaCuteHeader = if ($cudaRoot) { Join-Path $cudaRoot "include/cute/tensor.hpp" } else { "" }
$cudaCutlassState = "missing"
if ($cudaCutlassHeader -and (Test-Path $cudaCutlassHeader)) {
    $cudaCutlassState = "found"
}
$cudaCuteState = "missing"
if ($cudaCuteHeader -and (Test-Path $cudaCuteHeader)) {
    $cudaCuteState = "found"
}
Write-Host "CUDA Toolkit CUTLASS header: $cudaCutlassState"
Write-Host "CUDA Toolkit CuTe header: $cudaCuteState"

$cutlassRootDisplay = "<unset>"
if ($CutlassRoot) {
    $cutlassRootDisplay = $CutlassRoot
}
Write-Host "CUTLASS_ROOT: $cutlassRootDisplay"

$cuteHeader = if ($CutlassRoot) { Join-Path $CutlassRoot "include/cute/tensor.hpp" } else { "" }
if ($cuteHeader -and (Test-Path $cuteHeader)) {
    Write-Host "CuTe headers: found"
} else {
    Write-Host "CuTe headers: missing include/cute/tensor.hpp"
}
