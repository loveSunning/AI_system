param(
    [string]$CutlassRoot = $env:CUTLASS_ROOT
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not $CutlassRoot) {
    $CutlassRoot = Join-Path $repoRoot "3rdparty\cutlass"
}

$cutlassHeader = Join-Path $CutlassRoot "include\cutlass\cutlass.h"
if (-not (Test-Path $cutlassHeader)) {
    throw "CUTLASS header was not found at $cutlassHeader"
}

$preset = "windows-vs2022-cuda-release"

cmake -S $repoRoot --preset $preset -DAI_SYSTEM_CUTLASS_ROOT="$CutlassRoot"
