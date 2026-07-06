param(
    [string]$CutlassRoot = $env:CUTLASS_ROOT
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not $CutlassRoot) {
    $CutlassRoot = Join-Path $repoRoot "3rdparty\cutlass"
}

$cuteHeader = Join-Path $CutlassRoot "include\cute\tensor.hpp"
if (-not (Test-Path $cuteHeader)) {
    throw "CuTe header was not found at $cuteHeader"
}

$preset = "windows-vs2022-cuda-release"

$args = @("-S", $repoRoot, "--preset", $preset, "-DAI_SYSTEM_CUTLASS_ROOT=$CutlassRoot")

Write-Host "cmake $($args -join ' ')"
cmake @args
