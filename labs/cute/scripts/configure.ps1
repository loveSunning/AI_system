param(
    [string]$CutlassRoot = $env:CUTLASS_ROOT
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$preset = "windows-vs2022-cuda-release"

$args = @("-S", $repoRoot, "--preset", $preset)
if ($CutlassRoot) {
    $args += "-DAI_SYSTEM_CUTLASS_ROOT=$CutlassRoot"
}

Write-Host "cmake $($args -join ' ')"
cmake @args
