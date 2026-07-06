param(
    [string]$Target = "cute_layout_mapping",
    [string]$Configuration = "Release"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$preset = "windows-vs2022-cuda-release"

Write-Host "cmake --build --preset $preset --config $Configuration --target $Target"
Push-Location $repoRoot
try {
    cmake --build --preset $preset --config $Configuration --target $Target
} finally {
    Pop-Location
}
