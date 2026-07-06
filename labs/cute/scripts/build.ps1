param(
    [string]$Target = "cute_layout_mapping",
    [string]$Configuration = "Release"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$preset = "windows-vs2022-cuda-release"

Write-Host "cmake --build --preset $preset --config $Configuration --target $Target"
cmake --build --preset $preset --config $Configuration --target $Target
