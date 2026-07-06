param(
    [string]$CutlassRoot = $env:CUTLASS_ROOT
)

if (-not $CutlassRoot) {
    throw "CUTLASS_ROOT is required. Pass -CutlassRoot D:\deps\cutlass or set the CUTLASS_ROOT environment variable."
}

$preset = "windows-vs2022-cuda-release"

cmake --preset $preset -DAI_SYSTEM_CUTLASS_ROOT="$CutlassRoot"
