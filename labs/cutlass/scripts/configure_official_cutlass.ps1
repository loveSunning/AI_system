param(
    [string]$CutlassRoot = $env:CUTLASS_ROOT,
    [string]$BuildDir = "build\windows-vs2022-5060"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not $CutlassRoot) {
    $CutlassRoot = Join-Path $repoRoot "3rdparty\cutlass"
}

$cutlassCmake = Join-Path $CutlassRoot "CMakeLists.txt"
if (-not (Test-Path $cutlassCmake)) {
    throw "CUTLASS CMakeLists.txt was not found at $cutlassCmake"
}

$buildPath = Join-Path $CutlassRoot $BuildDir

cmake `
    -S $CutlassRoot `
    -B $buildPath `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCUTLASS_NVCC_ARCHS=120 `
    -DCUTLASS_ENABLE_TESTS=OFF `
    -DCUTLASS_ENABLE_CUBLAS=ON `
    -DCUTLASS_LIBRARY_OPERATIONS=gemm `
    "-DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_f16_*" `
    "-DCUTLASS_LIBRARY_IGNORE_KERNELS=cutlass_tensorop_s16816gemm_f16_s8_*,cutlass_tensorop_s16816gemm_f16_u8_*" `
    -DCUTLASS_PROFILER_DISABLE_REFERENCE=ON `
    -DCUTLASS_UNITY_BUILD_ENABLED=ON

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
