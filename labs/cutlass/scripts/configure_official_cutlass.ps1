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
    -DCUTLASS_UNITY_BUILD_ENABLED=ON

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
