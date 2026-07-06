param(
    [string]$CutlassRoot = $env:CUTLASS_ROOT,
    [string]$BuildDir = "build\windows-vs2022-5060",
    [string]$Configuration = "Release",
    [string]$Target = "cutlass_profiler"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not $CutlassRoot) {
    $CutlassRoot = Join-Path $repoRoot "3rdparty\cutlass"
}

$buildPath = Join-Path $CutlassRoot $BuildDir
if (-not (Test-Path (Join-Path $buildPath "CMakeCache.txt"))) {
    throw "CUTLASS build tree was not configured at $buildPath. Run configure_official_cutlass.ps1 first."
}

cmake --build $buildPath --config $Configuration --target $Target

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
