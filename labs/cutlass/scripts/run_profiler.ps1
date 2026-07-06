param(
    [string]$ProfilerPath = "",
    [string]$Operation = "Gemm",
    [string]$M = "4096",
    [string]$N = "4096",
    [string]$K = "4096",
    [string]$DataType = "f16"
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not $ProfilerPath) {
    $ProfilerPath = Join-Path $repoRoot "3rdparty\cutlass\build\windows-vs2022-5060\tools\profiler\Release\cutlass_profiler.exe"
}

if (-not (Test-Path $ProfilerPath)) {
    throw "cutlass_profiler was not found at: $ProfilerPath. Run labs\cutlass\scripts\configure_official_cutlass.ps1 and labs\cutlass\scripts\build_official_cutlass.ps1 first."
}

& $ProfilerPath `
    --operation=$Operation `
    --m=$M `
    --n=$N `
    --k=$K `
    --A=$DataType `
    --B=$DataType `
    --C=$DataType `
    --accumulator=f32
