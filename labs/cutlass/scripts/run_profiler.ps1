param(
    [Parameter(Mandatory = $true)]
    [string]$ProfilerPath,
    [string]$Operation = "Gemm",
    [string]$M = "4096",
    [string]$N = "4096",
    [string]$K = "4096",
    [string]$DataType = "f16"
)

if (-not (Test-Path $ProfilerPath)) {
    throw "cutlass_profiler was not found at: $ProfilerPath"
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
