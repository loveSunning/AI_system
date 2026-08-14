param(
    [string]$ProfilerPath = "",
    [string]$Operation = "Gemm",
    [string]$Kernel = "cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8",
    [string]$M = "4096",
    [string]$N = "4096",
    [string]$K = "4096",
    [string]$InputType = "f16",
    [string]$OutputType = "f32",
    [ValidateSet("row", "column")]
    [string]$OutputLayout = "column",
    [int]$WarmupIterations = 10,
    [int]$ProfilingIterations = 20
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not $ProfilerPath) {
    $ProfilerPath = Join-Path $repoRoot "3rdparty\cutlass\build\windows-vs2022-5060\tools\profiler\Release\cutlass_profiler.exe"
}

if (-not (Test-Path $ProfilerPath)) {
    throw "cutlass_profiler was not found at: $ProfilerPath. Run labs\cutlass\scripts\configure_official_cutlass.ps1 and labs\cutlass\scripts\build_official_cutlass.ps1 first."
}

# Visual Studio places cutlass.dll in tools/library/Release instead of beside
# cutlass_profiler.exe. CUDA 13 places cuBLAS DLLs in CUDA_PATH/bin/x64.
$profilerDir = Split-Path (Resolve-Path $ProfilerPath) -Parent
$cutlassDllDir = Resolve-Path (Join-Path $profilerDir "..\..\library\Release")
$runtimeDirs = @($cutlassDllDir)
if ($env:CUDA_PATH) {
    $cudaDllDir = Join-Path $env:CUDA_PATH "bin\x64"
    if (Test-Path $cudaDllDir) {
        $runtimeDirs += $cudaDllDir
    }
}
$env:PATH = ($runtimeDirs -join ";") + ";" + $env:PATH

& $ProfilerPath `
    --operation=$Operation `
    --kernels=$Kernel `
    --providers=cutlass `
    --m=$M `
    --n=$N `
    --k=$K `
    "--A=${InputType}:row" `
    "--B=${InputType}:column" `
    "--C=${OutputType}:${OutputLayout}" `
    "--D=${OutputType}:${OutputLayout}" `
    --accum=f32 `
    --op_class=tensorop `
    --alpha=1 `
    --beta=0 `
    --verification-enabled=true `
    --warmup-iterations=$WarmupIterations `
    --profiling-iterations=$ProfilingIterations `
    --verification-providers=cublas
