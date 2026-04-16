param(
    [string]$Preset = "windows-vs2022-cuda-release",
    [string]$Configuration = "Release",
    [string]$OutputPath = "",
    [string]$Trace = "cuda,nvtx",
    [string]$Sample = "none",
    [string]$ToolPath = "",
    [string]$ExecutablePath = "",
    [UInt64]$VectorSize = 1048576,
    [UInt64]$ReductionSize = 1048576,
    [UInt64]$GemmM = 1024,
    [UInt64]$GemmN = 1024,
    [UInt64]$GemmK = 1024,
    [UInt64]$Warmup = 2,
    [UInt64]$Iters = 5,
    [switch]$EnableWddm,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

function Resolve-NsysTool {
    param([string]$RequestedPath)

    if($RequestedPath) {
        return [System.IO.Path]::GetFullPath($RequestedPath)
    }

    $candidateRoots = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Directory -Filter "Nsight Systems *" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending

    foreach($root in $candidateRoots) {
        $candidate = Join-Path $root.FullName "target-windows-x64\nsys.exe"
        if(Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Unable to locate nsys.exe. Install Nsight Systems or pass -ToolPath."
}

function Resolve-Executable {
    param(
        [string]$RequestedPath,
        [string]$RepoRoot,
        [string]$PresetName,
        [string]$BuildConfiguration
    )

    if($RequestedPath) {
        return [System.IO.Path]::GetFullPath($RequestedPath)
    }

    return Join-Path $RepoRoot "out/build/$PresetName/labs/perf_engineering/$BuildConfiguration/perf_engineering_lab.exe"
}

function Resolve-OutputPath {
    param(
        [string]$RequestedPath,
        [string]$RepoRoot,
        [string]$PresetName
    )

    if($RequestedPath) {
        return [System.IO.Path]::GetFullPath($RequestedPath)
    }

    return Join-Path $RepoRoot "out/build/$PresetName/nsight/nsys-perf-engineering"
}

function Format-CommandLine {
    param([string[]]$Command)

    return ($Command | ForEach-Object {
        if($_ -match '\s') {
            '"' + $_ + '"'
        } else {
            $_
        }
    }) -join ' '
}

if($VectorSize -eq 0 -or $ReductionSize -eq 0 -or $GemmM -eq 0 -or $GemmN -eq 0 -or $GemmK -eq 0 -or $Warmup -eq 0 -or $Iters -eq 0) {
    throw "All size and iteration arguments must be positive integers."
}

$repoRoot = Resolve-RepoRoot
$resolvedTool = Resolve-NsysTool -RequestedPath $ToolPath
$resolvedExecutable = Resolve-Executable -RequestedPath $ExecutablePath -RepoRoot $repoRoot -PresetName $Preset -BuildConfiguration $Configuration
$resolvedOutput = Resolve-OutputPath -RequestedPath $OutputPath -RepoRoot $repoRoot -PresetName $Preset

if(!(Test-Path $resolvedTool)) {
    throw "Nsight Systems CLI not found at '$resolvedTool'."
}

if(!(Test-Path $resolvedExecutable)) {
    throw "Target executable not found at '$resolvedExecutable'. Build the preset first or pass -ExecutablePath."
}

$traceValue = $Trace
if($EnableWddm -and $traceValue -notmatch '(^|,)wddm(,|$)') {
    $traceValue = "$traceValue,wddm"
}

$outputDirectory = Split-Path -Parent $resolvedOutput
if($outputDirectory) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$command = @(
    $resolvedTool,
    "profile",
    "--sample=$Sample",
    "--cpuctxsw=none",
    "--trace=$traceValue",
    "--force-overwrite=true",
    "-o", $resolvedOutput,
    $resolvedExecutable,
    "--vector-size", $VectorSize,
    "--reduction-size", $ReductionSize,
    "--gemm-m", $GemmM,
    "--gemm-n", $GemmN,
    "--gemm-k", $GemmK,
    "--warmup", $Warmup,
    "--iters", $Iters
)

Write-Host "Nsight Systems CLI: $resolvedTool"
Write-Host "Profiling target   : $resolvedExecutable"
Write-Host "Report prefix      : $resolvedOutput"
Write-Host "Command            : $(Format-CommandLine $command)"

if($DryRun) {
    return
}

& $resolvedTool `
    profile `
    "--sample=$Sample" `
    "--cpuctxsw=none" `
    "--trace=$traceValue" `
    "--force-overwrite=true" `
    -o $resolvedOutput `
    $resolvedExecutable `
    --vector-size $VectorSize `
    --reduction-size $ReductionSize `
    --gemm-m $GemmM `
    --gemm-n $GemmN `
    --gemm-k $GemmK `
    --warmup $Warmup `
    --iters $Iters
