param(
    [string]$Preset = "windows-vs2022-cuda-release",
    [string]$Configuration = "Release",
    [string]$OutputPath = "",
    [string]$ToolPath = "",
    [string]$ExecutablePath = "",
    [string]$KernelRegex = "vector_add_kernel",
    [string]$SetName = "basic",
    [string]$TargetProcesses = "all",
    [UInt64]$LaunchCount = 1,
    [UInt64]$VectorSize = 1048576,
    [UInt64]$ReductionSize = 1024,
    [UInt64]$GemmM = 32,
    [UInt64]$GemmN = 32,
    [UInt64]$GemmK = 32,
    [UInt64]$Warmup = 1,
    [UInt64]$Iters = 1,
    [switch]$NoExport,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

function Resolve-NcuTool {
    param([string]$RequestedPath)

    if($RequestedPath) {
        return [System.IO.Path]::GetFullPath($RequestedPath)
    }

    $candidateRoots = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Directory -Filter "Nsight Compute *" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending

    foreach($root in $candidateRoots) {
        $candidate = Join-Path $root.FullName "ncu.bat"
        if(Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Unable to locate ncu.bat. Install Nsight Compute or pass -ToolPath."
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

    return Join-Path $RepoRoot "out/build/$PresetName/nsight/ncu-perf-engineering"
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

if($LaunchCount -eq 0 -or $VectorSize -eq 0 -or $ReductionSize -eq 0 -or $GemmM -eq 0 -or $GemmN -eq 0 -or $GemmK -eq 0 -or $Warmup -eq 0 -or $Iters -eq 0) {
    throw "All size and iteration arguments must be positive integers."
}

$repoRoot = Resolve-RepoRoot
$resolvedTool = Resolve-NcuTool -RequestedPath $ToolPath
$resolvedExecutable = Resolve-Executable -RequestedPath $ExecutablePath -RepoRoot $repoRoot -PresetName $Preset -BuildConfiguration $Configuration
$resolvedOutput = Resolve-OutputPath -RequestedPath $OutputPath -RepoRoot $repoRoot -PresetName $Preset

if(!(Test-Path $resolvedTool)) {
    throw "Nsight Compute CLI not found at '$resolvedTool'."
}

if(!(Test-Path $resolvedExecutable)) {
    throw "Target executable not found at '$resolvedExecutable'. Build the preset first or pass -ExecutablePath."
}

$outputDirectory = Split-Path -Parent $resolvedOutput
if($outputDirectory) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$command = @(
    $resolvedTool,
    "--set", $SetName,
    "--target-processes", $TargetProcesses,
    "--launch-count", $LaunchCount
)

if($KernelRegex) {
    $command += @("--kernel-name", "regex:$KernelRegex")
}

if(!$NoExport) {
    $command += @("--export", $resolvedOutput)
}

$command += @(
    $resolvedExecutable,
    "--vector-size", $VectorSize,
    "--reduction-size", $ReductionSize,
    "--gemm-m", $GemmM,
    "--gemm-n", $GemmN,
    "--gemm-k", $GemmK,
    "--warmup", $Warmup,
    "--iters", $Iters
)

Write-Host "Nsight Compute CLI: $resolvedTool"
Write-Host "Profiling target   : $resolvedExecutable"
if(!$NoExport) {
    Write-Host "Report prefix      : $resolvedOutput"
}
Write-Host "Command            : $(Format-CommandLine $command)"

if($DryRun) {
    return
}

$previousHome = $env:HOME
$env:HOME = $outputDirectory

try {
    if($KernelRegex -and -not $NoExport) {
        & $resolvedTool `
            --set $SetName `
            --target-processes $TargetProcesses `
            --launch-count $LaunchCount `
            --kernel-name "regex:$KernelRegex" `
            --export $resolvedOutput `
            $resolvedExecutable `
            --vector-size $VectorSize `
            --reduction-size $ReductionSize `
            --gemm-m $GemmM `
            --gemm-n $GemmN `
            --gemm-k $GemmK `
            --warmup $Warmup `
            --iters $Iters
    } elseif($KernelRegex) {
        & $resolvedTool `
            --set $SetName `
            --target-processes $TargetProcesses `
            --launch-count $LaunchCount `
            --kernel-name "regex:$KernelRegex" `
            $resolvedExecutable `
            --vector-size $VectorSize `
            --reduction-size $ReductionSize `
            --gemm-m $GemmM `
            --gemm-n $GemmN `
            --gemm-k $GemmK `
            --warmup $Warmup `
            --iters $Iters
    } elseif(!$NoExport) {
        & $resolvedTool `
            --set $SetName `
            --target-processes $TargetProcesses `
            --launch-count $LaunchCount `
            --export $resolvedOutput `
            $resolvedExecutable `
            --vector-size $VectorSize `
            --reduction-size $ReductionSize `
            --gemm-m $GemmM `
            --gemm-n $GemmN `
            --gemm-k $GemmK `
            --warmup $Warmup `
            --iters $Iters
    } else {
        & $resolvedTool `
            --set $SetName `
            --target-processes $TargetProcesses `
            --launch-count $LaunchCount `
            $resolvedExecutable `
            --vector-size $VectorSize `
            --reduction-size $ReductionSize `
            --gemm-m $GemmM `
            --gemm-n $GemmN `
            --gemm-k $GemmK `
            --warmup $Warmup `
            --iters $Iters
    }
} finally {
    $env:HOME = $previousHome
}
