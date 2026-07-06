param(
    [string]$Target = "cutlass_header_probe",
    [string]$Configuration = "Release"
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$preset = "windows-vs2022-cuda-release"

Push-Location $repoRoot
try {
    cmake --build --preset $preset --config $Configuration --target $Target
} finally {
    Pop-Location
}
