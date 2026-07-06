param(
    [string]$Target = "cutlass_header_probe",
    [string]$Configuration = "Release"
)

$preset = "windows-vs2022-cuda-release"

cmake --build --preset $preset --config $Configuration --target $Target
