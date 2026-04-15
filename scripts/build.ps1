param(
    [string]$Preset = "windows-vs2022-cuda-release",
    [string]$Configuration = "Release"
)

cmake --build --preset $Preset --config $Configuration
