param(
    [string]$Preset = "windows-vs2022-cuda-release",
    [string]$GpuProfile = "native"
)

cmake --preset $Preset "-DAI_SYSTEM_GPU_PROFILE=$GpuProfile"
