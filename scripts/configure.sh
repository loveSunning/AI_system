#!/usr/bin/env bash
set -euo pipefail

preset="${1:-linux-make-cuda-release}"
gpu_profile="${2:-native}"

cmake --preset "${preset}" "-DAI_SYSTEM_GPU_PROFILE=${gpu_profile}"
