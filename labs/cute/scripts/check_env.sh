#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CUTLASS_ROOT_ARG="${1:-${CUTLASS_ROOT:-${REPO_ROOT}/3rdparty/cutlass}}"
CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-}}"

echo "[CuTe] environment check"
echo "OS: $(uname -a)"

for tool in cmake nvcc nvidia-smi git; do
  if command -v "${tool}" >/dev/null 2>&1; then
    echo "${tool}: $(command -v "${tool}")"
  else
    echo "${tool}: not found"
  fi
done

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | sed -n '1,4p'
  if [[ -z "${CUDA_ROOT}" ]]; then
    NVCC_PATH="$(command -v nvcc)"
    CUDA_ROOT="$(cd "$(dirname "${NVCC_PATH}")/.." && pwd)"
  fi
  if nvcc --list-gpu-code 2>/dev/null | grep -q '89'; then
    echo "nvcc sm_89 support: yes"
  else
    echo "nvcc sm_89 support: not listed"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
fi

echo "CUDA Toolkit root: ${CUDA_ROOT:-<unset>}"
if [[ -n "${CUDA_ROOT}" && -f "${CUDA_ROOT}/include/cutlass/cutlass.h" ]]; then
  echo "CUDA Toolkit CUTLASS header: found"
else
  echo "CUDA Toolkit CUTLASS header: missing"
fi

if [[ -n "${CUDA_ROOT}" && -f "${CUDA_ROOT}/include/cute/tensor.hpp" ]]; then
  echo "CUDA Toolkit CuTe header: found"
else
  echo "CUDA Toolkit CuTe header: missing"
fi

echo "CUTLASS_ROOT: ${CUTLASS_ROOT_ARG:-<unset>}"
if [[ -n "${CUTLASS_ROOT_ARG}" && -f "${CUTLASS_ROOT_ARG}/include/cute/tensor.hpp" ]]; then
  echo "CuTe headers: found"
else
  echo "CuTe headers: missing include/cute/tensor.hpp"
fi
