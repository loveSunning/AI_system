#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CUTLASS_ROOT_ARG="${CUTLASS_ROOT:-${REPO_ROOT}/3rdparty/cutlass}"
BUILD_DIR="build/linux-4090d"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cutlass-root)
      CUTLASS_ROOT_ARG="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CUTLASS_ROOT_ARG}/CMakeLists.txt" ]]; then
  echo "CUTLASS CMakeLists.txt was not found at ${CUTLASS_ROOT_ARG}/CMakeLists.txt" >&2
  exit 2
fi

cmake \
  -S "${CUTLASS_ROOT_ARG}" \
  -B "${CUTLASS_ROOT_ARG}/${BUILD_DIR}" \
  -DCUTLASS_NVCC_ARCHS=89 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
