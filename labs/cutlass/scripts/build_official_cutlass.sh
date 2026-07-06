#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CUTLASS_ROOT_ARG="${CUTLASS_ROOT:-${REPO_ROOT}/3rdparty/cutlass}"
BUILD_DIR="build/linux-4090d"
TARGET="cutlass_profiler"

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
    --target)
      TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

BUILD_PATH="${CUTLASS_ROOT_ARG}/${BUILD_DIR}"
if [[ ! -f "${BUILD_PATH}/CMakeCache.txt" ]]; then
  echo "CUTLASS build tree was not configured at ${BUILD_PATH}." >&2
  echo "Run: bash ./labs/cutlass/scripts/configure_official_cutlass.sh" >&2
  exit 2
fi

cmake --build "${BUILD_PATH}" --target "${TARGET}" -j
