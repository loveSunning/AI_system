#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROFILER="${REPO_ROOT}/3rdparty/cutlass/build/linux-4090d/tools/profiler/cutlass_profiler"
OPERATION="Gemm"
M="4096"
N="4096"
K="4096"
DATA_TYPE="f16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiler)
      PROFILER="$2"
      shift 2
      ;;
    --operation)
      OPERATION="$2"
      shift 2
      ;;
    --m)
      M="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --dtype)
      DATA_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${PROFILER}" ]]; then
  echo "cutlass_profiler was not found: ${PROFILER}" >&2
  echo "Run: bash ./labs/cutlass/scripts/configure_official_cutlass.sh" >&2
  echo "Then: bash ./labs/cutlass/scripts/build_official_cutlass.sh" >&2
  exit 2
fi

if [[ ! -x "${PROFILER}" ]]; then
  echo "cutlass_profiler exists but is not executable: ${PROFILER}" >&2
  echo "Try: chmod +x ${PROFILER}" >&2
  exit 2
fi

"${PROFILER}" \
  --operation="${OPERATION}" \
  --m="${M}" \
  --n="${N}" \
  --k="${K}" \
  --A="${DATA_TYPE}" \
  --B="${DATA_TYPE}" \
  --C="${DATA_TYPE}" \
  --accumulator=f32
