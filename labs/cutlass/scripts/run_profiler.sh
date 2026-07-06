#!/usr/bin/env bash
set -euo pipefail

PROFILER=""
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

if [[ -z "${PROFILER}" || ! -x "${PROFILER}" ]]; then
  echo "cutlass_profiler was not found or is not executable: ${PROFILER:-<unset>}" >&2
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
