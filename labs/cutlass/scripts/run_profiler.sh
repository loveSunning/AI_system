#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROFILER="${REPO_ROOT}/3rdparty/cutlass/build/linux-4090d/tools/profiler/cutlass_profiler"
OPERATION="Gemm"
KERNEL="cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8"
M="4096"
N="4096"
K="4096"
INPUT_TYPE="f16"
OUTPUT_TYPE="f32"
OUTPUT_LAYOUT="column"
WARMUP_ITERATIONS="10"
PROFILING_ITERATIONS="20"

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
    --kernel)
      KERNEL="$2"
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
      INPUT_TYPE="$2"
      shift 2
      ;;
    --output-dtype)
      OUTPUT_TYPE="$2"
      shift 2
      ;;
    --output-layout)
      OUTPUT_LAYOUT="$2"
      shift 2
      ;;
    --warmup-iterations)
      WARMUP_ITERATIONS="$2"
      shift 2
      ;;
    --profiling-iterations)
      PROFILING_ITERATIONS="$2"
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
  --kernels="${KERNEL}" \
  --providers=cutlass \
  --m="${M}" \
  --n="${N}" \
  --k="${K}" \
  --A="${INPUT_TYPE}:row" \
  --B="${INPUT_TYPE}:column" \
  --C="${OUTPUT_TYPE}:${OUTPUT_LAYOUT}" \
  --D="${OUTPUT_TYPE}:${OUTPUT_LAYOUT}" \
  --accum=f32 \
  --op_class=tensorop \
  --alpha=1 \
  --beta=0 \
  --verification-enabled=true \
  --warmup-iterations="${WARMUP_ITERATIONS}" \
  --profiling-iterations="${PROFILING_ITERATIONS}" \
  --verification-providers=cublas
