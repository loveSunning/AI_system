#!/usr/bin/env bash
set -euo pipefail

PROFILE="4090d"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CUTLASS_ROOT_ARG="${CUTLASS_ROOT:-${REPO_ROOT}/3rdparty/cutlass}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --cutlass-root)
      CUTLASS_ROOT_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${PROFILE}" != "4090d" ]]; then
  echo "Unsupported Linux profile: ${PROFILE}" >&2
  echo "This lab's Linux scripts target RTX 4090D only." >&2
  exit 2
fi

PRESET="linux-make-cuda-release"

if [[ -z "${CUTLASS_ROOT_ARG}" ]]; then
  echo "CUTLASS root path is empty." >&2
  exit 2
fi

if [[ ! -f "${CUTLASS_ROOT_ARG}/include/cute/tensor.hpp" ]]; then
  echo "CuTe header was not found at ${CUTLASS_ROOT_ARG}/include/cute/tensor.hpp" >&2
  exit 2
fi

cmake -S "${REPO_ROOT}" --preset "${PRESET}" -DAI_SYSTEM_CUTLASS_ROOT="${CUTLASS_ROOT_ARG}"
