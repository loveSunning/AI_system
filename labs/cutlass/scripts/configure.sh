#!/usr/bin/env bash
set -euo pipefail

PROFILE="4090d"
CUTLASS_ROOT_ARG="${CUTLASS_ROOT:-}"

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
  echo "CUTLASS_ROOT is required. Pass --cutlass-root /path/to/cutlass or export CUTLASS_ROOT." >&2
  exit 2
fi

cmake --preset "${PRESET}" -DAI_SYSTEM_CUTLASS_ROOT="${CUTLASS_ROOT_ARG}"
