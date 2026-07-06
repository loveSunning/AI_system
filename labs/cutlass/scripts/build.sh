#!/usr/bin/env bash
set -euo pipefail

PROFILE="4090d"
TARGET="cutlass_header_probe"
CONFIGURATION="Release"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --config)
      CONFIGURATION="$2"
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

cmake --build --preset "${PRESET}" --config "${CONFIGURATION}" --target "${TARGET}"
