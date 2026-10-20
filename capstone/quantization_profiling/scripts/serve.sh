#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/run_model.py" --serve --backend vllm --dtype bfloat16 "$@"
