#!/usr/bin/env bash
set -euo pipefail

preset="${1:-linux-make-cuda-release}"
cmake --build --preset "${preset}"
