#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

preset="linux-make-cuda-release"
output_path=""
trace="cuda,nvtx,osrt"
sample="none"
tool_path=""
executable_path=""
vector_size=1048576
reduction_size=1048576
gemm_m=256
gemm_n=256
gemm_k=256
warmup=2
iters=5
dry_run=0

resolve_default_tool_path() {
    if command -v nsys >/dev/null 2>&1; then
        command -v nsys
        return 0
    fi

    if [[ -x "/usr/local/cuda/bin/nsys" ]]; then
        printf '%s\n' "/usr/local/cuda/bin/nsys"
        return 0
    fi

    local candidate
    for candidate in /usr/local/cuda-*/bin/nsys; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

print_usage() {
    cat <<'EOF'
Usage: ./scripts/profile_nsys.sh [options]

Options:
  --preset NAME            Build preset to profile (default: linux-make-cuda-release)
  --output PATH            Report prefix path
  --trace LIST             Nsight Systems trace list (default: cuda,nvtx,osrt)
  --sample MODE            Sampling mode (default: none)
  --tool PATH              Override nsys executable path
  --exe PATH               Override profiled executable path
  --vector-size N          Vector add input length
  --reduction-size N       Reduction input length
  --gemm-m N               GEMM M dimension
  --gemm-n N               GEMM N dimension
  --gemm-k N               GEMM K dimension
  --warmup N               Warmup iterations
  --iters N                Measured iterations
  --dry-run                Print command without executing
  --help                   Show this help message
EOF
}

while (($# > 0)); do
    case "$1" in
        --preset) preset="$2"; shift 2 ;;
        --output) output_path="$2"; shift 2 ;;
        --trace) trace="$2"; shift 2 ;;
        --sample) sample="$2"; shift 2 ;;
        --tool) tool_path="$2"; shift 2 ;;
        --exe) executable_path="$2"; shift 2 ;;
        --vector-size) vector_size="$2"; shift 2 ;;
        --reduction-size) reduction_size="$2"; shift 2 ;;
        --gemm-m) gemm_m="$2"; shift 2 ;;
        --gemm-n) gemm_n="$2"; shift 2 ;;
        --gemm-k) gemm_k="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --help) print_usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ "${vector_size}" == "0" || "${reduction_size}" == "0" || "${gemm_m}" == "0" || "${gemm_n}" == "0" || "${gemm_k}" == "0" || "${warmup}" == "0" || "${iters}" == "0" ]]; then
    echo "All size and iteration arguments must be positive integers." >&2
    exit 1
fi

if [[ -z "${tool_path}" ]]; then
    if ! tool_path="$(resolve_default_tool_path)"; then
        echo "Unable to locate nsys. Install Nsight Systems or pass --tool." >&2
        exit 1
    fi
fi

if [[ -z "${executable_path}" ]]; then
    executable_path="${repo_root}/out/build/${preset}/labs/perf_engineering/perf_engineering_lab"
fi

if [[ -z "${output_path}" ]]; then
    output_path="${repo_root}/out/build/${preset}/nsight/nsys-perf-engineering"
fi

if [[ ! -x "${tool_path}" ]]; then
    echo "Nsight Systems CLI not found at '${tool_path}'." >&2
    exit 1
fi

if [[ ! -x "${executable_path}" ]]; then
    echo "Target executable not found at '${executable_path}'. Build the preset first or pass --exe." >&2
    exit 1
fi

mkdir -p "$(dirname "${output_path}")"

cmd=(
    "${tool_path}"
    profile
    "--sample=${sample}"
    "--cpuctxsw=none"
    "--trace=${trace}"
    "--force-overwrite=true"
    -o "${output_path}"
    "${executable_path}"
    --vector-size "${vector_size}"
    --reduction-size "${reduction_size}"
    --gemm-m "${gemm_m}"
    --gemm-n "${gemm_n}"
    --gemm-k "${gemm_k}"
    --warmup "${warmup}"
    --iters "${iters}"
)

printf 'Nsight Systems CLI: %s\n' "${tool_path}"
printf 'Profiling target   : %s\n' "${executable_path}"
printf 'Report prefix      : %s\n' "${output_path}"
printf 'Command            : '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${dry_run}" == "1" ]]; then
    exit 0
fi

"${cmd[@]}"
