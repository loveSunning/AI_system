#!/usr/bin/env bash
# NSYS around the same non-server LLM.generate workload used by NCU.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage:
  bash profile_offline_nsys.sh --model MODEL [options]
  bash profile_offline_nsys.sh MODEL MODE [SCOPE] [options]

Required:
  --model MODEL          bf16 | awq | fp8-official | fp8-local | all
Collection:
  --mode MODE            eager | graph | all (default: graph)
  --scope SCOPE          steady | all (default: steady)
                        steady: after warmup; all: initialization through completion
Workload / output:
  --case ID              Frozen case (default: PROFILE_CASE or w128_o128)
  --count N              Measured generate calls (default: PROFILE_COUNT or 1)
  --batch-size N         Requests per generate (default: PROFILE_BATCH_SIZE or 1)
  --batch-tokens N       Token budget (default: PROFILE_BATCH_TOKENS or 2048)
  --warmup N             Warmup generate calls, at least 5 (default: 5)
  --requests-file PATH   requests.jsonl inside a validated snapshot
  --output-dir PATH      Reports (default: NSYS_OUT or log/nsys_offline)
  -h, --help             Print this help; no GPU or model files required
Environment fallback:
  MODEL_ROOT, REQUESTS, NSYS_OUT, PROFILE_CASE, PROFILE_COUNT,
  PROFILE_BATCH_SIZE, PROFILE_BATCH_TOKENS, QUANT_PROFILE_LOG_EVERY_N_STEPS,
  NSYS_BIN, PYTHON_BIN. Explicit command-line options take precedence.
HELP
}
die() { echo "Error: $*" >&2; exit 2; }
value_required() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
positive() { [[ "$2" =~ ^[0-9]+$ ]] && (( 10#$2 > 0 )) || die "$1 must be a positive integer"; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
demo_dir="$(cd -- "$script_dir/.." && pwd)"
selected_model='' selected_mode=graph capture_scope=steady named_identity=0
requests="${REQUESTS:-$demo_dir/data/frozen/requests.jsonl}"
output_root="${NSYS_OUT:-$demo_dir/log/nsys_offline}"
nsys_bin="${NSYS_BIN:-nsys}" python_bin="${PYTHON_BIN:-python}"
case_id="${PROFILE_CASE:-w128_o128}" count="${PROFILE_COUNT:-1}"
batch_size="${PROFILE_BATCH_SIZE:-1}" batch_tokens="${PROFILE_BATCH_TOKENS:-2048}" warmup=5
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --model|--mode|--scope)
      value_required "$@"; named_identity=1
      case "$1" in --model) selected_model=$2;; --mode) selected_mode=$2;; --scope) capture_scope=$2;; esac
      shift 2;;
    --case) value_required "$@"; case_id=$2; shift 2;;
    --count) value_required "$@"; count=$2; shift 2;;
    --batch-size) value_required "$@"; batch_size=$2; shift 2;;
    --batch-tokens) value_required "$@"; batch_tokens=$2; shift 2;;
    --warmup) value_required "$@"; warmup=$2; shift 2;;
    --requests-file) value_required "$@"; requests=$2; shift 2;;
    --output-dir) value_required "$@"; output_root=$2; shift 2;;
    --*) die "unknown option: $1";;
    *) positional+=("$1"); shift;;
  esac
done
if [[ ${#positional[@]} -gt 0 ]]; then
  [[ ${#positional[@]} -ge 2 && ${#positional[@]} -le 3 && "$named_identity" == 0 ]] ||
    die "use either MODEL MODE [SCOPE] or named model/mode/scope options"
  selected_model=${positional[0]} selected_mode=${positional[1]}
  if [[ ${#positional[@]} == 3 ]]; then capture_scope=${positional[2]}; fi
fi
case "$selected_model" in
  bf16|awq|fp8-official|fp8-local) models=("$selected_model");;
  all) models=(bf16 awq fp8-official fp8-local);;
  *) die "--model must select bf16, awq, fp8-official, fp8-local or all";;
esac
case "$selected_mode" in
  eager|graph) modes=("$selected_mode");;
  all) modes=(eager graph);;
  *) die "--mode must be eager, graph or all";;
esac
case "$capture_scope" in steady|all) ;; *) die "--scope must be steady or all";; esac
[[ "$case_id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid case ID"
positive --count "$count"; positive --batch-size "$batch_size"
positive --batch-tokens "$batch_tokens"; positive --warmup "$warmup"
(( 10#$warmup >= 5 )) || die "--warmup must be at least 5"
[[ -f "$requests" ]] || die "frozen requests missing: $requests"
mkdir -p "$output_root"
unset VLLM_TORCH_CUDA_PROFILE VLLM_TORCH_PROFILER_DIR
unset VLLM_PROFILER_DELAY_ITERS VLLM_PROFILER_MAX_ITERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
export QUANT_PROFILE_LOG_EVERY_N_STEPS="${QUANT_PROFILE_LOG_EVERY_N_STEPS:-32}"

for mode in "${modes[@]}"; do
  for model in "${models[@]}"; do
    tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    prefix="$output_root/offline_${model}_${mode}_${capture_scope}_${case_id}_${tag}"
    graph_args=()
    audit_args=(--require-worker-ranges --measure-range offline/measure --fail-on-incomplete-cuda)
    if [[ "$mode" == graph ]]; then graph_args=(--graph); audit_args+=(--expect-graph); fi
    capture_args=()
    profiler=none
    if [[ "$capture_scope" == steady ]]; then
      capture_args=(--capture-range=cudaProfilerApi --capture-range-end=stop)
      audit_args+=(--skip-load-model-range)
      profiler=cuda
    fi
    "$nsys_bin" --version > "${prefix}.version.txt"
    printf 'model=%s\nmode=%s\nscope=%s\ncase=%s\ncount=%s\nbatch_size=%s\nbatch_tokens=%s\nwarmup=%s\nrequests=%s\n' \
      "$model" "$mode" "$capture_scope" "$case_id" "$count" "$batch_size" \
      "$batch_tokens" "$warmup" "$requests" > "${prefix}.settings.txt"

    "$nsys_bin" profile \
      --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      --cuda-event-trace=false --cuda-graph-trace=node \
      --cuda-flush-interval=1000 --trace-fork-before-exec=false \
      --wait=all --kill=none "${capture_args[@]}" --output="$prefix" \
      "$python_bin" "$script_dir/profile_kernels.py" \
      --profile "$model" --requests-file "$requests" --case "$case_id" \
      --count "$count" --batch-size "$batch_size" --batch-tokens "$batch_tokens" --warmup "$warmup" \
      --profiler "$profiler" --delay-iters 0 --max-iters 0 "${graph_args[@]}" \
      --nvtx --event-dir "${prefix}.events" --output "${prefix}.workload.json" \
      2>&1 | tee "${prefix}.log"

    "$nsys_bin" export --type=sqlite --output="${prefix}.sqlite" "${prefix}.nsys-rep"
    "$python_bin" "$script_dir/check_nsys_capture.py" "${prefix}.sqlite" \
      "${audit_args[@]}" | tee "${prefix}.coverage.json"
    printf 'Report: %s\n' "${prefix}.nsys-rep"
  done
done
