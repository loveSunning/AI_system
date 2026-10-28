#!/usr/bin/env bash
# Non-server NCU: same LLM.generate workload as profile_offline_nsys.sh.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage:
  bash profile_offline_ncu.sh --model MODEL [options]
  bash profile_offline_ncu.sh MODEL PHASE TARGET LEVEL [options]

Required:
  --model MODEL             bf16 | awq | fp8-official | fp8-local
Selection (mutually exclusive; default: --target first4):
  --target TARGET           first4 | qkv | o | gate_up | down | quant | norm_quant
  --kernel-name NAME        Exact name, as interpreted by --kernel-name-base
  --kernel-regex REGEX      Partial regex match; do not prepend "regex:"
  --kernel-name-base BASE   demangled (default) | function | mangled
Collection:
  --phase PHASE            prefill | decode (default: decode)
  --level LEVEL            basic | detail (default: basic)
  --launch-skip N          Matching launches to skip (preset default, else 0)
  --launch-count N         Matching launches to collect (first4: 4, else 1)
  --delay-iters N          Worker gate delay (prefill: 0, decode: 8)
  --max-iters N            Worker gate length (prefill: 1, decode: 2; 0=unlimited)
  --mode MODE              graph | eager (default: NCU_GRAPH=1 => graph)
  --cache-control MODE     all | none (default: all)
  --clock-control MODE     none | base (default: none)
Workload / output:
  --case ID                Frozen case (default: PROFILE_CASE or w128_o128)
  --batch-size N           Requests per generate (default: PROFILE_BATCH_SIZE or 1)
  --batch-tokens N         Token budget (default: PROFILE_BATCH_TOKENS or 2048)
  --warmup N               Warmup generate calls, at least 5 (default: 5)
  --requests-file PATH     requests.jsonl inside a validated snapshot
  --output-dir PATH        Report directory (default: NCU_OUT or log/ncu_offline)
  -h, --help               Print this help; no GPU or model files required
Environment fallback:
  MODEL_ROOT, REQUESTS, NCU_OUT, NCU_GRAPH, NCU_CACHE_CONTROL,
  NCU_CLOCK_CONTROL, PROFILE_CASE, PROFILE_BATCH_SIZE, PROFILE_BATCH_TOKENS,
  NCU_BIN, PYTHON_BIN. Explicit command-line options take precedence.
Permissions:
  Non-root callers run NCU through sudo; root callers run it directly.
  The current Python environment and runtime environment are preserved.
HELP
}
die() { echo "Error: $*" >&2; exit 2; }
value_required() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
nonnegative() { [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a nonnegative integer"; }
positive() { nonnegative "$1" "$2"; (( 10#$2 > 0 )) || die "$1 must be positive"; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
demo_dir="$(cd -- "$script_dir/.." && pwd)"
model='' phase=decode level=basic target='' selector='' name_base=demangled
kernel_expression='' selector_count=0 named_identity=0
requests="${REQUESTS:-$demo_dir/data/frozen/requests.jsonl}"
output_root="${NCU_OUT:-$demo_dir/log/ncu_offline}"
ncu_bin="${NCU_BIN:-ncu}" python_bin="${PYTHON_BIN:-python}"
case "${NCU_GRAPH:-1}" in 1) engine_mode=graph;; 0) engine_mode=eager;; *) die "NCU_GRAPH must be 0 or 1";; esac
clock_mode="${NCU_CLOCK_CONTROL:-none}" cache_mode="${NCU_CACHE_CONTROL:-all}"
case_id="${PROFILE_CASE:-w128_o128}" batch_size="${PROFILE_BATCH_SIZE:-1}"
batch_tokens="${PROFILE_BATCH_TOKENS:-2048}" warmup=5
skip_override='' count_override='' delay_override='' max_override=''
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --model|--phase|--level)
      value_required "$@"; named_identity=1
      case "$1" in --model) model=$2;; --phase) phase=$2;; --level) level=$2;; esac
      shift 2;;
    --target|--kernel-name|--kernel-regex)
      value_required "$@"; selector_count=$((selector_count+1))
      case "$1" in
        --target) selector=preset; target=$2;;
        --kernel-name) selector=exact; kernel_expression=$2;;
        --kernel-regex) selector=regex; kernel_expression=$2;;
      esac
      shift 2;;
    --kernel-name-base) value_required "$@"; name_base=$2; shift 2;;
    --launch-skip) value_required "$@"; skip_override=$2; shift 2;;
    --launch-count) value_required "$@"; count_override=$2; shift 2;;
    --delay-iters) value_required "$@"; delay_override=$2; shift 2;;
    --max-iters) value_required "$@"; max_override=$2; shift 2;;
    --mode) value_required "$@"; engine_mode=$2; shift 2;;
    --case) value_required "$@"; case_id=$2; shift 2;;
    --batch-size) value_required "$@"; batch_size=$2; shift 2;;
    --batch-tokens) value_required "$@"; batch_tokens=$2; shift 2;;
    --warmup) value_required "$@"; warmup=$2; shift 2;;
    --requests-file) value_required "$@"; requests=$2; shift 2;;
    --output-dir) value_required "$@"; output_root=$2; shift 2;;
    --cache-control) value_required "$@"; cache_mode=$2; shift 2;;
    --clock-control) value_required "$@"; clock_mode=$2; shift 2;;
    --*) die "unknown option: $1";;
    *) positional+=("$1"); shift;;
  esac
done
if [[ ${#positional[@]} -gt 0 ]]; then
  [[ ${#positional[@]} -eq 4 && "$named_identity" == 0 && "$selector_count" == 0 ]] ||
    die "use either MODEL PHASE TARGET LEVEL or named model/phase/selector/level options"
  model=${positional[0]} phase=${positional[1]} target=${positional[2]} level=${positional[3]}
  selector=preset; selector_count=1
fi
[[ "$selector_count" -le 1 ]] || die "--target, --kernel-name and --kernel-regex are mutually exclusive"
if [[ -z "$selector" ]]; then selector=preset; target=first4; fi
case "$model" in bf16|awq|fp8-official|fp8-local) ;; *) die "--model must select bf16, awq, fp8-official or fp8-local";; esac
case "$phase" in decode|prefill) ;; *) die "--phase must be decode or prefill";; esac
case "$level" in basic|detail) ;; *) die "--level must be basic or detail";; esac
case "$engine_mode" in graph|eager) ;; *) die "--mode must be graph or eager";; esac
case "$name_base" in demangled|function|mangled) ;; *) die "invalid --kernel-name-base";; esac
case "$clock_mode" in none|base) ;; *) die "--clock-control must be none or base";; esac
case "$cache_mode" in all|none) ;; *) die "--cache-control must be all or none";; esac
[[ "$case_id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid case ID"
positive --batch-size "$batch_size"; positive --batch-tokens "$batch_tokens"
positive --warmup "$warmup"; (( 10#$warmup >= 5 )) || die "--warmup must be at least 5"

skip=0 count=1
if [[ "$selector" == preset ]]; then
  [[ "$name_base" == demangled ]] || die "presets require --kernel-name-base demangled"
  if [[ "$model" == fp8-* ]]; then
    kernel_expression='.*_w8a8_triton_block_scaled_mm.*'
  elif [[ "$model" == awq ]]; then
    kernel_expression='.*marlin::Marlin<.*'
  elif [[ "$phase" == decode ]]; then
    kernel_expression='.*internal::gemvx::kernel<.*'
  else
    kernel_expression='.*cutlass::Kernel2<cutlass_80_tensorop_bf16_.*'
  fi
  case "$target" in
    first4|qkv|o|gate_up|down)
      [[ "$case_id" == w128_o128 && "$batch_size" == 1 ]] ||
        die "projection presets require w128_o128 and batch 1; use --kernel-name or --kernel-regex for other workloads"
      case "$target" in first4) count=4;; qkv) skip=0;; o) skip=1;; gate_up) skip=2;; down) skip=3;; esac;;
    quant|norm_quant)
      [[ "$model" == fp8-* ]] || die "quant/norm_quant presets require an FP8 model"
      if [[ "$target" == quant ]]; then kernel_expression='.*per_token_group_quant_8bit.*'
      else kernel_expression='.*rms_norm_per_block_quant.*'; fi;;
    *) die "invalid --target preset: $target";;
  esac
  kernel_filter="regex:$kernel_expression"
  selection_label=$target
elif [[ "$selector" == regex ]]; then
  [[ "$kernel_expression" != regex:* ]] || die "omit the regex: prefix in --kernel-regex"
  kernel_filter="regex:$kernel_expression"; selection_label=regex
else
  [[ "$kernel_expression" != regex:* ]] || die "use --kernel-regex for regular expressions"
  kernel_filter=$kernel_expression; selection_label=kernel
fi
[[ -z "$skip_override" ]] || skip=$skip_override
[[ -z "$count_override" ]] || count=$count_override
if [[ "$phase" == prefill ]]; then delay=0; max_iters=1; else delay=8; max_iters=2; fi
[[ -z "$delay_override" ]] || delay=$delay_override
[[ -z "$max_override" ]] || max_iters=$max_override
nonnegative --launch-skip "$skip"; positive --launch-count "$count"
nonnegative --delay-iters "$delay"; nonnegative --max-iters "$max_iters"
[[ -f "$requests" ]] || die "frozen requests missing: $requests"
sections=(--section LaunchStats --section SpeedOfLight --section Occupancy)
if [[ "$level" == detail ]]; then
  sections+=(--section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis
             --section SchedulerStats --section WarpStateStats --section SourceCounters)
fi
graph_args=()
if [[ "$engine_mode" == graph ]]; then graph_args=(--graph); fi
# '/' inside the literal worker range name must be escaped for NCU NVTX syntax.
nvtx_filter="quantization_profiling@worker\\/$phase/"
# Resolve the active virtualenv executables before sudo changes PATH.
# Keep the Python entry point as-is: resolving its symlink can lose the venv.
ncu_path="$(command -v "$ncu_bin")" || die "NCU executable not found: $ncu_bin"
python_path="$(command -v "$python_bin")" || die "Python executable not found: $python_bin"
ncu_cmd=("$ncu_path")
if (( EUID != 0 )); then
  command -v sudo >/dev/null 2>&1 || die "sudo is required to access GPU performance counters"
  sudo -v
  # sudo can strip LD_LIBRARY_PATH even with -E; restore it explicitly.
  ncu_cmd=(sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "$ncu_path")
fi
mkdir -p "$output_root"
tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
prefix="$output_root/${model}_${phase}_${selection_label}_${level}_${engine_mode}_${tag}"

unset VLLM_TORCH_CUDA_PROFILE VLLM_TORCH_PROFILER_DIR
unset VLLM_PROFILER_DELAY_ITERS VLLM_PROFILER_MAX_ITERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
"${ncu_cmd[@]}" --version > "${prefix}.version.txt"
printf 'model=%s\nphase=%s\nlevel=%s\nselector=%s\ntarget=%s\nkernel_filter=%s\nkernel_name_base=%s\nmatching_skip=%s\nmatching_count=%s\ndelay_iters=%s\nmax_iters=%s\nmode=%s\ncache_control=%s\nclock_control=%s\ncase=%s\nbatch_size=%s\nbatch_tokens=%s\nwarmup=%s\nrequests=%s\nnvtx_filter=%s\n' \
  "$model" "$phase" "$level" "$selector" "$target" "$kernel_filter" "$name_base" \
  "$skip" "$count" "$delay" "$max_iters" "$engine_mode" "$cache_mode" "$clock_mode" \
  "$case_id" "$batch_size" "$batch_tokens" "$warmup" "$requests" "$nvtx_filter" > "${prefix}.settings.txt"

"${ncu_cmd[@]}" \
  --target-processes all --profile-from-start off \
  --replay-mode kernel --graph-profiling node \
  --nvtx --nvtx-include "$nvtx_filter" \
  --kernel-name-base "$name_base" --kernel-name "$kernel_filter" \
  --filter-mode global --launch-skip "$skip" --launch-count "$count" \
  --cache-control "$cache_mode" --clock-control "$clock_mode" \
  "${sections[@]}" --export "$prefix" \
  "$python_path" "$script_dir/profile_kernels.py" \
  --profile "$model" --requests-file "$requests" --case "$case_id" \
  --count 1 --batch-size "$batch_size" --warmup "$warmup" --batch-tokens "$batch_tokens" \
  --profiler cuda --delay-iters "$delay" --max-iters "$max_iters" "${graph_args[@]}" \
  --nvtx --event-dir "${prefix}.events" --output "${prefix}.workload.json" \
  2>&1 | tee "${prefix}.log"

if [[ ! -f "${prefix}.ncu-rep" ]]; then
  echo "No NCU report. Check kernel filter, worker phase, profiler window and ${prefix}.log." >&2
  exit 1
fi
"${ncu_cmd[@]}" --import "${prefix}.ncu-rep" --page raw --csv > "${prefix}.metrics.csv"
"${ncu_cmd[@]}" --import "${prefix}.ncu-rep" --page details > "${prefix}.details.txt"
printf 'Report: %s\n' "${prefix}.ncu-rep"
