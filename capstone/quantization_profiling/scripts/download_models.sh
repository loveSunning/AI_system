#!/usr/bin/env bash
set -Eeuo pipefail

# Follow the reference script's ModelScope CLI and repo-name directory layout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
MODELS=()
REVISION=""
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage: bash download_models.sh [options]

Download complete model repositories with the ModelScope CLI.
Default model: Qwen/Qwen3-8B (only).
Default directory: <repository-root>/models/<repo-name>

Options:
  --model OWNER/NAME  Select a model; repeat for multiple models.
                      Supplying --model replaces the default model list.
  --revision VALUE    ModelScope revision for all selected models (optional).
                      Without it, ModelScope uses the repository default.
  --model-root PATH   Download root; overrides the MODEL_ROOT environment variable.
  --dry-run           Print the commands without creating directories or downloading.
  -h, --help          Show this help.

Examples:
  bash download_models.sh --dry-run
  bash download_models.sh --model-root /data/models
  bash download_models.sh --model Qwen/Qwen3-8B --revision <modelscope-revision>
  MODEL_ROOT=/data/models bash download_models.sh --model Qwen/Qwen3-8B

Use a revision available on ModelScope; Hugging Face revisions need not match.
A source marker prevents mixing owners or revisions in the same directory.
An existing nonempty directory without a marker is refused; choose a new root.
This script does not install dependencies or check GPU capacity.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value."
}

while (( $# > 0 )); do
  case "$1" in
    --model)
      require_value "$@"
      MODELS+=("$2")
      shift 2
      ;;
    --revision)
      require_value "$@"
      REVISION="$2"
      shift 2
      ;;
    --model-root)
      require_value "$@"
      MODEL_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) fail "Unknown argument: $1 (use --help)." ;;
  esac
done

if (( ${#MODELS[@]} == 0 )); then
  MODELS=("Qwen/Qwen3-8B")
fi
[[ -n "$MODEL_ROOT" ]] || fail 'The model root must not be empty.'
[[ "$REVISION" != *$'\n'* && "$REVISION" != *$'\r'* ]] || fail 'Revision must be a single line.'
REQUESTED_REVISION="${REVISION:-<default>}"
MARKER_NAME='.modelscope-download-source'

# Validate every target before starting any download.
for (( i = 0; i < ${#MODELS[@]}; i++ )); do
  model_id="${MODELS[i]}"
  [[ "$model_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "Invalid model ID: $model_id (expected OWNER/NAME)."
  model_name="${model_id##*/}"
  for (( j = 0; j < i; j++ )); do
    previous_name="${MODELS[j]##*/}"
    [[ "${model_name,,}" != "${previous_name,,}" ]] \
      || fail "Directory name collision: ${MODELS[j]} and $model_id. Use separate invocations and model roots."
  done

  local_dir="${MODEL_ROOT}/${model_name}"
  marker="${local_dir}/${MARKER_NAME}"
  [[ ! -L "$local_dir" ]] || fail "Refusing a symlink target: $local_dir"
  [[ ! -e "$local_dir" || -d "$local_dir" ]] || fail "Target is not a directory: $local_dir"
  if [[ -e "$marker" ]]; then
    [[ -f "$marker" && ! -L "$marker" ]] || fail "Invalid source marker: $marker"
    stored_model=''
    stored_revision=''
    { IFS= read -r stored_model && IFS= read -r stored_revision; } < "$marker" \
      || fail "Cannot read source marker: $marker"
    [[ "$stored_model" == "$model_id" && "$stored_revision" == "$REQUESTED_REVISION" ]] \
      || fail "Target belongs to $stored_model (revision $stored_revision): $local_dir. Use a separate model root."
  elif [[ -d "$local_dir" && -n "$(ls -A -- "$local_dir")" ]]; then
    fail "Existing nonempty directory has no source marker: $local_dir. Use a new model root."
  fi
done

if ! "$DRY_RUN" && ! command -v modelscope >/dev/null 2>&1; then
  printf 'ERROR: modelscope CLI not found in PATH.\n' >&2
  printf 'Provide an environment supporting: modelscope download --model ID --local_dir PATH [--revision VALUE].\n' >&2
  printf 'No dependencies were installed. Use --dry-run to preview without the CLI.\n' >&2
  exit 1
fi

for model_id in "${MODELS[@]}"; do
  model_name="${model_id##*/}"
  local_dir="${MODEL_ROOT}/${model_name}"
  command_args=(modelscope download --model "$model_id" --local_dir "$local_dir")
  if [[ -n "$REVISION" ]]; then
    command_args+=(--revision "$REVISION")
  fi
  printf '\n==> Model: %s\n    Revision: %s\n    Target: %s\n' "$model_id" "$REQUESTED_REVISION" "$local_dir"
  if "$DRY_RUN"; then
    printf '    Command: '
    printf '%q ' "${command_args[@]}"
    printf '\n'
    continue
  fi

  mkdir -p -- "$local_dir"
  printf '%s\n%s\n' "$model_id" "$REQUESTED_REVISION" > "${local_dir}/${MARKER_NAME}"
  if "${command_args[@]}"; then
    printf 'Downloaded: %s\n' "$model_id"
  else
    download_status=$?
    printf 'ERROR: download failed for %s (exit %s). Rerun with the same arguments to retry.\n' \
      "$model_id" "$download_status" >&2
    exit "$download_status"
  fi
done

if "$DRY_RUN"; then
  printf '\nDry run complete; no directories or model weights were created.\n'
else
  printf '\nAll requested models were downloaded under %s.\n' "$MODEL_ROOT"
fi
