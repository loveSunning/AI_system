#!/usr/bin/env bash
set -Eeuo pipefail

# Follow the reference script's ModelScope CLI and repo-name directory layout.
MODEL_ROOT="${MODEL_ROOT:-/workspace/weights_pub}"
MODELS=()
REVISION=""
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage: bash download_models.sh [options]

Download complete model repositories with the ModelScope CLI.
Default model: Qwen/Qwen3-8B (only).
Default directory: /workspace/weights_pub/<repo-name>

Options:
  --model OWNER/NAME  Select a model; repeat for multiple models.
                      Supplying --model replaces the default model list.
  --revision VALUE    ModelScope revision for all selected models (optional).
                      Without it, ModelScope uses the repository default.
  --model-root PATH   Download root; overrides the MODEL_ROOT environment variable.
  --dry-run           Show skipped models and pending commands without writing files.
  -h, --help          Show this help.

Examples:
  bash download_models.sh --dry-run
  bash download_models.sh --model-root /workspace/weights_pub
  bash download_models.sh --model Qwen/Qwen3-8B --model Qwen/Qwen3-8B-AWQ
  bash download_models.sh --model Qwen/Qwen3-8B --revision "<modelscope-revision>"
  MODEL_ROOT=/workspace/weights_pub bash download_models.sh --model Qwen/Qwen3-8B

Existing nonempty directories are skipped without modifying their contents.
Matching source markers without a completion marker are retried/resumed.
A completion marker is written only after a successful ModelScope command.
Existing external directories are not checked for complete model weights.
Use a separate root to download another source or revision into the same name.
Use a revision available on ModelScope; Hugging Face revisions need not match.
Downloads run sequentially; a failed download does not stop the remaining models.
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

marker_matches() {
  local marker="$1" expected_model="$2" expected_revision="$3"
  local stored_model='' stored_revision=''
  [[ -f "$marker" && ! -L "$marker" ]] || return 1
  { IFS= read -r stored_model && IFS= read -r stored_revision; } < "$marker" || return 1
  [[ "$stored_model" == "$expected_model" && "$stored_revision" == "$expected_revision" ]]
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
COMPLETE_MARKER_NAME='.modelscope-download-complete'
ACTIONS=()
REASONS=()
pending_count=0

# Classify all targets before writing; skipped directories are never modified.
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
  complete_marker="${local_dir}/${COMPLETE_MARKER_NAME}"
  [[ ! -e "$local_dir" || -d "$local_dir" ]] || fail "Target is not a directory: $local_dir"
  action=download
  reason=''
  if [[ -d "$local_dir" && -n "$(ls -A -- "$local_dir")" ]]; then
    action=skip
    reason='existing nonempty directory; completeness not verified'
    if marker_matches "$marker" "$model_id" "$REQUESTED_REVISION"; then
      if marker_matches "$complete_marker" "$model_id" "$REQUESTED_REVISION"; then
        reason='previous download completed'
      elif [[ ! -L "$local_dir" ]]; then
        action=resume
        reason='matching source without completion; retrying/resuming'
      fi
    elif [[ -e "$marker" || -L "$marker" ]]; then
      reason='existing source/revision differs or is unverified; use another --model-root for the requested version'
    fi
  fi

  if [[ "$action" != skip ]]; then
    [[ ! -L "$local_dir" ]] || fail "Refusing to write through a symlink target: $local_dir"
    for metadata_path in "$marker" "$complete_marker"; do
      [[ ! -L "$metadata_path" && ( ! -e "$metadata_path" || -f "$metadata_path" ) ]] \
        || fail "Invalid download marker: $metadata_path"
    done
    pending_count=$((pending_count + 1))
  fi
  ACTIONS+=("$action")
  REASONS+=("$reason")
done

# Reusing existing models requires neither ModelScope nor an internet connection.
if (( pending_count > 0 )) && ! "$DRY_RUN" && ! command -v modelscope >/dev/null 2>&1; then
  printf 'ERROR: modelscope CLI not found in PATH.\n' >&2
  printf 'Provide an environment supporting: modelscope download --model ID --local_dir PATH [--revision VALUE].\n' >&2
  printf 'No dependencies were installed. Use --dry-run to preview without the CLI.\n' >&2
  exit 1
fi

downloaded_count=0
skipped_count=0
failed_count=0
first_failure=0
for (( i = 0; i < ${#MODELS[@]}; i++ )); do
  model_id="${MODELS[i]}"
  model_name="${model_id##*/}"
  local_dir="${MODEL_ROOT}/${model_name}"
  printf '\n==> Model: %s\n    Revision: %s\n    Target: %s\n' "$model_id" "$REQUESTED_REVISION" "$local_dir"
  if [[ "${ACTIONS[i]}" == skip ]]; then
    printf '    SKIP: %s\n' "${REASONS[i]}"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  if [[ "${ACTIONS[i]}" == resume ]]; then
    printf '    RESUME: %s\n' "${REASONS[i]}"
  fi

  command_args=(modelscope download --model "$model_id" --local_dir "$local_dir")
  if [[ -n "$REVISION" ]]; then
    command_args+=(--revision "$REVISION")
  fi
  if "$DRY_RUN"; then
    printf '    Command: '
    printf '%q ' "${command_args[@]}"
    printf '\n'
    continue
  fi

  mkdir -p -- "$local_dir"
  printf '%s\n%s\n' "$model_id" "$REQUESTED_REVISION" > "${local_dir}/${MARKER_NAME}"
  if "${command_args[@]}"; then
    printf '%s\n%s\n' "$model_id" "$REQUESTED_REVISION" > "${local_dir}/${COMPLETE_MARKER_NAME}"
    printf 'Downloaded: %s\n' "$model_id"
    downloaded_count=$((downloaded_count + 1))
  else
    download_status=$?
    failed_count=$((failed_count + 1))
    if (( first_failure == 0 )); then
      first_failure="$download_status"
    fi
    printf 'ERROR: download failed for %s (exit %s). Continuing with remaining models; rerun the same arguments to retry.\n' \
      "$model_id" "$download_status" >&2
  fi
done

if "$DRY_RUN"; then
  printf '\nDry run complete: %s to download/resume, %s skipped; no files were created.\n' \
    "$pending_count" "$skipped_count"
else
  printf '\nSummary: downloaded=%s skipped=%s failed=%s; root=%s\n' \
    "$downloaded_count" "$skipped_count" "$failed_count" "$MODEL_ROOT"
fi
exit "$first_failure"
