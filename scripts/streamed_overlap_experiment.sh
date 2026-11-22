#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'HELP'
Usage: bash scripts/streamed_overlap_experiment.sh --mode audit \
  --config scripts/streamed_overlap_audit.example.json --run-id <new-id> \
  [--output-root /Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap] \
  [--print-commands]

Monitoring an existing suite:
  --mode tensorboard --run-id <id> --run-dir <existing-suite-directory>
  --mode tensorboard-status --run-id <id> --run-dir <existing-suite-directory>
  --mode tensorboard-stop --run-id <id> --run-dir <existing-suite-directory>

No-optimizer MPS smoke:
  --mode smoke --config <smoke-config.json> --run-id <new-id>

Prepare a suite-input record without training:
  --mode prepare --config <suite-settings.json> --run-id <new-id> \
    --mps-memory-fraction <limit> --max-wall-seconds <limit>

TensorBoard binds to 127.0.0.1:6006 and remains running after this shell exits.
Only a verified run-owned PID can be reused or stopped; occupied ports are errors.
No training or event generation occurs in these monitoring modes.

This runner implements audit, no-optimizer MPS smoke, preparation and monitoring, not training.
Train/resume/evaluate modes remain blocked until runner integration,
evaluation prerequisites and the separate training approval are complete.
Audit output is LOCAL_TARGET_VALIDATION_PENDING.
Combined stdout/stderr is captured at logs/<run-id>/training.log.
PYTHON may select the interpreter; default is this repository's uv-managed .venv/bin/python.
Print-only mode creates no files and starts no processes.
HELP
}

mode=""
config=""
run_id=""
run_dir=""
mps_memory_fraction=""
max_wall_seconds=""
output_root="/Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap"
print_only=0
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --mode|--config|--run-id|--output-root|--run-dir|--mps-memory-fraction|--max-wall-seconds)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        printf 'Missing value for %s\n' "$1" >&2; exit 2
      fi
      case "$1" in
        --mode) mode="$2" ;;
        --config) config="$2" ;;
        --run-id) run_id="$2" ;;
        --output-root) output_root="$2" ;;
        --run-dir) run_dir="$2" ;;
        --mps-memory-fraction) mps_memory_fraction="$2" ;;
        --max-wall-seconds) max_wall_seconds="$2" ;;
      esac
      shift 2 ;;
    --print-commands) print_only=1; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$mode" in
  audit|smoke|prepare|tensorboard|tensorboard-status|tensorboard-stop) ;;
  train|resume|evaluate)
    printf 'TRAINING_NOT_READY: integrated runner and evaluation prerequisites remain incomplete; audit status is LOCAL_TARGET_VALIDATION_PENDING. No training authorized.\n' >&2
    exit 2 ;;
  *) printf 'Unknown or missing --mode. Use --help.\n' >&2; exit 2 ;;
esac
if [[ "$mode" != prepare && ( -n "$mps_memory_fraction" || -n "$max_wall_seconds" ) ]]; then
  printf 'Preparation limits are only supported with --mode prepare.\n' >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}$ ]]; then
  printf 'Provide a new --run-id using 1-101 letters, digits, dots, underscores or hyphens.\n' >&2
  exit 2
fi
python="${PYTHON:-.venv/bin/python}"
output="${output_root}/${run_id}"
log_dir="logs/${run_id}"
command=("$python" -u -m scripts.streamed_overlap_runner --action "$mode" --log-dir "$log_dir")
if [[ "$mode" == audit || "$mode" == smoke || "$mode" == prepare ]]; then
  if [[ -z "$config" ]]; then
    printf 'Audit, smoke and prepare require --config.\n' >&2; exit 2
  fi
  command+=(--run-id "$run_id" --config "$config" --output "$output")
  if [[ "$mode" == prepare ]]; then
    if [[ -z "$mps_memory_fraction" || -z "$max_wall_seconds" ]]; then
      printf 'Prepare requires --mps-memory-fraction and --max-wall-seconds.\n' >&2; exit 2
    fi
    command+=(--mps-memory-fraction "$mps_memory_fraction" --max-wall-seconds "$max_wall_seconds")
  fi
else
  if [[ -z "$run_dir" ]]; then
    printf 'TensorBoard actions require --run-dir.\n' >&2; exit 2
  fi
  command+=(--run-dir "$run_dir")
fi
if ((print_only)); then
  if [[ "$mode" == audit || "$mode" == smoke || "$mode" == prepare ]]; then printf '%s output: %s\n' "$mode" "$output"; fi
  printf 'Combined log: %s/training.log\n' "$log_dir"
  printf '%q ' "${command[@]}"; printf '\n'
  printf 'Watch output: tail -n 100 -F %q\n' "$log_dir/training.log"
  if [[ "$mode" == prepare ]]; then
    printf 'After successful log capture: '
    printf '%q ' "$python" -u -m scripts.streamed_overlap_runner --action confirm-prepare-log --log-dir "$log_dir"
    printf '\n'
  fi
  exit 0
fi
if [[ "$mode" == audit || "$mode" == smoke || "$mode" == prepare ]]; then
  if [[ -e "$output" || -L "$output" || -e "$log_dir" || -L "$log_dir" ]]; then
    printf 'Refusing existing output or log directory: %s, %s\n' "$output" "$log_dir" >&2
    exit 1
  fi
else
  if [[ ! -d "$log_dir" || -L "$log_dir" || ! -f "$log_dir/run_manifest.json" ]]; then
    printf 'Monitoring requires an existing run log directory and run manifest: %s\n' "$log_dir" >&2
    exit 1
  fi
fi
if [[ "$mode" == audit || "$mode" == smoke || "$mode" == prepare ]]; then
  mkdir -p logs
  mkdir "$log_dir"
fi
(
  failure_label="$mode"
  if [[ "$mode" == audit ]]; then failure_label="AUDIT"; fi
  trap 'status=$?; if ((status)); then printf "%s FAILED: exit %s\n" "$failure_label" "$status" >&2; fi' EXIT
  printf 'Streamed experiment runner; mode=%s; run=%s; UTC=%s\n' "$mode" "$run_id" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'Git revision: '; git rev-parse HEAD
  printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
  printf 'Watch progress: tail -n 100 -F %q\n' "$log_dir/training.log"
  if [[ ! -x "$python" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$python" >&2; exit 1
  fi
  if [[ ( "$mode" == audit || "$mode" == smoke || "$mode" == prepare ) && ! -f "$config" ]]; then
    printf 'Missing %s config: %s\n' "$mode" "$config" >&2; exit 2
  fi
  "$python" --version
  "${command[@]}"
) 2>&1 | tee -a "$log_dir/training.log"
if [[ "$mode" == prepare ]]; then
  "$python" -u -m scripts.streamed_overlap_runner --action confirm-prepare-log --log-dir "$log_dir"
fi
