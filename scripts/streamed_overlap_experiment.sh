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

Evaluate an already completed suite:
  --mode evaluate --config <suite-input-record.json> --run-dir <suite-directory> \
    --run-id <new-evaluation-id> --config-sha256 <suite-config-hash> \
    --mps-memory-fraction <limit> --min-suite-free-bytes <bytes> \
    --min-log-free-bytes <bytes> --acknowledge-evaluation-resources

Train or resume only after separate launch approval and resource review:
  --mode train|resume --config <suite-input-record.json> --run-dir <suite-directory> \
    --run-id <new-attempt-id> --config-sha256 <suite-config-hash> \
    --epochs <approved-epochs> --batches-per-epoch <approved-batches> \
    --mps-memory-fraction <limit> --min-suite-free-bytes <bytes> \
    --min-log-free-bytes <bytes> --acknowledge-training-approval \
    --acknowledge-training-resources [--tensorboard-run-id <original-training-id>]

TensorBoard binds to 127.0.0.1:6006 and remains running after this shell exits.
Only a verified run-owned PID can be reused or stopped; occupied ports are errors.
No training or event generation occurs in these monitoring modes.

Evaluation requires an existing completed suite and an explicit resource review.
Train/resume are blocked pending integrated real-data validation, resource review
and separate launch approval. Acknowledgements do not prove these.
Use --print-commands to inspect their proposed commands without launching.
The MPS fraction is per process, not a combined memory limit.
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
config_sha256=""
min_suite_free_bytes=""
min_log_free_bytes=""
evaluation_resources_reviewed=0
epochs=""
batches_per_epoch=""
tensorboard_run_id=""
training_approved=0
training_resources_reviewed=0
output_root="/Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap"
print_only=0
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --mode|--config|--run-id|--output-root|--run-dir|--mps-memory-fraction|--max-wall-seconds|--config-sha256|--min-suite-free-bytes|--min-log-free-bytes|--epochs|--batches-per-epoch|--tensorboard-run-id)
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
        --config-sha256) config_sha256="$2" ;;
        --min-suite-free-bytes) min_suite_free_bytes="$2" ;;
        --min-log-free-bytes) min_log_free_bytes="$2" ;;
        --epochs) epochs="$2" ;;
        --batches-per-epoch) batches_per_epoch="$2" ;;
        --tensorboard-run-id) tensorboard_run_id="$2" ;;
      esac
      shift 2 ;;
    --print-commands) print_only=1; shift ;;
    --acknowledge-evaluation-resources) evaluation_resources_reviewed=1; shift ;;
    --acknowledge-training-approval) training_approved=1; shift ;;
    --acknowledge-training-resources) training_resources_reviewed=1; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$mode" in
  train|resume)
    if ((!print_only)); then
      printf 'TRAINING_NOT_READY: integrated real-data validation, resource review and separate launch approval are outstanding; LOCAL_TARGET_VALIDATION_PENDING. No training authorized.\n' >&2
      exit 2
    fi ;;
  audit|smoke|prepare|evaluate|tensorboard|tensorboard-status|tensorboard-stop) ;;
  *) printf 'Unknown or missing --mode. Use --help.\n' >&2; exit 2 ;;
esac
if [[ "$mode" != prepare && "$mode" != evaluate && "$mode" != train && "$mode" != resume && -n "$mps_memory_fraction" ]] \
    || [[ "$mode" != prepare && -n "$max_wall_seconds" ]]; then
  printf 'MPS and wall-time limits are only supported by their respective modes.\n' >&2
  exit 2
fi
if [[ "$mode" != evaluate && "$mode" != train && "$mode" != resume \
    && ( -n "$config_sha256" || -n "$min_suite_free_bytes" || -n "$min_log_free_bytes" ) ]] \
    || [[ "$mode" != evaluate && "$evaluation_resources_reviewed" == 1 ]]; then
  printf 'Suite resource options require an applicable suite mode.\n' >&2
  exit 2
fi
if [[ "$mode" != train && "$mode" != resume \
    && ( -n "$epochs" || -n "$batches_per_epoch" || -n "$tensorboard_run_id" \
      || "$training_approved" == 1 || "$training_resources_reviewed" == 1 ) ]]; then
  printf 'Training options require --mode train or resume.\n' >&2
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
elif [[ "$mode" == evaluate ]]; then
  if [[ -z "$config" || -z "$run_dir" || ! "$config_sha256" =~ ^[0-9a-f]{64}$ \
      || -z "$mps_memory_fraction" || ! "$min_suite_free_bytes" =~ ^[1-9][0-9]*$ \
      || ! "$min_log_free_bytes" =~ ^[1-9][0-9]*$ || "$evaluation_resources_reviewed" != 1 ]]; then
    printf 'Evaluate requires a suite record, suite directory, config hash, MPS limit, positive free-space floors and --acknowledge-evaluation-resources.\n' >&2
    exit 2
  fi
  command=("$python" -u -m scripts.streamed_overlap_runner --action evaluate-attempt --log-dir "$log_dir"
    --run-id "$run_id" --config "$config" --run-dir "$run_dir" --config-sha256 "$config_sha256"
    --mps-memory-fraction "$mps_memory_fraction" --min-suite-free-bytes "$min_suite_free_bytes"
    --min-log-free-bytes "$min_log_free_bytes")
elif [[ "$mode" == train || "$mode" == resume ]]; then
  if [[ -z "$config" || -z "$run_dir" || ! "$config_sha256" =~ ^[0-9a-f]{64}$ \
      || ! "$epochs" =~ ^[1-9][0-9]*$ || ! "$batches_per_epoch" =~ ^[1-9][0-9]*$ \
      || -z "$mps_memory_fraction" || ! "$min_suite_free_bytes" =~ ^[1-9][0-9]*$ \
      || ! "$min_log_free_bytes" =~ ^[1-9][0-9]*$ \
      || ( "$print_only" != 1 && ( "$training_approved" != 1 || "$training_resources_reviewed" != 1 ) ) ]]; then
    printf 'TRAINING_NOT_READY: pinned inputs, explicit budgets, resource review and separate approval are required.\n' >&2
    exit 2
  fi
  if [[ "$mode" == train ]]; then
    if [[ -n "$tensorboard_run_id" ]]; then
      printf 'Fresh training owns TensorBoard under its own run ID.\n' >&2; exit 2
    fi
    tensorboard_run_id="$run_id"
  elif [[ ! "$tensorboard_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}$ ]]; then
    printf 'Resume requires the original training --tensorboard-run-id.\n' >&2; exit 2
  fi
  command=("$python" -u -m scripts.streamed_overlap_runner --action "${mode}-attempt" --log-dir "$log_dir"
    --run-id "$run_id" --config "$config" --run-dir "$run_dir" --config-sha256 "$config_sha256"
    --epochs "$epochs" --batches-per-epoch "$batches_per_epoch" --mps-memory-fraction "$mps_memory_fraction"
    --min-suite-free-bytes "$min_suite_free_bytes" --min-log-free-bytes "$min_log_free_bytes"
    --tensorboard-log-dir "logs/$tensorboard_run_id" --acknowledge-training-approval
    --acknowledge-training-resources)
else
  if [[ -z "$run_dir" ]]; then
    printf 'TensorBoard actions require --run-dir.\n' >&2; exit 2
  fi
  command+=(--run-dir "$run_dir")
fi
if ((print_only)); then
  if [[ "$mode" == train || "$mode" == resume ]]; then
    printf 'PREVIEW ONLY: do not run these commands before validation, resource review and launch approval.\n'
  fi
  if [[ "$mode" == audit || "$mode" == smoke || "$mode" == prepare ]]; then printf '%s output: %s\n' "$mode" "$output"; fi
  printf 'Combined log: %s/training.log\n' "$log_dir"
  printf '%q ' "${command[@]}"; printf '\n'
  printf 'Watch output: tail -n 100 -F %q\n' "$log_dir/training.log"
  if [[ "$mode" == prepare ]]; then
    printf 'After successful log capture: '
    printf '%q ' "$python" -u -m scripts.streamed_overlap_runner --action confirm-prepare-log --log-dir "$log_dir"
    printf '\n'
  elif [[ "$mode" == evaluate ]]; then
    printf 'After successful log capture: '
    printf '%q ' "$python" -u -m scripts.streamed_overlap_runner --action confirm-evaluation-log --log-dir "$log_dir"
    printf '\n'
  elif [[ "$mode" == train || "$mode" == resume ]]; then
    printf 'After successful log capture: '
    printf '%q ' "$python" -u -m scripts.streamed_overlap_runner --action confirm-training-log --log-dir "$log_dir"
    printf '\n'
  fi
  exit 0
fi
if [[ "$mode" == audit || "$mode" == smoke || "$mode" == prepare || "$mode" == evaluate || "$mode" == train || "$mode" == resume ]]; then
  if [[ -e "$log_dir" || -L "$log_dir" ]] \
      || [[ ( "$mode" == audit || "$mode" == smoke || "$mode" == prepare ) && ( -e "$output" || -L "$output" ) ]]; then
    printf 'Refusing existing output or log directory: %s, %s\n' "$output" "$log_dir" >&2
    exit 1
  fi
else
  if [[ ! -d "$log_dir" || -L "$log_dir" || ! -f "$log_dir/run_manifest.json" ]]; then
    printf 'Monitoring requires an existing run log directory and run manifest: %s\n' "$log_dir" >&2
    exit 1
  fi
fi
if [[ "$mode" == audit || "$mode" == smoke || "$mode" == prepare || "$mode" == evaluate || "$mode" == train || "$mode" == resume ]]; then
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
  if [[ "$mode" == evaluate ]]; then
    printf 'Caller acknowledged evaluation resource review; aggregate memory headroom is not verified by this runner.\n'
  elif [[ "$mode" == train || "$mode" == resume ]]; then
    printf 'Caller acknowledged training approval and resource review; aggregate memory headroom is not verified by this runner.\n'
  fi
  if [[ ! -x "$python" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$python" >&2; exit 1
  fi
  if [[ ( "$mode" == audit || "$mode" == smoke || "$mode" == prepare || "$mode" == evaluate || "$mode" == train || "$mode" == resume ) && ! -f "$config" ]]; then
    printf 'Missing %s config: %s\n' "$mode" "$config" >&2; exit 2
  fi
  "$python" --version
  "${command[@]}"
) 2>&1 | tee -a "$log_dir/training.log"
if [[ "$mode" == prepare ]]; then
  "$python" -u -m scripts.streamed_overlap_runner --action confirm-prepare-log --log-dir "$log_dir"
elif [[ "$mode" == evaluate ]]; then
  "$python" -u -m scripts.streamed_overlap_runner --action confirm-evaluation-log --log-dir "$log_dir"
  printf 'Evaluation log confirmed; reports are not an adoption decision.\n'
elif [[ "$mode" == train || "$mode" == resume ]]; then
  "$python" -u -m scripts.streamed_overlap_runner --action confirm-training-log --log-dir "$log_dir"
  printf 'Suite log confirmed; reports are not an adoption decision.\n'
fi
