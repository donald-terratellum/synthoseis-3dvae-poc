#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/remove_verified_local_duplicates.sh"

move_suggested_main() {
  local execute=0 entry source destination
  case "${1:-}" in
    '') ;;
    --execute) execute=1 ;;
    --help)
      printf 'Usage: bash %s [--execute]\nDefault: print plans only. --execute copies with rsync -aE, SHA-256 verifies every file, removes the local tree, and leaves a symlink.\nThis is a verified cross-drive move, not mv followed by redundant deletion. All jobs/readers must be stopped. Existing destinations are refused; incomplete copies are retained for investigation. macOS rsync -E preserves extended attributes.\n' "$0"
      return ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; return 2 ;;
  esac
  local suggestions=(
    '/Users/donaldpg/synthoseis-3dvae-poc/data|/Volumes/CrucialX9/3dvae-poc_data'
    '/Users/donaldpg/synthoseis-3dvae-poc/checkpoints|/Volumes/CrucialX9/3dvae-poc_checkpoints/legacy_local_checkpoints'
    '/Users/donaldpg/synthoseis-3dvae-poc/checkpoints_deep_supervise|/Volumes/CrucialX9/3dvae-poc_checkpoints/legacy_deep_supervise'
  )
  printf 'Suggested archive moves: data (~155.7 GB), checkpoints (~6.8 GB), deep supervision (~0.75 GB).\n'
  printf 'Excluded: .venv (rebuild rather than relocate), active external P7 outputs, and unique raw synthoseis volumes.\n'
  for entry in "${suggestions[@]}"; do
    source="${entry%%|*}"; destination="${entry#*|}"
    if [[ -L "$source" || ! -d "$source" ]]; then printf 'SKIP absent or already linked: %s\n' "$source"; continue; fi
    printf 'PLAN: copy %s -> %s; verify all files; remove local tree; link original path\n' "$source" "$destination"
    if [[ "$execute" == 1 ]]; then
      storage_check_idle
      storage_check_paths "$source" "$destination"
      if [[ -e "$destination" || -L "$destination" ]]; then printf 'Refusing existing destination: %s\n' "$destination" >&2; return 1; fi
      "$storage_python" - "$source" <<'PY'
import os, shutil, sys
from pathlib import Path
size = sum(path.stat().st_size for current, directories, names in os.walk(sys.argv[1]) for name in names
           if (path := Path(current)/name).is_file() and not path.is_symlink())
if shutil.disk_usage('/Volumes/CrucialX9').free < size + 15*1024**3:
    raise SystemExit('Insufficient external free space plus 15 GiB reserve')
print(f'External capacity check passed: estimated logical payload {size/1e9:.3f} GB', flush=True)
PY
      mkdir -p -- "$destination"
      rsync -aE -- "$source/" "$destination/"
      storage_verify_tree "$source" "$destination" 0
      storage_remove_and_link "$source" "$destination"
    fi
  done
  if [[ "$execute" == 0 ]]; then printf 'DRY RUN ONLY: nothing changed. Review paths and rerun with --execute to apply.\n'; fi
}

move_suggested_main "$@"