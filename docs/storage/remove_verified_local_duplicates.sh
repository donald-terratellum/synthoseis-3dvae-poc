#!/usr/bin/env bash
set -euo pipefail

storage_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
storage_python="${PYTHON:-$storage_repo_root/.venv/bin/python}"

storage_check_idle() {
  local running
  running=$(ps -axo pid=,command= | awk '/[p]ython/ && /train|sample_patches|add_dip_samples|evaluate|synthoseis\/.*main/ {print}')
  if [[ -n "$running" ]]; then
    printf 'Refusing storage changes while training/data jobs are active:\n%s\n' "$running" >&2
    return 1
  fi
}

storage_check_paths() {
  "$storage_python" - "$1" "$2" <<'PY'
import os, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
if not os.path.ismount('/Volumes/CrucialX9'):
    raise SystemExit('CrucialX9 must be mounted, not merely an existing directory')
if source.is_symlink() or not source.is_dir():
    raise SystemExit(f'Source must be a real directory, not a symlink: {source}')
if not source.resolve().is_relative_to('/Users/donaldpg'):
    raise SystemExit('Source resolves outside /Users/donaldpg')
if len(source.resolve().relative_to('/Users/donaldpg').parts) < 2:
  raise SystemExit('Refusing the home directory or a top-level project directory')
if not destination.parent.resolve().is_relative_to('/Volumes/CrucialX9'):
    raise SystemExit('Destination parent resolves outside CrucialX9')
if destination.is_symlink():
    raise SystemExit('Destination may not be a symlink')
if source.resolve() in destination.resolve().parents or destination.resolve() in source.resolve().parents:
    raise SystemExit('Source and destination may not be nested')
PY
}

storage_verify_tree() {
  local source="$1" destination="$2" ignore_sidecars="${3:-0}"
  "$storage_python" - "$source" "$destination" "$ignore_sidecars" <<'PY'
import hashlib, os, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:3])
ignore_sidecars = sys.argv[3] == '1'
def manifest(root):
    records = {}
    for current, directories, names in os.walk(root, followlinks=False):
        if ignore_sidecars:
          directories[:] = [name for name in directories if not name.startswith('._')]
        for name in directories + names:
            path = Path(current)/name
            if path.is_symlink():
                raise SystemExit(f'Refusing a tree containing symlinks: {path}')
            if ignore_sidecars and name.startswith('._'):
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                records[relative] = ('directory',)
            elif path.is_file():
                before = path.stat()
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                        digest.update(chunk)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise SystemExit(f'File changed while hashing: {path}')
                records[relative] = ('file', after.st_size, digest.hexdigest())
            else:
                raise SystemExit(f'Refusing special filesystem entry: {path}')
    return records
if not destination.is_dir():
    raise SystemExit(f'External copy is missing: {destination}')
print(f'Hashing source and external copy before removal: {source}', flush=True)
left, right = manifest(source), manifest(destination)
if left != right:
    differing = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
    raise SystemExit(f'Content verification FAILED; no deletion. Examples: {differing[:5]}')
if not left:
    raise SystemExit('Refusing empty-directory verification')
print(f'VERIFIED {len(left)} entries: {source} == {destination}', flush=True)
PY
}

storage_remove_and_link() {
  local source="$1" destination="$2"
  storage_check_idle
  storage_check_paths "$source" "$destination"
  printf 'REMOVE verified local directory: %s\n' "$source"
  rm -rf -- "$source"
  ln -s -- "$destination" "$source"
  printf 'LINK preserved original path: %s -> %s\n' "$source" "$destination"
}

remove_verified_duplicates_main() {
  local execute=0 source destination entry
  case "${1:-}" in
    '') ;;
    --execute) execute=1 ;;
    --help)
      printf 'Usage: bash %s [--execute]\nDefault: print plans only. --execute re-hashes each pair, removes only verified stores, and leaves symlinks.\nNever run alongside readers/writers; external drive becomes required. macOS ._* sidecars are ignored only for these Zarr duplicates.\n' "$0"
      return ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; return 2 ;;
  esac
  local pairs=(
    '/Users/donaldpg/synthoseis/real_data/Netherlands1.real.zarr|/Volumes/CrucialX9/real_data/Netherlands1.real.zarr'
    '/Users/donaldpg/synthoseis/real_data/Netherlands2.real.zarr|/Volumes/CrucialX9/real_data/Netherlands2.real.zarr'
    '/Users/donaldpg/synthoseis/real_data/Netherlands3.real.zarr|/Volumes/CrucialX9/real_data/Netherlands3.real.zarr'
    '/Users/donaldpg/synthoseis/real_data/Netherlands4.real.zarr|/Volumes/CrucialX9/real_data/Netherlands4.real.zarr'
    '/Users/donaldpg/synthoseis/fake_data/test/CostaRica_part1.real.zarr|/Volumes/CrucialX9/fake_data/test/CostaRica_part1.real.zarr'
    '/Users/donaldpg/synthoseis/fake_data/test/Poseidon_part1.real.zarr|/Volumes/CrucialX9/fake_data/test/Poseidon_part1.real.zarr'
    '/Users/donaldpg/synthoseis/fake_data/test/Poseidon_part2.real.zarr|/Volumes/CrucialX9/fake_data/test/Poseidon_part2.real.zarr'
    '/Users/donaldpg/synthoseis/fake_data/test/cn1.real.zarr|/Volumes/CrucialX9/fake_data/test/cn1.real.zarr'
    '/Users/donaldpg/synthoseis/fake_data/validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr|/Volumes/CrucialX9/fake_data/validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr'
    '/Users/donaldpg/synthoseis/fake_data/validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr|/Volumes/CrucialX9/fake_data/validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr'
    '/Users/donaldpg/synthoseis/fake_data/validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr|/Volumes/CrucialX9/fake_data/validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr'
    '/Users/donaldpg/synthoseis/fake_data/validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr|/Volumes/CrucialX9/fake_data/validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr'
    '/Users/donaldpg/synthoseis/fake_data/validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr|/Volumes/CrucialX9/fake_data/validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr'
  )
  for entry in "${pairs[@]}"; do
    source="${entry%%|*}"; destination="${entry#*|}"
    if [[ -L "$source" || ! -d "$source" ]]; then printf 'SKIP absent or already linked: %s\n' "$source"; continue; fi
    printf 'PLAN: verify %s against %s; remove local store; link original path\n' "$source" "$destination"
    if [[ "$execute" == 1 ]]; then
      storage_check_idle
      storage_check_paths "$source" "$destination"
      storage_verify_tree "$source" "$destination" 1
      storage_remove_and_link "$source" "$destination"
    fi
  done
  if [[ "$execute" == 0 ]]; then printf 'DRY RUN ONLY: nothing changed. Review paths and rerun with --execute to apply.\n'; fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then remove_verified_duplicates_main "$@"; fi