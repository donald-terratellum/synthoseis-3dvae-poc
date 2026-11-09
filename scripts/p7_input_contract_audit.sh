#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"
log="${LOG_PATH:-logs/p7_input_contract_audit_20261005.log}"
reports="${REPORT_DIR:-/Volumes/CrucialX9/3dvae-poc_logs/p7_input_contract_audit_20261005}"
device="${DEVICE:-mps}"
resume="${RESUME:-0}"
checkpoints=(
  "p0b=checkpoints/geoaware_p0b_r2_seed20260925_r2/vae_best.pt"
  "e0=/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e0_e1_seed20260925/e0/stage1_reconstruction/vae_best.pt"
  "e1=/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e0_e1_seed20260925/e1/stage1_reconstruction/vae_best.pt"
)
if [[ ! -d /Volumes/CrucialX9 ]]; then printf 'CrucialX9 is not mounted\n' >&2; exit 1; fi
if [[ "$resume" != 1 && ( -e "$log" || -L "$log" || -e "$reports" ) ]]; then printf 'Refusing existing audit log/report directory; use RESUME=1 or new LOG_PATH and REPORT_DIR\n' >&2; exit 1; fi
mkdir -p "$(dirname "$log")" "$reports"
run_evaluation() {
  local destination="$1" count="$2" mode="$3" data="$4"
  if [[ "$resume" == 1 && -f "$destination" ]]; then
    "$python" - "$destination" "$count" "$mode" "$data" "$device" "${checkpoints[@]}" <<'PY'
import json, sys
from pathlib import Path
path, count, mode, data, device, *checkpoints = sys.argv[1:]
report = json.loads(Path(path).read_text())
settings = report['settings']
expected = {'samples': int(count), 'input_mode': mode, 'synthetic_data': data, 'device': device,
            'seed': 20261003, 'batch_size': 2, 'checkpoints': checkpoints, 'contract_audit': True,
            'real_data': 'data/real_test_32-32-64.zarr', 'evaluation_only': True}
if any(settings.get(key) != value for key, value in expected.items()):
    raise SystemExit(f'Refusing reuse of incompatible report: {path}')
if set(report['arms']) != {'p0b', 'e0', 'e1'} or not all(values['adapter_parity']['passed'] for values in report['arms'].values()):
    raise SystemExit(f'Refusing incomplete/parity-failed report: {path}')
print(f'RESUME: reusing completed matching report {path}', flush=True)
PY
    return
  fi
  "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --contract_audit \
    --device "$device" --batch_size 2 --seed 20261003 --samples "$count" --input_mode "$mode" \
    --synthetic_data "$data" --checkpoints "${checkpoints[@]}" --out_json "$destination"
}
{
  printf 'P7 exact tokenizer contract audit started %s; no training or gate changes\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  for mode in clean extrema tokenizer; do
    for split in train validation; do
      data=data/synth_val_v2_uniform_32-32-64.zarr
      if [[ "$split" == train ]]; then data=data/synth_train_anchored_32-32-64.zarr; fi
      printf 'PROBE mode=%s split=%s samples=128\n' "$mode" "$split"
      run_evaluation "$reports/${split}_${mode}.json" 128 "$mode" "$data"
    done
  done
  printf 'FULL-STORE tokenizer evaluation: 5000 synthetic validation and 5000 real-test examples\n'
  run_evaluation "$reports/full_tokenizer.json" 5000 tokenizer data/synth_val_v2_uniform_32-32-64.zarr
  "$python" - "$reports" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
report = json.loads((root/'full_tokenizer.json').read_text())
if not report['reconstruction_gate']['full_dataset_coverage']:
    raise SystemExit('Final contract decision requires full-store coverage')
arms = report['arms']
deltas = {}
for reference in ('p0b', 'e0'):
    deltas[reference] = {
        'synthetic_recipe_pct': 100*(arms['e1']['evaluation']['synthetic']['sampled']['recipe_total']/arms[reference]['evaluation']['synthetic']['sampled']['recipe_total']-1),
        'real_mae_pct': 100*(arms['e1']['evaluation']['real']['sampled']['mae']/arms[reference]['evaluation']['real']['sampled']['mae']-1),
    }
eligible = all(value <= 2 for values in deltas.values() for value in values.values())
summary = {'e1_deltas_vs_reference': deltas, 'e1_contract_reconstruction_within_2pct': eligible,
           'original_clean_extrema_gate_changed': False, 'training_launched': False,
           'next_recommendation': 'Propose explicitly approved E1 geology pilot under a prospective tokenizer-contract protocol; no adoption'
           if eligible else 'Do not launch E1 geology; inspect input/target scaling and shared training before a capacity factorial',
           'scope': 'checkpoint response to deployment input; targets remain clean baked-scaled patches; no prediction rescaling'}
(root/'decision_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
print('CONTRACT DECISION:',json.dumps(summary,indent=2),flush=True)
PY
  printf 'COMPLETE: tokenizer input-contract audit; no weights, datasets, manifest or original gate modified.\n'
} 2>&1 | tee -a "$log"