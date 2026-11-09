# Folder and Zarr Inventory

Generated: 2026-10-05T22:07:59-05:00

## Save checkpoint (2026-11-09)

- Source, tests, runners, session notes, this inventory, and all seven locally ignored
  Markdown plans are included in the current-branch save commit. The November 8
  streamed-overlap plan remains a plan only; no experiment was started.
- Ignored benchmark reports/manifests and local logs were archived without moving or
  deleting originals:
  `/Volumes/CrucialX9/3dvae-poc_saved_work/2026-11-09-current-work/ignored-benchmarks-and-logs.tar.gz`.
- Archive SHA-256: `9a5d394114478d4089dbb8176cbae94c6361200dc317d9bd2376e0c9ac682586`.
  Compressed size: 4,438,399 bytes. All 192 regular files (26,568,339 source bytes) were
  verified against the archive by SHA-256; 11 log symlinks were verified with existing
  targets. Symlinks are preserved, not dereferenced: their external targets remain
  required under `/Volumes/CrucialX9/3dvae-poc_logs`.
- Completed P5a/P7/top-five outputs remain in their documented subfolders under
  `/Volumes/CrucialX9/3dvae-poc_checkpoints`; upstream encoders remain under
  `/Volumes/CrucialX9/pretrain_v2_checkpoints`. These directories were checked to exist,
  not re-hashed or copied by this save.
- Local `data/`, `checkpoints/`, and `checkpoints_deep_supervise/` remain unchanged and
  ignored. Raw external sources remain under `/Volumes/CrucialX9/fake_data`. This save
  is not a new backup of those large datasets or checkpoint trees.
- Generated HTML/PNG plan exports, the explicitly unused
  `scripts/generate_reconstructions.py`, caches, environments, temporary extraction
  files, and purgatory remain ignored and untouched. No storage cleanup/move script ran.

## Roots

1. `/Users/donaldpg/synthoseis-3dvae-poc`
2. `/Volumes/CrucialX9/real_data`
3. `/Volumes/CrucialX9/fake_data`
4. `/Users/donaldpg/synthoseis/real_data`
5. `/Users/donaldpg/synthoseis/fake_data`

## Scope

- Every recursive subfolder using strictly more than **0.5 decimal GB** is listed (1 GB = 1,000,000,000 bytes).
- Sizes are allocated disk space from macOS `du -k`, not logical uncompressed array size.
- Zarr stores are directories; they are listed as units. Their internal array/chunk directories are excluded from subfolder rows.
- Hidden directories are included. Symlinks are not followed; external targets are not silently counted under the repo root.
- macOS `._*` metadata sidecars contribute to allocated sizes but are not Zarr stores and are ignored in Zarr payload comparisons.
- Parent and child rows overlap. Do not sum subfolder rows to calculate root usage.
- Each root has an independent alphabetically sorted column list; horizontal placement does not imply a duplicate match.
- Listed Zarr paths are relative to the containing subfolder; `(self)` means that subfolder is the store.
- Name matches are candidates only. Confirmed Zarr payload duplicates require matching relative file layout, Zarr metadata and SHA-256 content fingerprints (excluding macOS sidecars).
- Non-Zarr directory contents are not exhaustively hashed. Differently encoded stores with equal decoded arrays are not detected as byte duplicates.
- This is a live filesystem snapshot; ongoing training may change checkpoint sizes after the scan.

## Root Sizes

| Root | Allocated GB | Qualifying subfolders | Zarr stores |
|---|---:|---:|---:|
| 1 | 166.051 | 35 | 17 |
| 2 | 7.624 | 4 | 5 |
| 3 | 682.201 | 414 | 209 |
| 4 | 4.446 | 2 | 4 |
| 5 | 35.963 | 34 | 19 |

## Fifteen-Column Inventory

| 1: Repo subfolder | 2: External real-data subfolder | 3: External fake-data subfolder | 4: Synthoseis real-data subfolder | 5: Synthoseis fake-data subfolder | 1: Zarr stores | 2: Zarr stores | 3: Zarr stores | 4: Zarr stores | 5: Zarr stores | 1: GB | 2: GB | 3: GB | 4: GB | 5: GB |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| .venv | Netherlands3.real.zarr | seismic__2026.55615090__synthoseis_run_1041 | Netherlands3.real.zarr | seismic__2026.55209454__synthoseis_run_1230 | - | (self) **DUP-03** | model_data.zarr | (self) **DUP-03** | model_data.zarr | 2.701 | 0.765 | 3.138 | 0.717 | 2.193 |
| .venv/lib | Netherlands4.real.zarr | seismic__2026.55615090__synthoseis_run_1041/model_data.zarr | Netherlands4.real.zarr | seismic__2026.55209454__synthoseis_run_1230/model_data.zarr | - | (self) **DUP-04** | (self) | (self) **DUP-04** | (self) | 2.700 | 0.630 | 3.080 | 0.579 | 2.139 |
| .venv/lib/python3.13 | Penobscot3D | seismic__2026.55619470__synthoseis_run_1042 |  | seismic__2026.55215129__synthoseis_run_1231 | - | Penobscot3D.real.zarr | model_data.zarr |  | model_data.zarr | 2.700 | 3.003 | 3.262 |  | 2.264 |
| .venv/lib/python3.13/site-packages | Penobscot3D/Penobscot3D.real.zarr | seismic__2026.55619470__synthoseis_run_1042/model_data.zarr |  | seismic__2026.55215129__synthoseis_run_1231/model_data.zarr | - | (self) | (self) |  | (self) | 2.700 | 0.693 | 3.202 |  | 2.217 |
| .venv/lib/python3.13/site-packages/PySide6 |  | seismic__2026.55623385__synthoseis_run_1043 |  | seismic__2026.55220631__synthoseis_run_1232 | - |  | model_data.zarr |  | model_data.zarr | 1.235 |  | 3.385 |  | 2.242 |
| .venv/lib/python3.13/site-packages/PySide6/Qt |  | seismic__2026.55623385__synthoseis_run_1043/model_data.zarr |  | seismic__2026.55220631__synthoseis_run_1232/model_data.zarr | - |  | (self) |  | (self) | 1.071 |  | 3.320 |  | 2.188 |
| .venv/lib/python3.13/site-packages/PySide6/Qt/lib |  | seismic__2026.55628342__synthoseis_run_1044 |  | seismic__2026.55226235__synthoseis_run_1233 | - |  | model_data.zarr |  | model_data.zarr | 0.944 |  | 3.301 |  | 2.218 |
| .venv/lib/python3.13/site-packages/PySide6/Qt/lib/QtWebEngineCore.framework |  | seismic__2026.55628342__synthoseis_run_1044/model_data.zarr |  | seismic__2026.55226235__synthoseis_run_1233/model_data.zarr | - |  | (self) |  | (self) | 0.620 |  | 3.231 |  | 2.170 |
| .venv/lib/python3.13/site-packages/PySide6/Qt/lib/QtWebEngineCore.framework/Versions |  | seismic__2026.55634773__synthoseis_run_1045 |  | seismic__2026.55232050__synthoseis_run_1234 | - |  | model_data.zarr |  | model_data.zarr | 0.548 |  | 3.263 |  | 2.280 |
| .venv/lib/python3.13/site-packages/PySide6/Qt/lib/QtWebEngineCore.framework/Versions/A |  | seismic__2026.55634773__synthoseis_run_1045/model_data.zarr |  | seismic__2026.55232050__synthoseis_run_1234/model_data.zarr | - |  | (self) |  | (self) | 0.548 |  | 3.195 |  | 2.231 |
| .venv/lib/python3.13/site-packages/vtkmodules |  | seismic__2026.55638871__synthoseis_run_1046 |  | seismic__2026.55238294__synthoseis_run_1235 | - |  | model_data.zarr |  | model_data.zarr | 0.620 |  | 3.388 |  | 2.221 |
| .venv/lib/python3.13/site-packages/vtkmodules/.dylibs |  | seismic__2026.55638871__synthoseis_run_1046/model_data.zarr |  | seismic__2026.55238294__synthoseis_run_1235/model_data.zarr | - |  | (self) |  | (self) | 0.551 |  | 3.315 |  | 2.168 |
| checkpoints |  | seismic__2026.55645882__synthoseis_run_1047 |  | seismic__2026.55244465__synthoseis_run_1236 | - |  | model_data.zarr |  | model_data.zarr | 6.847 |  | 3.237 |  | 2.255 |
| checkpoints/checkpoints_lpips |  | seismic__2026.55645882__synthoseis_run_1047/model_data.zarr |  | seismic__2026.55244465__synthoseis_run_1236/model_data.zarr | - |  | (self) |  | (self) | 1.767 |  | 3.177 |  | 2.202 |
| checkpoints/checkpoints_lpips_1pt5 |  | seismic__2026.55650934__synthoseis_run_1048 |  | seismic__2026.55250314__synthoseis_run_1237 | - |  | model_data.zarr |  | model_data.zarr | 1.768 |  | 3.504 |  | 2.274 |
| checkpoints/geoaware_p0a_dipfix_r2 |  | seismic__2026.55650934__synthoseis_run_1048/model_data.zarr |  | seismic__2026.55250314__synthoseis_run_1237/model_data.zarr | - |  | (self) |  | (self) | 0.617 |  | 3.437 |  | 2.224 |
| checkpoints/geoaware_p0b_r2_seed20260925_r2 |  | seismic__2026.55659211__synthoseis_run_1049 |  | seismic__2026.55255344__synthoseis_run_1238 | - |  | model_data.zarr |  | model_data.zarr | 0.616 |  | 3.525 |  | 2.224 |
| checkpoints/geoaware_p0b_r2_seed20260926_r2 |  | seismic__2026.55659211__synthoseis_run_1049/model_data.zarr |  | seismic__2026.55255344__synthoseis_run_1238/model_data.zarr | - |  | (self) |  | (self) | 0.616 |  | 3.442 |  | 2.169 |
| checkpoints/p4_real_mix_seed20260925 |  | seismic__2026.55667424__synthoseis_run_1050 |  | seismic__2026.55260908__synthoseis_run_1239 | - |  | model_data.zarr |  | model_data.zarr | 0.617 |  | 3.589 |  | 2.248 |
| checkpoints_deep_supervise |  | seismic__2026.55667424__synthoseis_run_1050/model_data.zarr |  | seismic__2026.55260908__synthoseis_run_1239/model_data.zarr | - |  | (self) |  | (self) | 0.749 |  | 3.502 |  | 2.190 |
| data |  | seismic__2026.55677408__synthoseis_run_1051 |  | test | real_test_32-32-64.zarr<br>real_train_32-32-64.zarr<br>real_validation_32-32-64.zarr<br>synth_bench_train.zarr<br>synth_bench_val.zarr<br>synth_train_32-32-64.zarr<br>synth_train_anchored_32-32-64.zarr<br>synth_train_anchored_all180_32-32-64.zarr<br>synth_train_anchored_all200_32-32-64.zarr<br>synth_train_geoscore_noval_32-32-64.zarr<br>synth_val_32-32-64.zarr<br>synth_val_anchored_32-32-64.zarr<br>synth_val_uniform_32-32-64.zarr<br>synth_val_v2_32-32-64.zarr<br>synth_val_v2_uniform_32-32-64.zarr<br>train_32-32-64.zarr<br>val_32-32-64.zarr |  | model_data.zarr |  | CostaRica_part1.real.zarr **DUP-05**<br>Poseidon_part1.real.zarr **DUP-06**<br>Poseidon_part2.real.zarr **DUP-07**<br>cn1.real.zarr **DUP-08** | 155.684 |  | 3.320 |  | 2.506 |
| data/real_test_32-32-64.zarr |  | seismic__2026.55677408__synthoseis_run_1051/model_data.zarr |  | test/cn1.real.zarr | (self) |  | (self) |  | (self) **DUP-08** | 0.710 |  | 3.246 |  | 0.522 |
| data/real_train_32-32-64.zarr |  | seismic__2026.55682963__synthoseis_run_1052 |  | test/CostaRica_part1.real.zarr | (self) |  | model_data.zarr |  | (self) **DUP-05** | 1.667 |  | 3.422 |  | 0.609 |
| data/synth_train_32-32-64.zarr |  | seismic__2026.55682963__synthoseis_run_1052/model_data.zarr |  | validation | (self) |  | (self) |  | seismic__2026.41133304__synthoseis_run_0570/model_data.zarr **DUP-09**<br>seismic__2026.41138579__synthoseis_run_0571/model_data.zarr **DUP-10**<br>seismic__2026.41143924__synthoseis_run_0572/model_data.zarr **DUP-11**<br>seismic__2026.41151562__synthoseis_run_0573/model_data.zarr **DUP-12**<br>seismic__2026.41158562__synthoseis_run_0574/model_data.zarr **DUP-13** | 26.349 |  | 3.360 |  | 11.029 |
| data/synth_train_anchored_32-32-64.zarr |  | seismic__2026.55688100__synthoseis_run_1053 |  | validation/seismic__2026.41133304__synthoseis_run_0570 | (self) |  | model_data.zarr |  | model_data.zarr **DUP-09** | 26.500 |  | 3.545 |  | 2.205 |
| data/synth_train_anchored_all180_32-32-64.zarr |  | seismic__2026.55688100__synthoseis_run_1053/model_data.zarr |  | validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr | (self) |  | (self) |  | (self) **DUP-09** | 26.500 |  | 3.459 |  | 2.150 |
| data/synth_train_anchored_all200_32-32-64.zarr |  | seismic__2026.55696664__synthoseis_run_1054 |  | validation/seismic__2026.41138579__synthoseis_run_0571 | (self) |  | model_data.zarr |  | model_data.zarr **DUP-10** | 28.563 |  | 3.413 |  | 2.199 |
| data/synth_train_geoscore_noval_32-32-64.zarr |  | seismic__2026.55696664__synthoseis_run_1054/model_data.zarr |  | validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr | (self) |  | (self) |  | (self) **DUP-10** | 26.353 |  | 3.338 |  | 2.147 |
| data/synth_val_32-32-64.zarr |  | seismic__2026.55702955__synthoseis_run_1055 |  | validation/seismic__2026.41143924__synthoseis_run_0572 | (self) |  | model_data.zarr |  | model_data.zarr **DUP-11** | 1.220 |  | 3.305 |  | 2.212 |
| data/synth_val_anchored_32-32-64.zarr |  | seismic__2026.55702955__synthoseis_run_1055/model_data.zarr |  | validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr | (self) |  | (self) |  | (self) **DUP-11** | 2.945 |  | 3.225 |  | 2.156 |
| data/synth_val_uniform_32-32-64.zarr |  | seismic__2026.55707149__synthoseis_run_1056 |  | validation/seismic__2026.41151562__synthoseis_run_0573 | (self) |  | model_data.zarr |  | model_data.zarr **DUP-12** | 0.982 |  | 3.374 |  | 2.221 |
| data/synth_val_v2_32-32-64.zarr |  | seismic__2026.55707149__synthoseis_run_1056/model_data.zarr |  | validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr | (self) |  | (self) |  | (self) **DUP-12** | 1.227 |  | 3.300 |  | 2.163 |
| data/synth_val_v2_uniform_32-32-64.zarr |  | seismic__2026.55714288__synthoseis_run_1057 |  | validation/seismic__2026.41158562__synthoseis_run_0574 | (self) |  | model_data.zarr |  | model_data.zarr **DUP-13** | 1.227 |  | 3.473 |  | 2.191 |
| data/train_32-32-64.zarr |  | seismic__2026.55714288__synthoseis_run_1057/model_data.zarr |  | validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr | (self) |  | (self) |  | (self) **DUP-13** | 8.562 |  | 3.410 |  | 2.139 |
| data/val_32-32-64.zarr |  | seismic__2026.55720365__synthoseis_run_1058 |  |  | (self) |  | model_data.zarr |  |  | 2.445 |  | 3.321 |  |  |
|  |  | seismic__2026.55720365__synthoseis_run_1058/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.249 |  |  |
|  |  | seismic__2026.55728007__synthoseis_run_1059 |  |  |  |  | model_data.zarr |  |  |  |  | 3.411 |  |  |
|  |  | seismic__2026.55728007__synthoseis_run_1059/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.333 |  |  |
|  |  | seismic__2026.55734525__synthoseis_run_1060 |  |  |  |  | model_data.zarr |  |  |  |  | 3.421 |  |  |
|  |  | seismic__2026.55734525__synthoseis_run_1060/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.348 |  |  |
|  |  | seismic__2026.55740899__synthoseis_run_1061 |  |  |  |  | model_data.zarr |  |  |  |  | 3.509 |  |  |
|  |  | seismic__2026.55740899__synthoseis_run_1061/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.434 |  |  |
|  |  | seismic__2026.55747692__synthoseis_run_1062 |  |  |  |  | model_data.zarr |  |  |  |  | 3.315 |  |  |
|  |  | seismic__2026.55747692__synthoseis_run_1062/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.248 |  |  |
|  |  | seismic__2026.55752927__synthoseis_run_1063 |  |  |  |  | model_data.zarr |  |  |  |  | 3.328 |  |  |
|  |  | seismic__2026.55752927__synthoseis_run_1063/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.250 |  |  |
|  |  | seismic__2026.55759561__synthoseis_run_1064 |  |  |  |  | model_data.zarr |  |  |  |  | 3.253 |  |  |
|  |  | seismic__2026.55759561__synthoseis_run_1064/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.188 |  |  |
|  |  | seismic__2026.55764208__synthoseis_run_1065 |  |  |  |  | model_data.zarr |  |  |  |  | 3.473 |  |  |
|  |  | seismic__2026.55764208__synthoseis_run_1065/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.410 |  |  |
|  |  | seismic__2026.55769983__synthoseis_run_1066 |  |  |  |  | model_data.zarr |  |  |  |  | 3.203 |  |  |
|  |  | seismic__2026.55769983__synthoseis_run_1066/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.145 |  |  |
|  |  | seismic__2026.55774107__synthoseis_run_1067 |  |  |  |  | model_data.zarr |  |  |  |  | 3.204 |  |  |
|  |  | seismic__2026.55774107__synthoseis_run_1067/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.139 |  |  |
|  |  | seismic__2026.55779024__synthoseis_run_1068 |  |  |  |  | model_data.zarr |  |  |  |  | 3.375 |  |  |
|  |  | seismic__2026.55779024__synthoseis_run_1068/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.294 |  |  |
|  |  | seismic__2026.55785831__synthoseis_run_1069 |  |  |  |  | model_data.zarr |  |  |  |  | 3.266 |  |  |
|  |  | seismic__2026.55785831__synthoseis_run_1069/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.184 |  |  |
|  |  | seismic__2026.55792481__synthoseis_run_1070 |  |  |  |  | model_data.zarr |  |  |  |  | 3.427 |  |  |
|  |  | seismic__2026.55792481__synthoseis_run_1070/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.349 |  |  |
|  |  | seismic__2026.55799982__synthoseis_run_1071 |  |  |  |  | model_data.zarr |  |  |  |  | 3.280 |  |  |
|  |  | seismic__2026.55799982__synthoseis_run_1071/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.208 |  |  |
|  |  | seismic__2026.55806229__synthoseis_run_1072 |  |  |  |  | model_data.zarr |  |  |  |  | 3.405 |  |  |
|  |  | seismic__2026.55806229__synthoseis_run_1072/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.325 |  |  |
|  |  | seismic__2026.55813827__synthoseis_run_1073 |  |  |  |  | model_data.zarr |  |  |  |  | 3.248 |  |  |
|  |  | seismic__2026.55813827__synthoseis_run_1073/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.189 |  |  |
|  |  | seismic__2026.55818117__synthoseis_run_1074 |  |  |  |  | model_data.zarr |  |  |  |  | 3.405 |  |  |
|  |  | seismic__2026.55818117__synthoseis_run_1074/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.338 |  |  |
|  |  | seismic__2026.55826154__synthoseis_run_1075 |  |  |  |  | model_data.zarr |  |  |  |  | 3.365 |  |  |
|  |  | seismic__2026.55826154__synthoseis_run_1075/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.305 |  |  |
|  |  | seismic__2026.55832797__synthoseis_run_1076 |  |  |  |  | model_data.zarr |  |  |  |  | 3.244 |  |  |
|  |  | seismic__2026.55832797__synthoseis_run_1076/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.167 |  |  |
|  |  | seismic__2026.55837522__synthoseis_run_1077 |  |  |  |  | model_data.zarr |  |  |  |  | 3.340 |  |  |
|  |  | seismic__2026.55837522__synthoseis_run_1077/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.268 |  |  |
|  |  | seismic__2026.55843143__synthoseis_run_1078 |  |  |  |  | model_data.zarr |  |  |  |  | 3.371 |  |  |
|  |  | seismic__2026.55843143__synthoseis_run_1078/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.299 |  |  |
|  |  | seismic__2026.55849882__synthoseis_run_1079 |  |  |  |  | model_data.zarr |  |  |  |  | 3.337 |  |  |
|  |  | seismic__2026.55849882__synthoseis_run_1079/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.273 |  |  |
|  |  | seismic__2026.55856356__synthoseis_run_1080 |  |  |  |  | model_data.zarr |  |  |  |  | 3.286 |  |  |
|  |  | seismic__2026.55856356__synthoseis_run_1080/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.224 |  |  |
|  |  | seismic__2026.55861365__synthoseis_run_1081 |  |  |  |  | model_data.zarr |  |  |  |  | 3.280 |  |  |
|  |  | seismic__2026.55861365__synthoseis_run_1081/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.207 |  |  |
|  |  | seismic__2026.55865636__synthoseis_run_1082 |  |  |  |  | model_data.zarr |  |  |  |  | 3.205 |  |  |
|  |  | seismic__2026.55865636__synthoseis_run_1082/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.142 |  |  |
|  |  | seismic__2026.55869867__synthoseis_run_1083 |  |  |  |  | model_data.zarr |  |  |  |  | 3.462 |  |  |
|  |  | seismic__2026.55869867__synthoseis_run_1083/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.398 |  |  |
|  |  | seismic__2026.55876744__synthoseis_run_1084 |  |  |  |  | model_data.zarr |  |  |  |  | 3.438 |  |  |
|  |  | seismic__2026.55876744__synthoseis_run_1084/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.377 |  |  |
|  |  | seismic__2026.55883755__synthoseis_run_1085 |  |  |  |  | model_data.zarr |  |  |  |  | 3.221 |  |  |
|  |  | seismic__2026.55883755__synthoseis_run_1085/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.163 |  |  |
|  |  | seismic__2026.55887811__synthoseis_run_1086 |  |  |  |  | model_data.zarr |  |  |  |  | 3.402 |  |  |
|  |  | seismic__2026.55887811__synthoseis_run_1086/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.334 |  |  |
|  |  | seismic__2026.55894225__synthoseis_run_1087 |  |  |  |  | model_data.zarr |  |  |  |  | 3.399 |  |  |
|  |  | seismic__2026.55894225__synthoseis_run_1087/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.325 |  |  |
|  |  | seismic__2026.55900440__synthoseis_run_1088 |  |  |  |  | model_data.zarr |  |  |  |  | 3.406 |  |  |
|  |  | seismic__2026.55900440__synthoseis_run_1088/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.332 |  |  |
|  |  | seismic__2026.55906851__synthoseis_run_1089 |  |  |  |  | model_data.zarr |  |  |  |  | 3.531 |  |  |
|  |  | seismic__2026.55906851__synthoseis_run_1089/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.461 |  |  |
|  |  | seismic__2026.55915960__synthoseis_run_1090 |  |  |  |  | model_data.zarr |  |  |  |  | 3.507 |  |  |
|  |  | seismic__2026.55915960__synthoseis_run_1090/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.445 |  |  |
|  |  | seismic__2026.55925234__synthoseis_run_1091 |  |  |  |  | model_data.zarr |  |  |  |  | 3.544 |  |  |
|  |  | seismic__2026.55925234__synthoseis_run_1091/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.478 |  |  |
|  |  | seismic__2026.55933127__synthoseis_run_1092 |  |  |  |  | model_data.zarr |  |  |  |  | 3.350 |  |  |
|  |  | seismic__2026.55933127__synthoseis_run_1092/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.274 |  |  |
|  |  | seismic__2026.55938287__synthoseis_run_1093 |  |  |  |  | model_data.zarr |  |  |  |  | 3.274 |  |  |
|  |  | seismic__2026.55938287__synthoseis_run_1093/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.200 |  |  |
|  |  | seismic__2026.55943336__synthoseis_run_1094 |  |  |  |  | model_data.zarr |  |  |  |  | 3.336 |  |  |
|  |  | seismic__2026.55943336__synthoseis_run_1094/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.274 |  |  |
|  |  | seismic__2026.55949285__synthoseis_run_1095 |  |  |  |  | model_data.zarr |  |  |  |  | 3.233 |  |  |
|  |  | seismic__2026.55949285__synthoseis_run_1095/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.161 |  |  |
|  |  | seismic__2026.55954249__synthoseis_run_1096 |  |  |  |  | model_data.zarr |  |  |  |  | 3.269 |  |  |
|  |  | seismic__2026.55954249__synthoseis_run_1096/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.204 |  |  |
|  |  | seismic__2026.55959629__synthoseis_run_1097 |  |  |  |  | model_data.zarr |  |  |  |  | 3.534 |  |  |
|  |  | seismic__2026.55959629__synthoseis_run_1097/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.464 |  |  |
|  |  | seismic__2026.55970803__synthoseis_run_1098 |  |  |  |  | model_data.zarr |  |  |  |  | 3.239 |  |  |
|  |  | seismic__2026.55970803__synthoseis_run_1098/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.167 |  |  |
|  |  | seismic__2026.55975550__synthoseis_run_1099 |  |  |  |  | model_data.zarr |  |  |  |  | 3.300 |  |  |
|  |  | seismic__2026.55975550__synthoseis_run_1099/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.233 |  |  |
|  |  | seismic__2026.55980563__synthoseis_run_1100 |  |  |  |  | model_data.zarr |  |  |  |  | 3.340 |  |  |
|  |  | seismic__2026.55980563__synthoseis_run_1100/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.280 |  |  |
|  |  | seismic__2026.55985718__synthoseis_run_1101 |  |  |  |  | model_data.zarr |  |  |  |  | 3.368 |  |  |
|  |  | seismic__2026.55985718__synthoseis_run_1101/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.292 |  |  |
|  |  | seismic__2026.55990770__synthoseis_run_1102 |  |  |  |  | model_data.zarr |  |  |  |  | 3.234 |  |  |
|  |  | seismic__2026.55990770__synthoseis_run_1102/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.162 |  |  |
|  |  | seismic__2026.55995415__synthoseis_run_1103 |  |  |  |  | model_data.zarr |  |  |  |  | 3.298 |  |  |
|  |  | seismic__2026.55995415__synthoseis_run_1103/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.236 |  |  |
|  |  | seismic__2026.56000028__synthoseis_run_1104 |  |  |  |  | model_data.zarr |  |  |  |  | 3.518 |  |  |
|  |  | seismic__2026.56000028__synthoseis_run_1104/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.450 |  |  |
|  |  | seismic__2026.56009525__synthoseis_run_1105 |  |  |  |  | model_data.zarr |  |  |  |  | 3.203 |  |  |
|  |  | seismic__2026.56009525__synthoseis_run_1105/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.144 |  |  |
|  |  | seismic__2026.56013592__synthoseis_run_1106 |  |  |  |  | model_data.zarr |  |  |  |  | 3.240 |  |  |
|  |  | seismic__2026.56013592__synthoseis_run_1106/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.175 |  |  |
|  |  | seismic__2026.56019860__synthoseis_run_1107 |  |  |  |  | model_data.zarr |  |  |  |  | 3.296 |  |  |
|  |  | seismic__2026.56019860__synthoseis_run_1107/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.223 |  |  |
|  |  | seismic__2026.56025395__synthoseis_run_1108 |  |  |  |  | model_data.zarr |  |  |  |  | 3.368 |  |  |
|  |  | seismic__2026.56025395__synthoseis_run_1108/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.302 |  |  |
|  |  | seismic__2026.56031354__synthoseis_run_1109 |  |  |  |  | model_data.zarr |  |  |  |  | 3.370 |  |  |
|  |  | seismic__2026.56031354__synthoseis_run_1109/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.296 |  |  |
|  |  | seismic__2026.56038121__synthoseis_run_1110 |  |  |  |  | model_data.zarr |  |  |  |  | 3.319 |  |  |
|  |  | seismic__2026.56038121__synthoseis_run_1110/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.237 |  |  |
|  |  | seismic__2026.56044585__synthoseis_run_1111 |  |  |  |  | model_data.zarr |  |  |  |  | 3.386 |  |  |
|  |  | seismic__2026.56044585__synthoseis_run_1111/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.310 |  |  |
|  |  | seismic__2026.56049817__synthoseis_run_1112 |  |  |  |  | model_data.zarr |  |  |  |  | 3.284 |  |  |
|  |  | seismic__2026.56049817__synthoseis_run_1112/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.212 |  |  |
|  |  | seismic__2026.56055905__synthoseis_run_1113 |  |  |  |  | model_data.zarr |  |  |  |  | 3.285 |  |  |
|  |  | seismic__2026.56055905__synthoseis_run_1113/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.210 |  |  |
|  |  | seismic__2026.56061132__synthoseis_run_1114 |  |  |  |  | model_data.zarr |  |  |  |  | 3.296 |  |  |
|  |  | seismic__2026.56061132__synthoseis_run_1114/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.233 |  |  |
|  |  | seismic__2026.56064862__synthoseis_run_1115 |  |  |  |  | model_data.zarr |  |  |  |  | 3.457 |  |  |
|  |  | seismic__2026.56064862__synthoseis_run_1115/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.390 |  |  |
|  |  | seismic__2026.56071894__synthoseis_run_1116 |  |  |  |  | model_data.zarr |  |  |  |  | 3.352 |  |  |
|  |  | seismic__2026.56071894__synthoseis_run_1116/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.273 |  |  |
|  |  | seismic__2026.56077949__synthoseis_run_1117 |  |  |  |  | model_data.zarr |  |  |  |  | 3.288 |  |  |
|  |  | seismic__2026.56077949__synthoseis_run_1117/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.216 |  |  |
|  |  | seismic__2026.56083437__synthoseis_run_1118 |  |  |  |  | model_data.zarr |  |  |  |  | 3.373 |  |  |
|  |  | seismic__2026.56083437__synthoseis_run_1118/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.304 |  |  |
|  |  | seismic__2026.56089675__synthoseis_run_1119 |  |  |  |  | model_data.zarr |  |  |  |  | 3.502 |  |  |
|  |  | seismic__2026.56089675__synthoseis_run_1119/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.435 |  |  |
|  |  | seismic__2026.56096908__synthoseis_run_1120 |  |  |  |  | model_data.zarr |  |  |  |  | 3.414 |  |  |
|  |  | seismic__2026.56096908__synthoseis_run_1120/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.336 |  |  |
|  |  | seismic__2026.56104435__synthoseis_run_1121 |  |  |  |  | model_data.zarr |  |  |  |  | 3.383 |  |  |
|  |  | seismic__2026.56104435__synthoseis_run_1121/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.301 |  |  |
|  |  | seismic__2026.56113120__synthoseis_run_1122 |  |  |  |  | model_data.zarr |  |  |  |  | 3.319 |  |  |
|  |  | seismic__2026.56113120__synthoseis_run_1122/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.250 |  |  |
|  |  | seismic__2026.56119063__synthoseis_run_1123 |  |  |  |  | model_data.zarr |  |  |  |  | 3.370 |  |  |
|  |  | seismic__2026.56119063__synthoseis_run_1123/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.286 |  |  |
|  |  | seismic__2026.56126576__synthoseis_run_1124 |  |  |  |  | model_data.zarr |  |  |  |  | 3.233 |  |  |
|  |  | seismic__2026.56126576__synthoseis_run_1124/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.167 |  |  |
|  |  | seismic__2026.56132292__synthoseis_run_1125 |  |  |  |  | model_data.zarr |  |  |  |  | 3.181 |  |  |
|  |  | seismic__2026.56132292__synthoseis_run_1125/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.114 |  |  |
|  |  | seismic__2026.56136194__synthoseis_run_1126 |  |  |  |  | model_data.zarr |  |  |  |  | 3.398 |  |  |
|  |  | seismic__2026.56136194__synthoseis_run_1126/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.333 |  |  |
|  |  | seismic__2026.56142556__synthoseis_run_1127 |  |  |  |  | model_data.zarr |  |  |  |  | 3.468 |  |  |
|  |  | seismic__2026.56142556__synthoseis_run_1127/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.392 |  |  |
|  |  | seismic__2026.56151509__synthoseis_run_1128 |  |  |  |  | model_data.zarr |  |  |  |  | 3.305 |  |  |
|  |  | seismic__2026.56151509__synthoseis_run_1128/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.223 |  |  |
|  |  | seismic__2026.56157445__synthoseis_run_1129 |  |  |  |  | model_data.zarr |  |  |  |  | 3.153 |  |  |
|  |  | seismic__2026.56157445__synthoseis_run_1129/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.092 |  |  |
|  |  | seismic__2026.56160841__synthoseis_run_1130 |  |  |  |  | model_data.zarr |  |  |  |  | 3.505 |  |  |
|  |  | seismic__2026.56160841__synthoseis_run_1130/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.423 |  |  |
|  |  | seismic__2026.56168074__synthoseis_run_1131 |  |  |  |  | model_data.zarr |  |  |  |  | 3.321 |  |  |
|  |  | seismic__2026.56168074__synthoseis_run_1131/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.240 |  |  |
|  |  | seismic__2026.56174639__synthoseis_run_1132 |  |  |  |  | model_data.zarr |  |  |  |  | 3.277 |  |  |
|  |  | seismic__2026.56174639__synthoseis_run_1132/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.203 |  |  |
|  |  | seismic__2026.56179768__synthoseis_run_1133 |  |  |  |  | model_data.zarr |  |  |  |  | 3.309 |  |  |
|  |  | seismic__2026.56179768__synthoseis_run_1133/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.246 |  |  |
|  |  | seismic__2026.56186239__synthoseis_run_1134 |  |  |  |  | model_data.zarr |  |  |  |  | 3.243 |  |  |
|  |  | seismic__2026.56186239__synthoseis_run_1134/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.164 |  |  |
|  |  | seismic__2026.56192225__synthoseis_run_1135 |  |  |  |  | model_data.zarr |  |  |  |  | 3.452 |  |  |
|  |  | seismic__2026.56192225__synthoseis_run_1135/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.384 |  |  |
|  |  | seismic__2026.56198621__synthoseis_run_1136 |  |  |  |  | model_data.zarr |  |  |  |  | 3.304 |  |  |
|  |  | seismic__2026.56198621__synthoseis_run_1136/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.231 |  |  |
|  |  | seismic__2026.56205358__synthoseis_run_1137 |  |  |  |  | model_data.zarr |  |  |  |  | 3.426 |  |  |
|  |  | seismic__2026.56205358__synthoseis_run_1137/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.351 |  |  |
|  |  | seismic__2026.56213927__synthoseis_run_1138 |  |  |  |  | model_data.zarr |  |  |  |  | 3.409 |  |  |
|  |  | seismic__2026.56213927__synthoseis_run_1138/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.328 |  |  |
|  |  | seismic__2026.56220514__synthoseis_run_1139 |  |  |  |  | model_data.zarr |  |  |  |  | 3.498 |  |  |
|  |  | seismic__2026.56220514__synthoseis_run_1139/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.416 |  |  |
|  |  | seismic__2026.56228494__synthoseis_run_1140 |  |  |  |  | model_data.zarr |  |  |  |  | 3.332 |  |  |
|  |  | seismic__2026.56228494__synthoseis_run_1140/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.255 |  |  |
|  |  | seismic__2026.56233920__synthoseis_run_1141 |  |  |  |  | model_data.zarr |  |  |  |  | 3.312 |  |  |
|  |  | seismic__2026.56233920__synthoseis_run_1141/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.228 |  |  |
|  |  | seismic__2026.56239876__synthoseis_run_1142 |  |  |  |  | model_data.zarr |  |  |  |  | 3.268 |  |  |
|  |  | seismic__2026.56239876__synthoseis_run_1142/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.194 |  |  |
|  |  | seismic__2026.56245558__synthoseis_run_1143 |  |  |  |  | model_data.zarr |  |  |  |  | 3.271 |  |  |
|  |  | seismic__2026.56245558__synthoseis_run_1143/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.204 |  |  |
|  |  | seismic__2026.56252250__synthoseis_run_1144 |  |  |  |  | model_data.zarr |  |  |  |  | 3.330 |  |  |
|  |  | seismic__2026.56252250__synthoseis_run_1144/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.257 |  |  |
|  |  | seismic__2026.56258253__synthoseis_run_1145 |  |  |  |  | model_data.zarr |  |  |  |  | 3.189 |  |  |
|  |  | seismic__2026.56258253__synthoseis_run_1145/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.116 |  |  |
|  |  | seismic__2026.56261740__synthoseis_run_1146 |  |  |  |  | model_data.zarr |  |  |  |  | 3.289 |  |  |
|  |  | seismic__2026.56261740__synthoseis_run_1146/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.225 |  |  |
|  |  | seismic__2026.56267034__synthoseis_run_1147 |  |  |  |  | model_data.zarr |  |  |  |  | 3.268 |  |  |
|  |  | seismic__2026.56267034__synthoseis_run_1147/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.202 |  |  |
|  |  | seismic__2026.56273085__synthoseis_run_1148 |  |  |  |  | model_data.zarr |  |  |  |  | 3.528 |  |  |
|  |  | seismic__2026.56273085__synthoseis_run_1148/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.458 |  |  |
|  |  | seismic__2026.56280144__synthoseis_run_1149 |  |  |  |  | model_data.zarr |  |  |  |  | 3.342 |  |  |
|  |  | seismic__2026.56280144__synthoseis_run_1149/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.273 |  |  |
|  |  | seismic__2026.56285784__synthoseis_run_1150 |  |  |  |  | model_data.zarr |  |  |  |  | 3.333 |  |  |
|  |  | seismic__2026.56285784__synthoseis_run_1150/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.269 |  |  |
|  |  | seismic__2026.56291577__synthoseis_run_1151 |  |  |  |  | model_data.zarr |  |  |  |  | 3.423 |  |  |
|  |  | seismic__2026.56291577__synthoseis_run_1151/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.352 |  |  |
|  |  | seismic__2026.56298204__synthoseis_run_1152 |  |  |  |  | model_data.zarr |  |  |  |  | 3.283 |  |  |
|  |  | seismic__2026.56298204__synthoseis_run_1152/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.216 |  |  |
|  |  | seismic__2026.56303059__synthoseis_run_1153 |  |  |  |  | model_data.zarr |  |  |  |  | 3.237 |  |  |
|  |  | seismic__2026.56303059__synthoseis_run_1153/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.169 |  |  |
|  |  | seismic__2026.56307499__synthoseis_run_1154 |  |  |  |  | model_data.zarr |  |  |  |  | 3.224 |  |  |
|  |  | seismic__2026.56307499__synthoseis_run_1154/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.160 |  |  |
|  |  | seismic__2026.56312567__synthoseis_run_1155 |  |  |  |  | model_data.zarr |  |  |  |  | 3.287 |  |  |
|  |  | seismic__2026.56312567__synthoseis_run_1155/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.213 |  |  |
|  |  | seismic__2026.56317572__synthoseis_run_1156 |  |  |  |  | model_data.zarr |  |  |  |  | 3.315 |  |  |
|  |  | seismic__2026.56317572__synthoseis_run_1156/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.255 |  |  |
|  |  | seismic__2026.56321877__synthoseis_run_1157 |  |  |  |  | model_data.zarr |  |  |  |  | 3.409 |  |  |
|  |  | seismic__2026.56321877__synthoseis_run_1157/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.334 |  |  |
|  |  | seismic__2026.56329232__synthoseis_run_1158 |  |  |  |  | model_data.zarr |  |  |  |  | 3.337 |  |  |
|  |  | seismic__2026.56329232__synthoseis_run_1158/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.267 |  |  |
|  |  | seismic__2026.56336776__synthoseis_run_1159 |  |  |  |  | model_data.zarr |  |  |  |  | 3.397 |  |  |
|  |  | seismic__2026.56336776__synthoseis_run_1159/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.314 |  |  |
|  |  | seismic__2026.56343799__synthoseis_run_1160 |  |  |  |  | model_data.zarr |  |  |  |  | 3.460 |  |  |
|  |  | seismic__2026.56343799__synthoseis_run_1160/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.386 |  |  |
|  |  | seismic__2026.56350941__synthoseis_run_1161 |  |  |  |  | model_data.zarr |  |  |  |  | 3.385 |  |  |
|  |  | seismic__2026.56350941__synthoseis_run_1161/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.311 |  |  |
|  |  | seismic__2026.56358337__synthoseis_run_1162 |  |  |  |  | model_data.zarr |  |  |  |  | 3.330 |  |  |
|  |  | seismic__2026.56358337__synthoseis_run_1162/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.251 |  |  |
|  |  | seismic__2026.56365415__synthoseis_run_1163 |  |  |  |  | model_data.zarr |  |  |  |  | 3.376 |  |  |
|  |  | seismic__2026.56365415__synthoseis_run_1163/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.310 |  |  |
|  |  | seismic__2026.56371809__synthoseis_run_1164 |  |  |  |  | model_data.zarr |  |  |  |  | 3.418 |  |  |
|  |  | seismic__2026.56371809__synthoseis_run_1164/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.353 |  |  |
|  |  | seismic__2026.56378600__synthoseis_run_1165 |  |  |  |  | model_data.zarr |  |  |  |  | 3.283 |  |  |
|  |  | seismic__2026.56378600__synthoseis_run_1165/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.215 |  |  |
|  |  | seismic__2026.56384982__synthoseis_run_1166 |  |  |  |  | model_data.zarr |  |  |  |  | 3.578 |  |  |
|  |  | seismic__2026.56384982__synthoseis_run_1166/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.496 |  |  |
|  |  | seismic__2026.56394820__synthoseis_run_1167 |  |  |  |  | model_data.zarr |  |  |  |  | 3.223 |  |  |
|  |  | seismic__2026.56394820__synthoseis_run_1167/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.150 |  |  |
|  |  | seismic__2026.56400145__synthoseis_run_1168 |  |  |  |  | model_data.zarr |  |  |  |  | 3.464 |  |  |
|  |  | seismic__2026.56400145__synthoseis_run_1168/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.391 |  |  |
|  |  | seismic__2026.56407506__synthoseis_run_1169 |  |  |  |  | model_data.zarr |  |  |  |  | 3.574 |  |  |
|  |  | seismic__2026.56407506__synthoseis_run_1169/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.487 |  |  |
|  |  | seismic__2026.56419037__synthoseis_run_1170 |  |  |  |  | model_data.zarr |  |  |  |  | 3.274 |  |  |
|  |  | seismic__2026.56419037__synthoseis_run_1170/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.209 |  |  |
|  |  | seismic__2026.56424623__synthoseis_run_1171 |  |  |  |  | model_data.zarr |  |  |  |  | 3.330 |  |  |
|  |  | seismic__2026.56424623__synthoseis_run_1171/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.257 |  |  |
|  |  | seismic__2026.56431068__synthoseis_run_1172 |  |  |  |  | model_data.zarr |  |  |  |  | 3.383 |  |  |
|  |  | seismic__2026.56431068__synthoseis_run_1172/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.317 |  |  |
|  |  | seismic__2026.56436181__synthoseis_run_1173 |  |  |  |  | model_data.zarr |  |  |  |  | 3.207 |  |  |
|  |  | seismic__2026.56436181__synthoseis_run_1173/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.142 |  |  |
|  |  | seismic__2026.56439991__synthoseis_run_1174 |  |  |  |  | model_data.zarr |  |  |  |  | 3.381 |  |  |
|  |  | seismic__2026.56439991__synthoseis_run_1174/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.313 |  |  |
|  |  | seismic__2026.56446329__synthoseis_run_1175 |  |  |  |  | model_data.zarr |  |  |  |  | 3.343 |  |  |
|  |  | seismic__2026.56446329__synthoseis_run_1175/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.274 |  |  |
|  |  | seismic__2026.56451385__synthoseis_run_1176 |  |  |  |  | model_data.zarr |  |  |  |  | 3.405 |  |  |
|  |  | seismic__2026.56451385__synthoseis_run_1176/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.323 |  |  |
|  |  | seismic__2026.56457514__synthoseis_run_1177 |  |  |  |  | model_data.zarr |  |  |  |  | 3.387 |  |  |
|  |  | seismic__2026.56457514__synthoseis_run_1177/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.310 |  |  |
|  |  | seismic__2026.56464690__synthoseis_run_1178 |  |  |  |  | model_data.zarr |  |  |  |  | 3.223 |  |  |
|  |  | seismic__2026.56464690__synthoseis_run_1178/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.146 |  |  |
|  |  | seismic__2026.56469525__synthoseis_run_1179 |  |  |  |  | model_data.zarr |  |  |  |  | 3.319 |  |  |
|  |  | seismic__2026.56469525__synthoseis_run_1179/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.246 |  |  |
|  |  | seismic__2026.56475284__synthoseis_run_1180 |  |  |  |  | model_data.zarr |  |  |  |  | 3.301 |  |  |
|  |  | seismic__2026.56475284__synthoseis_run_1180/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.229 |  |  |
|  |  | seismic__2026.56480261__synthoseis_run_1181 |  |  |  |  | model_data.zarr |  |  |  |  | 3.374 |  |  |
|  |  | seismic__2026.56480261__synthoseis_run_1181/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.299 |  |  |
|  |  | seismic__2026.56486862__synthoseis_run_1182 |  |  |  |  | model_data.zarr |  |  |  |  | 3.493 |  |  |
|  |  | seismic__2026.56486862__synthoseis_run_1182/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.415 |  |  |
|  |  | seismic__2026.56493954__synthoseis_run_1183 |  |  |  |  | model_data.zarr |  |  |  |  | 3.271 |  |  |
|  |  | seismic__2026.56493954__synthoseis_run_1183/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.211 |  |  |
|  |  | seismic__2026.56498627__synthoseis_run_1184 |  |  |  |  | model_data.zarr |  |  |  |  | 3.232 |  |  |
|  |  | seismic__2026.56498627__synthoseis_run_1184/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.165 |  |  |
|  |  | seismic__2026.56504435__synthoseis_run_1185 |  |  |  |  | model_data.zarr |  |  |  |  | 3.187 |  |  |
|  |  | seismic__2026.56504435__synthoseis_run_1185/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.122 |  |  |
|  |  | seismic__2026.56510121__synthoseis_run_1186 |  |  |  |  | model_data.zarr |  |  |  |  | 3.316 |  |  |
|  |  | seismic__2026.56510121__synthoseis_run_1186/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.237 |  |  |
|  |  | seismic__2026.56515746__synthoseis_run_1187 |  |  |  |  | model_data.zarr |  |  |  |  | 3.360 |  |  |
|  |  | seismic__2026.56515746__synthoseis_run_1187/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.291 |  |  |
|  |  | seismic__2026.56522588__synthoseis_run_1188 |  |  |  |  | model_data.zarr |  |  |  |  | 3.325 |  |  |
|  |  | seismic__2026.56522588__synthoseis_run_1188/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.242 |  |  |
|  |  | seismic__2026.56528962__synthoseis_run_1189 |  |  |  |  | model_data.zarr |  |  |  |  | 3.298 |  |  |
|  |  | seismic__2026.56528962__synthoseis_run_1189/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.233 |  |  |
|  |  | seismic__2026.56534062__synthoseis_run_1190 |  |  |  |  | model_data.zarr |  |  |  |  | 3.261 |  |  |
|  |  | seismic__2026.56534062__synthoseis_run_1190/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.192 |  |  |
|  |  | seismic__2026.56538485__synthoseis_run_1191 |  |  |  |  | model_data.zarr |  |  |  |  | 3.311 |  |  |
|  |  | seismic__2026.56538485__synthoseis_run_1191/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.239 |  |  |
|  |  | seismic__2026.56542936__synthoseis_run_1192 |  |  |  |  | model_data.zarr |  |  |  |  | 3.544 |  |  |
|  |  | seismic__2026.56542936__synthoseis_run_1192/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.465 |  |  |
|  |  | seismic__2026.56551113__synthoseis_run_1193 |  |  |  |  | model_data.zarr |  |  |  |  | 3.217 |  |  |
|  |  | seismic__2026.56551113__synthoseis_run_1193/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.139 |  |  |
|  |  | seismic__2026.56557181__synthoseis_run_1194 |  |  |  |  | model_data.zarr |  |  |  |  | 3.267 |  |  |
|  |  | seismic__2026.56557181__synthoseis_run_1194/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.205 |  |  |
|  |  | seismic__2026.56562134__synthoseis_run_1195 |  |  |  |  | model_data.zarr |  |  |  |  | 3.446 |  |  |
|  |  | seismic__2026.56562134__synthoseis_run_1195/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.372 |  |  |
|  |  | seismic__2026.56569270__synthoseis_run_1196 |  |  |  |  | model_data.zarr |  |  |  |  | 3.249 |  |  |
|  |  | seismic__2026.56569270__synthoseis_run_1196/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.189 |  |  |
|  |  | seismic__2026.56573275__synthoseis_run_1197 |  |  |  |  | model_data.zarr |  |  |  |  | 3.309 |  |  |
|  |  | seismic__2026.56573275__synthoseis_run_1197/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.243 |  |  |
|  |  | seismic__2026.56578709__synthoseis_run_1198 |  |  |  |  | model_data.zarr |  |  |  |  | 3.320 |  |  |
|  |  | seismic__2026.56578709__synthoseis_run_1198/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.240 |  |  |
|  |  | seismic__2026.56584647__synthoseis_run_1199 |  |  |  |  | model_data.zarr |  |  |  |  | 3.381 |  |  |
|  |  | seismic__2026.56584647__synthoseis_run_1199/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.304 |  |  |
|  |  | seismic__2026.56591748__synthoseis_run_1200 |  |  |  |  | model_data.zarr |  |  |  |  | 3.227 |  |  |
|  |  | seismic__2026.56591748__synthoseis_run_1200/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.158 |  |  |
|  |  | seismic__2026.56595177__synthoseis_run_1201 |  |  |  |  | model_data.zarr |  |  |  |  | 3.196 |  |  |
|  |  | seismic__2026.56595177__synthoseis_run_1201/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.127 |  |  |
|  |  | seismic__2026.56599982__synthoseis_run_1202 |  |  |  |  | model_data.zarr |  |  |  |  | 3.362 |  |  |
|  |  | seismic__2026.56599982__synthoseis_run_1202/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.300 |  |  |
|  |  | seismic__2026.56605050__synthoseis_run_1203 |  |  |  |  | model_data.zarr |  |  |  |  | 3.522 |  |  |
|  |  | seismic__2026.56605050__synthoseis_run_1203/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.440 |  |  |
|  |  | seismic__2026.56613975__synthoseis_run_1204 |  |  |  |  | model_data.zarr |  |  |  |  | 3.332 |  |  |
|  |  | seismic__2026.56613975__synthoseis_run_1204/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.266 |  |  |
|  |  | seismic__2026.56619645__synthoseis_run_1205 |  |  |  |  | model_data.zarr |  |  |  |  | 3.320 |  |  |
|  |  | seismic__2026.56619645__synthoseis_run_1205/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.243 |  |  |
|  |  | seismic__2026.56626453__synthoseis_run_1206 |  |  |  |  | model_data.zarr |  |  |  |  | 3.204 |  |  |
|  |  | seismic__2026.56626453__synthoseis_run_1206/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.137 |  |  |
|  |  | seismic__2026.56630485__synthoseis_run_1207 |  |  |  |  | model_data.zarr |  |  |  |  | 3.321 |  |  |
|  |  | seismic__2026.56630485__synthoseis_run_1207/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.248 |  |  |
|  |  | seismic__2026.56635043__synthoseis_run_1208 |  |  |  |  | model_data.zarr |  |  |  |  | 3.298 |  |  |
|  |  | seismic__2026.56635043__synthoseis_run_1208/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.224 |  |  |
|  |  | seismic__2026.56640519__synthoseis_run_1209 |  |  |  |  | model_data.zarr |  |  |  |  | 3.238 |  |  |
|  |  | seismic__2026.56640519__synthoseis_run_1209/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.170 |  |  |
|  |  | seismic__2026.56644802__synthoseis_run_1210 |  |  |  |  | model_data.zarr |  |  |  |  | 3.259 |  |  |
|  |  | seismic__2026.56644802__synthoseis_run_1210/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.198 |  |  |
|  |  | seismic__2026.56650130__synthoseis_run_1211 |  |  |  |  | model_data.zarr |  |  |  |  | 3.331 |  |  |
|  |  | seismic__2026.56650130__synthoseis_run_1211/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.264 |  |  |
|  |  | seismic__2026.56657048__synthoseis_run_1212 |  |  |  |  | model_data.zarr |  |  |  |  | 3.506 |  |  |
|  |  | seismic__2026.56657048__synthoseis_run_1212/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.425 |  |  |
|  |  | seismic__2026.56667915__synthoseis_run_1213 |  |  |  |  | model_data.zarr |  |  |  |  | 3.317 |  |  |
|  |  | seismic__2026.56667915__synthoseis_run_1213/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.244 |  |  |
|  |  | seismic__2026.56674553__synthoseis_run_1214 |  |  |  |  | model_data.zarr |  |  |  |  | 3.252 |  |  |
|  |  | seismic__2026.56674553__synthoseis_run_1214/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.186 |  |  |
|  |  | seismic__2026.56679515__synthoseis_run_1215 |  |  |  |  | model_data.zarr |  |  |  |  | 3.279 |  |  |
|  |  | seismic__2026.56679515__synthoseis_run_1215/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.205 |  |  |
|  |  | seismic__2026.56686066__synthoseis_run_1216 |  |  |  |  | model_data.zarr |  |  |  |  | 3.298 |  |  |
|  |  | seismic__2026.56686066__synthoseis_run_1216/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.238 |  |  |
|  |  | seismic__2026.56690147__synthoseis_run_1217 |  |  |  |  | model_data.zarr |  |  |  |  | 3.276 |  |  |
|  |  | seismic__2026.56690147__synthoseis_run_1217/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.209 |  |  |
|  |  | seismic__2026.56695710__synthoseis_run_1218 |  |  |  |  | model_data.zarr |  |  |  |  | 3.235 |  |  |
|  |  | seismic__2026.56695710__synthoseis_run_1218/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.161 |  |  |
|  |  | seismic__2026.56700464__synthoseis_run_1219 |  |  |  |  | model_data.zarr |  |  |  |  | 3.370 |  |  |
|  |  | seismic__2026.56700464__synthoseis_run_1219/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.304 |  |  |
|  |  | seismic__2026.56707148__synthoseis_run_1220 |  |  |  |  | model_data.zarr |  |  |  |  | 3.387 |  |  |
|  |  | seismic__2026.56707148__synthoseis_run_1220/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.311 |  |  |
|  |  | test |  |  |  |  | CostaRica_part1.real.zarr **DUP-05**<br>Poseidon_part1.real.zarr **DUP-06**<br>Poseidon_part2.real.zarr **DUP-07**<br>cn1.real.zarr **DUP-08** |  |  |  |  | 2.666 |  |  |
|  |  | test/cn1.real.zarr |  |  |  |  | (self) **DUP-08** |  |  |  |  | 0.573 |  |  |
|  |  | test/CostaRica_part1.real.zarr |  |  |  |  | (self) **DUP-05** |  |  |  |  | 0.650 |  |  |
|  |  | validation |  |  |  |  | seismic__2026.41133304__synthoseis_run_0570/model_data.zarr **DUP-09**<br>seismic__2026.41138579__synthoseis_run_0571/model_data.zarr **DUP-10**<br>seismic__2026.41143924__synthoseis_run_0572/model_data.zarr **DUP-11**<br>seismic__2026.41151562__synthoseis_run_0573/model_data.zarr **DUP-12**<br>seismic__2026.41158562__synthoseis_run_0574/model_data.zarr **DUP-13**<br>seismic__2026.56713658__synthoseis_run_1221/model_data.zarr<br>seismic__2026.56719363__synthoseis_run_1222/model_data.zarr<br>seismic__2026.56725661__synthoseis_run_1223/model_data.zarr<br>seismic__2026.56733317__synthoseis_run_1224/model_data.zarr<br>seismic__2026.56738951__synthoseis_run_1225/model_data.zarr<br>seismic__2026.56745992__synthoseis_run_1226/model_data.zarr<br>seismic__2026.56749584__synthoseis_run_1227/model_data.zarr<br>seismic__2026.56754446__synthoseis_run_1228/model_data.zarr<br>seismic__2026.56758041__synthoseis_run_1229/model_data.zarr<br>seismic__2026.56761906__synthoseis_run_1230/model_data.zarr<br>seismic__2026.56767780__synthoseis_run_1231/model_data.zarr<br>seismic__2026.56773490__synthoseis_run_1232/model_data.zarr<br>seismic__2026.56779472__synthoseis_run_1233/model_data.zarr<br>seismic__2026.56785579__synthoseis_run_1234/model_data.zarr<br>seismic__2026.56792013__synthoseis_run_1235/model_data.zarr<br>seismic__2026.56798438__synthoseis_run_1236/model_data.zarr<br>seismic__2026.56804549__synthoseis_run_1237/model_data.zarr<br>seismic__2026.56809936__synthoseis_run_1238/model_data.zarr<br>seismic__2026.56815746__synthoseis_run_1239/model_data.zarr<br>seismic__2026.56822189__synthoseis_run_1040/model_data.zarr |  |  |  |  | 77.918 |  |  |
|  |  | validation/seismic__2026.41133304__synthoseis_run_0570 |  |  |  |  | model_data.zarr **DUP-09** |  |  |  |  | 2.389 |  |  |
|  |  | validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr |  |  |  |  | (self) **DUP-09** |  |  |  |  | 2.291 |  |  |
|  |  | validation/seismic__2026.41138579__synthoseis_run_0571 |  |  |  |  | model_data.zarr **DUP-10** |  |  |  |  | 2.388 |  |  |
|  |  | validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr |  |  |  |  | (self) **DUP-10** |  |  |  |  | 2.291 |  |  |
|  |  | validation/seismic__2026.41143924__synthoseis_run_0572 |  |  |  |  | model_data.zarr **DUP-11** |  |  |  |  | 2.395 |  |  |
|  |  | validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr |  |  |  |  | (self) **DUP-11** |  |  |  |  | 2.309 |  |  |
|  |  | validation/seismic__2026.41151562__synthoseis_run_0573 |  |  |  |  | model_data.zarr **DUP-12** |  |  |  |  | 2.405 |  |  |
|  |  | validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr |  |  |  |  | (self) **DUP-12** |  |  |  |  | 2.307 |  |  |
|  |  | validation/seismic__2026.41158562__synthoseis_run_0574 |  |  |  |  | model_data.zarr **DUP-13** |  |  |  |  | 2.377 |  |  |
|  |  | validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr |  |  |  |  | (self) **DUP-13** |  |  |  |  | 2.290 |  |  |
|  |  | validation/seismic__2026.56713658__synthoseis_run_1221 |  |  |  |  | model_data.zarr |  |  |  |  | 3.312 |  |  |
|  |  | validation/seismic__2026.56713658__synthoseis_run_1221/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.243 |  |  |
|  |  | validation/seismic__2026.56719363__synthoseis_run_1222 |  |  |  |  | model_data.zarr |  |  |  |  | 3.291 |  |  |
|  |  | validation/seismic__2026.56719363__synthoseis_run_1222/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.227 |  |  |
|  |  | validation/seismic__2026.56725661__synthoseis_run_1223 |  |  |  |  | model_data.zarr |  |  |  |  | 3.549 |  |  |
|  |  | validation/seismic__2026.56725661__synthoseis_run_1223/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.487 |  |  |
|  |  | validation/seismic__2026.56733317__synthoseis_run_1224 |  |  |  |  | model_data.zarr |  |  |  |  | 3.319 |  |  |
|  |  | validation/seismic__2026.56733317__synthoseis_run_1224/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.251 |  |  |
|  |  | validation/seismic__2026.56738951__synthoseis_run_1225 |  |  |  |  | model_data.zarr |  |  |  |  | 3.286 |  |  |
|  |  | validation/seismic__2026.56738951__synthoseis_run_1225/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.202 |  |  |
|  |  | validation/seismic__2026.56745992__synthoseis_run_1226 |  |  |  |  | model_data.zarr |  |  |  |  | 3.162 |  |  |
|  |  | validation/seismic__2026.56745992__synthoseis_run_1226/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.086 |  |  |
|  |  | validation/seismic__2026.56749584__synthoseis_run_1227 |  |  |  |  | model_data.zarr |  |  |  |  | 3.330 |  |  |
|  |  | validation/seismic__2026.56749584__synthoseis_run_1227/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.271 |  |  |
|  |  | validation/seismic__2026.56754446__synthoseis_run_1228 |  |  |  |  | model_data.zarr |  |  |  |  | 3.258 |  |  |
|  |  | validation/seismic__2026.56754446__synthoseis_run_1228/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.196 |  |  |
|  |  | validation/seismic__2026.56758041__synthoseis_run_1229 |  |  |  |  | model_data.zarr |  |  |  |  | 3.221 |  |  |
|  |  | validation/seismic__2026.56758041__synthoseis_run_1229/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.152 |  |  |
|  |  | validation/seismic__2026.56761906__synthoseis_run_1230 |  |  |  |  | model_data.zarr |  |  |  |  | 3.204 |  |  |
|  |  | validation/seismic__2026.56761906__synthoseis_run_1230/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.132 |  |  |
|  |  | validation/seismic__2026.56767780__synthoseis_run_1231 |  |  |  |  | model_data.zarr |  |  |  |  | 3.284 |  |  |
|  |  | validation/seismic__2026.56767780__synthoseis_run_1231/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.223 |  |  |
|  |  | validation/seismic__2026.56773490__synthoseis_run_1232 |  |  |  |  | model_data.zarr |  |  |  |  | 3.320 |  |  |
|  |  | validation/seismic__2026.56773490__synthoseis_run_1232/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.247 |  |  |
|  |  | validation/seismic__2026.56779472__synthoseis_run_1233 |  |  |  |  | model_data.zarr |  |  |  |  | 3.329 |  |  |
|  |  | validation/seismic__2026.56779472__synthoseis_run_1233/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.266 |  |  |
|  |  | validation/seismic__2026.56785579__synthoseis_run_1234 |  |  |  |  | model_data.zarr |  |  |  |  | 3.334 |  |  |
|  |  | validation/seismic__2026.56785579__synthoseis_run_1234/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.272 |  |  |
|  |  | validation/seismic__2026.56792013__synthoseis_run_1235 |  |  |  |  | model_data.zarr |  |  |  |  | 3.310 |  |  |
|  |  | validation/seismic__2026.56792013__synthoseis_run_1235/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.232 |  |  |
|  |  | validation/seismic__2026.56798438__synthoseis_run_1236 |  |  |  |  | model_data.zarr |  |  |  |  | 3.309 |  |  |
|  |  | validation/seismic__2026.56798438__synthoseis_run_1236/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.241 |  |  |
|  |  | validation/seismic__2026.56804549__synthoseis_run_1237 |  |  |  |  | model_data.zarr |  |  |  |  | 3.307 |  |  |
|  |  | validation/seismic__2026.56804549__synthoseis_run_1237/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.241 |  |  |
|  |  | validation/seismic__2026.56809936__synthoseis_run_1238 |  |  |  |  | model_data.zarr |  |  |  |  | 3.250 |  |  |
|  |  | validation/seismic__2026.56809936__synthoseis_run_1238/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.174 |  |  |
|  |  | validation/seismic__2026.56815746__synthoseis_run_1239 |  |  |  |  | model_data.zarr |  |  |  |  | 3.278 |  |  |
|  |  | validation/seismic__2026.56815746__synthoseis_run_1239/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.192 |  |  |
|  |  | validation/seismic__2026.56822189__synthoseis_run_1040 |  |  |  |  | model_data.zarr |  |  |  |  | 3.308 |  |  |
|  |  | validation/seismic__2026.56822189__synthoseis_run_1040/model_data.zarr |  |  |  |  | (self) |  |  |  |  | 3.233 |  |  |

## Confirmed Zarr Content Duplicates

Confirmed groups have byte-identical Zarr payloads and metadata; macOS sidecars are excluded. This is not a deletion recommendation.

### DUP-01
- Root 2: `Netherlands1.real.zarr`
- Root 4: `Netherlands1.real.zarr`
### DUP-02
- Root 2: `Netherlands2.real.zarr`
- Root 4: `Netherlands2.real.zarr`
### DUP-03
- Root 2: `Netherlands3.real.zarr`
- Root 4: `Netherlands3.real.zarr`
### DUP-04
- Root 2: `Netherlands4.real.zarr`
- Root 4: `Netherlands4.real.zarr`
### DUP-05
- Root 3: `test/CostaRica_part1.real.zarr`
- Root 5: `test/CostaRica_part1.real.zarr`
### DUP-06
- Root 3: `test/Poseidon_part1.real.zarr`
- Root 5: `test/Poseidon_part1.real.zarr`
### DUP-07
- Root 3: `test/Poseidon_part2.real.zarr`
- Root 5: `test/Poseidon_part2.real.zarr`
### DUP-08
- Root 3: `test/cn1.real.zarr`
- Root 5: `test/cn1.real.zarr`
### DUP-09
- Root 3: `validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr`
- Root 5: `validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr`
### DUP-10
- Root 3: `validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr`
- Root 5: `validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr`
### DUP-11
- Root 3: `validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr`
- Root 5: `validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr`
### DUP-12
- Root 3: `validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr`
- Root 5: `validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr`
### DUP-13
- Root 3: `validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr`
- Root 5: `validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr`

## Cross-Root Name Matches

- **CostaRica_part1.real.zarr** (name only): root 3: `test/CostaRica_part1.real.zarr`; root 5: `test/CostaRica_part1.real.zarr`
- **Netherlands3.real.zarr** (name only): root 2: `Netherlands3.real.zarr`; root 4: `Netherlands3.real.zarr`
- **Netherlands4.real.zarr** (name only): root 2: `Netherlands4.real.zarr`; root 4: `Netherlands4.real.zarr`
- **cn1.real.zarr** (name only): root 3: `test/cn1.real.zarr`; root 5: `test/cn1.real.zarr`
- **model_data.zarr** (name only): root 3: `seismic__2026.55615090__synthoseis_run_1041/model_data.zarr`; root 3: `seismic__2026.55619470__synthoseis_run_1042/model_data.zarr`; root 3: `seismic__2026.55623385__synthoseis_run_1043/model_data.zarr`; root 3: `seismic__2026.55628342__synthoseis_run_1044/model_data.zarr`; root 3: `seismic__2026.55634773__synthoseis_run_1045/model_data.zarr`; root 3: `seismic__2026.55638871__synthoseis_run_1046/model_data.zarr`; root 3: `seismic__2026.55645882__synthoseis_run_1047/model_data.zarr`; root 3: `seismic__2026.55650934__synthoseis_run_1048/model_data.zarr`; root 3: `seismic__2026.55659211__synthoseis_run_1049/model_data.zarr`; root 3: `seismic__2026.55667424__synthoseis_run_1050/model_data.zarr`; root 3: `seismic__2026.55677408__synthoseis_run_1051/model_data.zarr`; root 3: `seismic__2026.55682963__synthoseis_run_1052/model_data.zarr`; root 3: `seismic__2026.55688100__synthoseis_run_1053/model_data.zarr`; root 3: `seismic__2026.55696664__synthoseis_run_1054/model_data.zarr`; root 3: `seismic__2026.55702955__synthoseis_run_1055/model_data.zarr`; root 3: `seismic__2026.55707149__synthoseis_run_1056/model_data.zarr`; root 3: `seismic__2026.55714288__synthoseis_run_1057/model_data.zarr`; root 3: `seismic__2026.55720365__synthoseis_run_1058/model_data.zarr`; root 3: `seismic__2026.55728007__synthoseis_run_1059/model_data.zarr`; root 3: `seismic__2026.55734525__synthoseis_run_1060/model_data.zarr`; root 3: `seismic__2026.55740899__synthoseis_run_1061/model_data.zarr`; root 3: `seismic__2026.55747692__synthoseis_run_1062/model_data.zarr`; root 3: `seismic__2026.55752927__synthoseis_run_1063/model_data.zarr`; root 3: `seismic__2026.55759561__synthoseis_run_1064/model_data.zarr`; root 3: `seismic__2026.55764208__synthoseis_run_1065/model_data.zarr`; root 3: `seismic__2026.55769983__synthoseis_run_1066/model_data.zarr`; root 3: `seismic__2026.55774107__synthoseis_run_1067/model_data.zarr`; root 3: `seismic__2026.55779024__synthoseis_run_1068/model_data.zarr`; root 3: `seismic__2026.55785831__synthoseis_run_1069/model_data.zarr`; root 3: `seismic__2026.55792481__synthoseis_run_1070/model_data.zarr`; root 3: `seismic__2026.55799982__synthoseis_run_1071/model_data.zarr`; root 3: `seismic__2026.55806229__synthoseis_run_1072/model_data.zarr`; root 3: `seismic__2026.55813827__synthoseis_run_1073/model_data.zarr`; root 3: `seismic__2026.55818117__synthoseis_run_1074/model_data.zarr`; root 3: `seismic__2026.55826154__synthoseis_run_1075/model_data.zarr`; root 3: `seismic__2026.55832797__synthoseis_run_1076/model_data.zarr`; root 3: `seismic__2026.55837522__synthoseis_run_1077/model_data.zarr`; root 3: `seismic__2026.55843143__synthoseis_run_1078/model_data.zarr`; root 3: `seismic__2026.55849882__synthoseis_run_1079/model_data.zarr`; root 3: `seismic__2026.55856356__synthoseis_run_1080/model_data.zarr`; root 3: `seismic__2026.55861365__synthoseis_run_1081/model_data.zarr`; root 3: `seismic__2026.55865636__synthoseis_run_1082/model_data.zarr`; root 3: `seismic__2026.55869867__synthoseis_run_1083/model_data.zarr`; root 3: `seismic__2026.55876744__synthoseis_run_1084/model_data.zarr`; root 3: `seismic__2026.55883755__synthoseis_run_1085/model_data.zarr`; root 3: `seismic__2026.55887811__synthoseis_run_1086/model_data.zarr`; root 3: `seismic__2026.55894225__synthoseis_run_1087/model_data.zarr`; root 3: `seismic__2026.55900440__synthoseis_run_1088/model_data.zarr`; root 3: `seismic__2026.55906851__synthoseis_run_1089/model_data.zarr`; root 3: `seismic__2026.55915960__synthoseis_run_1090/model_data.zarr`; root 3: `seismic__2026.55925234__synthoseis_run_1091/model_data.zarr`; root 3: `seismic__2026.55933127__synthoseis_run_1092/model_data.zarr`; root 3: `seismic__2026.55938287__synthoseis_run_1093/model_data.zarr`; root 3: `seismic__2026.55943336__synthoseis_run_1094/model_data.zarr`; root 3: `seismic__2026.55949285__synthoseis_run_1095/model_data.zarr`; root 3: `seismic__2026.55954249__synthoseis_run_1096/model_data.zarr`; root 3: `seismic__2026.55959629__synthoseis_run_1097/model_data.zarr`; root 3: `seismic__2026.55970803__synthoseis_run_1098/model_data.zarr`; root 3: `seismic__2026.55975550__synthoseis_run_1099/model_data.zarr`; root 3: `seismic__2026.55980563__synthoseis_run_1100/model_data.zarr`; root 3: `seismic__2026.55985718__synthoseis_run_1101/model_data.zarr`; root 3: `seismic__2026.55990770__synthoseis_run_1102/model_data.zarr`; root 3: `seismic__2026.55995415__synthoseis_run_1103/model_data.zarr`; root 3: `seismic__2026.56000028__synthoseis_run_1104/model_data.zarr`; root 3: `seismic__2026.56009525__synthoseis_run_1105/model_data.zarr`; root 3: `seismic__2026.56013592__synthoseis_run_1106/model_data.zarr`; root 3: `seismic__2026.56019860__synthoseis_run_1107/model_data.zarr`; root 3: `seismic__2026.56025395__synthoseis_run_1108/model_data.zarr`; root 3: `seismic__2026.56031354__synthoseis_run_1109/model_data.zarr`; root 3: `seismic__2026.56038121__synthoseis_run_1110/model_data.zarr`; root 3: `seismic__2026.56044585__synthoseis_run_1111/model_data.zarr`; root 3: `seismic__2026.56049817__synthoseis_run_1112/model_data.zarr`; root 3: `seismic__2026.56055905__synthoseis_run_1113/model_data.zarr`; root 3: `seismic__2026.56061132__synthoseis_run_1114/model_data.zarr`; root 3: `seismic__2026.56064862__synthoseis_run_1115/model_data.zarr`; root 3: `seismic__2026.56071894__synthoseis_run_1116/model_data.zarr`; root 3: `seismic__2026.56077949__synthoseis_run_1117/model_data.zarr`; root 3: `seismic__2026.56083437__synthoseis_run_1118/model_data.zarr`; root 3: `seismic__2026.56089675__synthoseis_run_1119/model_data.zarr`; root 3: `seismic__2026.56096908__synthoseis_run_1120/model_data.zarr`; root 3: `seismic__2026.56104435__synthoseis_run_1121/model_data.zarr`; root 3: `seismic__2026.56113120__synthoseis_run_1122/model_data.zarr`; root 3: `seismic__2026.56119063__synthoseis_run_1123/model_data.zarr`; root 3: `seismic__2026.56126576__synthoseis_run_1124/model_data.zarr`; root 3: `seismic__2026.56132292__synthoseis_run_1125/model_data.zarr`; root 3: `seismic__2026.56136194__synthoseis_run_1126/model_data.zarr`; root 3: `seismic__2026.56142556__synthoseis_run_1127/model_data.zarr`; root 3: `seismic__2026.56151509__synthoseis_run_1128/model_data.zarr`; root 3: `seismic__2026.56157445__synthoseis_run_1129/model_data.zarr`; root 3: `seismic__2026.56160841__synthoseis_run_1130/model_data.zarr`; root 3: `seismic__2026.56168074__synthoseis_run_1131/model_data.zarr`; root 3: `seismic__2026.56174639__synthoseis_run_1132/model_data.zarr`; root 3: `seismic__2026.56179768__synthoseis_run_1133/model_data.zarr`; root 3: `seismic__2026.56186239__synthoseis_run_1134/model_data.zarr`; root 3: `seismic__2026.56192225__synthoseis_run_1135/model_data.zarr`; root 3: `seismic__2026.56198621__synthoseis_run_1136/model_data.zarr`; root 3: `seismic__2026.56205358__synthoseis_run_1137/model_data.zarr`; root 3: `seismic__2026.56213927__synthoseis_run_1138/model_data.zarr`; root 3: `seismic__2026.56220514__synthoseis_run_1139/model_data.zarr`; root 3: `seismic__2026.56228494__synthoseis_run_1140/model_data.zarr`; root 3: `seismic__2026.56233920__synthoseis_run_1141/model_data.zarr`; root 3: `seismic__2026.56239876__synthoseis_run_1142/model_data.zarr`; root 3: `seismic__2026.56245558__synthoseis_run_1143/model_data.zarr`; root 3: `seismic__2026.56252250__synthoseis_run_1144/model_data.zarr`; root 3: `seismic__2026.56258253__synthoseis_run_1145/model_data.zarr`; root 3: `seismic__2026.56261740__synthoseis_run_1146/model_data.zarr`; root 3: `seismic__2026.56267034__synthoseis_run_1147/model_data.zarr`; root 3: `seismic__2026.56273085__synthoseis_run_1148/model_data.zarr`; root 3: `seismic__2026.56280144__synthoseis_run_1149/model_data.zarr`; root 3: `seismic__2026.56285784__synthoseis_run_1150/model_data.zarr`; root 3: `seismic__2026.56291577__synthoseis_run_1151/model_data.zarr`; root 3: `seismic__2026.56298204__synthoseis_run_1152/model_data.zarr`; root 3: `seismic__2026.56303059__synthoseis_run_1153/model_data.zarr`; root 3: `seismic__2026.56307499__synthoseis_run_1154/model_data.zarr`; root 3: `seismic__2026.56312567__synthoseis_run_1155/model_data.zarr`; root 3: `seismic__2026.56317572__synthoseis_run_1156/model_data.zarr`; root 3: `seismic__2026.56321877__synthoseis_run_1157/model_data.zarr`; root 3: `seismic__2026.56329232__synthoseis_run_1158/model_data.zarr`; root 3: `seismic__2026.56336776__synthoseis_run_1159/model_data.zarr`; root 3: `seismic__2026.56343799__synthoseis_run_1160/model_data.zarr`; root 3: `seismic__2026.56350941__synthoseis_run_1161/model_data.zarr`; root 3: `seismic__2026.56358337__synthoseis_run_1162/model_data.zarr`; root 3: `seismic__2026.56365415__synthoseis_run_1163/model_data.zarr`; root 3: `seismic__2026.56371809__synthoseis_run_1164/model_data.zarr`; root 3: `seismic__2026.56378600__synthoseis_run_1165/model_data.zarr`; root 3: `seismic__2026.56384982__synthoseis_run_1166/model_data.zarr`; root 3: `seismic__2026.56394820__synthoseis_run_1167/model_data.zarr`; root 3: `seismic__2026.56400145__synthoseis_run_1168/model_data.zarr`; root 3: `seismic__2026.56407506__synthoseis_run_1169/model_data.zarr`; root 3: `seismic__2026.56419037__synthoseis_run_1170/model_data.zarr`; root 3: `seismic__2026.56424623__synthoseis_run_1171/model_data.zarr`; root 3: `seismic__2026.56431068__synthoseis_run_1172/model_data.zarr`; root 3: `seismic__2026.56436181__synthoseis_run_1173/model_data.zarr`; root 3: `seismic__2026.56439991__synthoseis_run_1174/model_data.zarr`; root 3: `seismic__2026.56446329__synthoseis_run_1175/model_data.zarr`; root 3: `seismic__2026.56451385__synthoseis_run_1176/model_data.zarr`; root 3: `seismic__2026.56457514__synthoseis_run_1177/model_data.zarr`; root 3: `seismic__2026.56464690__synthoseis_run_1178/model_data.zarr`; root 3: `seismic__2026.56469525__synthoseis_run_1179/model_data.zarr`; root 3: `seismic__2026.56475284__synthoseis_run_1180/model_data.zarr`; root 3: `seismic__2026.56480261__synthoseis_run_1181/model_data.zarr`; root 3: `seismic__2026.56486862__synthoseis_run_1182/model_data.zarr`; root 3: `seismic__2026.56493954__synthoseis_run_1183/model_data.zarr`; root 3: `seismic__2026.56498627__synthoseis_run_1184/model_data.zarr`; root 3: `seismic__2026.56504435__synthoseis_run_1185/model_data.zarr`; root 3: `seismic__2026.56510121__synthoseis_run_1186/model_data.zarr`; root 3: `seismic__2026.56515746__synthoseis_run_1187/model_data.zarr`; root 3: `seismic__2026.56522588__synthoseis_run_1188/model_data.zarr`; root 3: `seismic__2026.56528962__synthoseis_run_1189/model_data.zarr`; root 3: `seismic__2026.56534062__synthoseis_run_1190/model_data.zarr`; root 3: `seismic__2026.56538485__synthoseis_run_1191/model_data.zarr`; root 3: `seismic__2026.56542936__synthoseis_run_1192/model_data.zarr`; root 3: `seismic__2026.56551113__synthoseis_run_1193/model_data.zarr`; root 3: `seismic__2026.56557181__synthoseis_run_1194/model_data.zarr`; root 3: `seismic__2026.56562134__synthoseis_run_1195/model_data.zarr`; root 3: `seismic__2026.56569270__synthoseis_run_1196/model_data.zarr`; root 3: `seismic__2026.56573275__synthoseis_run_1197/model_data.zarr`; root 3: `seismic__2026.56578709__synthoseis_run_1198/model_data.zarr`; root 3: `seismic__2026.56584647__synthoseis_run_1199/model_data.zarr`; root 3: `seismic__2026.56591748__synthoseis_run_1200/model_data.zarr`; root 3: `seismic__2026.56595177__synthoseis_run_1201/model_data.zarr`; root 3: `seismic__2026.56599982__synthoseis_run_1202/model_data.zarr`; root 3: `seismic__2026.56605050__synthoseis_run_1203/model_data.zarr`; root 3: `seismic__2026.56613975__synthoseis_run_1204/model_data.zarr`; root 3: `seismic__2026.56619645__synthoseis_run_1205/model_data.zarr`; root 3: `seismic__2026.56626453__synthoseis_run_1206/model_data.zarr`; root 3: `seismic__2026.56630485__synthoseis_run_1207/model_data.zarr`; root 3: `seismic__2026.56635043__synthoseis_run_1208/model_data.zarr`; root 3: `seismic__2026.56640519__synthoseis_run_1209/model_data.zarr`; root 3: `seismic__2026.56644802__synthoseis_run_1210/model_data.zarr`; root 3: `seismic__2026.56650130__synthoseis_run_1211/model_data.zarr`; root 3: `seismic__2026.56657048__synthoseis_run_1212/model_data.zarr`; root 3: `seismic__2026.56667915__synthoseis_run_1213/model_data.zarr`; root 3: `seismic__2026.56674553__synthoseis_run_1214/model_data.zarr`; root 3: `seismic__2026.56679515__synthoseis_run_1215/model_data.zarr`; root 3: `seismic__2026.56686066__synthoseis_run_1216/model_data.zarr`; root 3: `seismic__2026.56690147__synthoseis_run_1217/model_data.zarr`; root 3: `seismic__2026.56695710__synthoseis_run_1218/model_data.zarr`; root 3: `seismic__2026.56700464__synthoseis_run_1219/model_data.zarr`; root 3: `seismic__2026.56707148__synthoseis_run_1220/model_data.zarr`; root 3: `validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr`; root 3: `validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr`; root 3: `validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr`; root 3: `validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr`; root 3: `validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr`; root 3: `validation/seismic__2026.56713658__synthoseis_run_1221/model_data.zarr`; root 3: `validation/seismic__2026.56719363__synthoseis_run_1222/model_data.zarr`; root 3: `validation/seismic__2026.56725661__synthoseis_run_1223/model_data.zarr`; root 3: `validation/seismic__2026.56733317__synthoseis_run_1224/model_data.zarr`; root 3: `validation/seismic__2026.56738951__synthoseis_run_1225/model_data.zarr`; root 3: `validation/seismic__2026.56745992__synthoseis_run_1226/model_data.zarr`; root 3: `validation/seismic__2026.56749584__synthoseis_run_1227/model_data.zarr`; root 3: `validation/seismic__2026.56754446__synthoseis_run_1228/model_data.zarr`; root 3: `validation/seismic__2026.56758041__synthoseis_run_1229/model_data.zarr`; root 3: `validation/seismic__2026.56761906__synthoseis_run_1230/model_data.zarr`; root 3: `validation/seismic__2026.56767780__synthoseis_run_1231/model_data.zarr`; root 3: `validation/seismic__2026.56773490__synthoseis_run_1232/model_data.zarr`; root 3: `validation/seismic__2026.56779472__synthoseis_run_1233/model_data.zarr`; root 3: `validation/seismic__2026.56785579__synthoseis_run_1234/model_data.zarr`; root 3: `validation/seismic__2026.56792013__synthoseis_run_1235/model_data.zarr`; root 3: `validation/seismic__2026.56798438__synthoseis_run_1236/model_data.zarr`; root 3: `validation/seismic__2026.56804549__synthoseis_run_1237/model_data.zarr`; root 3: `validation/seismic__2026.56809936__synthoseis_run_1238/model_data.zarr`; root 3: `validation/seismic__2026.56815746__synthoseis_run_1239/model_data.zarr`; root 3: `validation/seismic__2026.56822189__synthoseis_run_1040/model_data.zarr`; root 5: `seismic__2026.55209454__synthoseis_run_1230/model_data.zarr`; root 5: `seismic__2026.55215129__synthoseis_run_1231/model_data.zarr`; root 5: `seismic__2026.55220631__synthoseis_run_1232/model_data.zarr`; root 5: `seismic__2026.55226235__synthoseis_run_1233/model_data.zarr`; root 5: `seismic__2026.55232050__synthoseis_run_1234/model_data.zarr`; root 5: `seismic__2026.55238294__synthoseis_run_1235/model_data.zarr`; root 5: `seismic__2026.55244465__synthoseis_run_1236/model_data.zarr`; root 5: `seismic__2026.55250314__synthoseis_run_1237/model_data.zarr`; root 5: `seismic__2026.55255344__synthoseis_run_1238/model_data.zarr`; root 5: `seismic__2026.55260908__synthoseis_run_1239/model_data.zarr`; root 5: `validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr`; root 5: `validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr`; root 5: `validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr`; root 5: `validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr`; root 5: `validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr`
- **seismic__2026.41133304__synthoseis_run_0570** (name only): root 3: `validation/seismic__2026.41133304__synthoseis_run_0570`; root 5: `validation/seismic__2026.41133304__synthoseis_run_0570`
- **seismic__2026.41138579__synthoseis_run_0571** (name only): root 3: `validation/seismic__2026.41138579__synthoseis_run_0571`; root 5: `validation/seismic__2026.41138579__synthoseis_run_0571`
- **seismic__2026.41143924__synthoseis_run_0572** (name only): root 3: `validation/seismic__2026.41143924__synthoseis_run_0572`; root 5: `validation/seismic__2026.41143924__synthoseis_run_0572`
- **seismic__2026.41151562__synthoseis_run_0573** (name only): root 3: `validation/seismic__2026.41151562__synthoseis_run_0573`; root 5: `validation/seismic__2026.41151562__synthoseis_run_0573`
- **seismic__2026.41158562__synthoseis_run_0574** (name only): root 3: `validation/seismic__2026.41158562__synthoseis_run_0574`; root 5: `validation/seismic__2026.41158562__synthoseis_run_0574`
- **test** (name only): root 3: `test`; root 5: `test`
- **validation** (name only): root 3: `validation`; root 5: `validation`

## Repeated Zarr Names

### CostaRica_part1.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 3: `test/CostaRica_part1.real.zarr`
- Root 5: `test/CostaRica_part1.real.zarr`
### Netherlands1.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 2: `Netherlands1.real.zarr`
- Root 4: `Netherlands1.real.zarr`
### Netherlands2.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 2: `Netherlands2.real.zarr`
- Root 4: `Netherlands2.real.zarr`
### Netherlands3.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 2: `Netherlands3.real.zarr`
- Root 4: `Netherlands3.real.zarr`
### Netherlands4.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 2: `Netherlands4.real.zarr`
- Root 4: `Netherlands4.real.zarr`
### Poseidon_part1.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 3: `test/Poseidon_part1.real.zarr`
- Root 5: `test/Poseidon_part1.real.zarr`
### Poseidon_part2.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 3: `test/Poseidon_part2.real.zarr`
- Root 5: `test/Poseidon_part2.real.zarr`
### cn1.real.zarr

2 stores share this basename. This does not establish identical contents.

- Root 3: `test/cn1.real.zarr`
- Root 5: `test/cn1.real.zarr`
### model_data.zarr

220 stores share this basename. This does not establish identical contents.

- Root 3: `seismic__2026.55615090__synthoseis_run_1041/model_data.zarr`
- Root 3: `seismic__2026.55619470__synthoseis_run_1042/model_data.zarr`
- Root 3: `seismic__2026.55623385__synthoseis_run_1043/model_data.zarr`
- Root 3: `seismic__2026.55628342__synthoseis_run_1044/model_data.zarr`
- Root 3: `seismic__2026.55634773__synthoseis_run_1045/model_data.zarr`
- Root 3: `seismic__2026.55638871__synthoseis_run_1046/model_data.zarr`
- Root 3: `seismic__2026.55645882__synthoseis_run_1047/model_data.zarr`
- Root 3: `seismic__2026.55650934__synthoseis_run_1048/model_data.zarr`
- Root 3: `seismic__2026.55659211__synthoseis_run_1049/model_data.zarr`
- Root 3: `seismic__2026.55667424__synthoseis_run_1050/model_data.zarr`
- Root 3: `seismic__2026.55677408__synthoseis_run_1051/model_data.zarr`
- Root 3: `seismic__2026.55682963__synthoseis_run_1052/model_data.zarr`
- Root 3: `seismic__2026.55688100__synthoseis_run_1053/model_data.zarr`
- Root 3: `seismic__2026.55696664__synthoseis_run_1054/model_data.zarr`
- Root 3: `seismic__2026.55702955__synthoseis_run_1055/model_data.zarr`
- Root 3: `seismic__2026.55707149__synthoseis_run_1056/model_data.zarr`
- Root 3: `seismic__2026.55714288__synthoseis_run_1057/model_data.zarr`
- Root 3: `seismic__2026.55720365__synthoseis_run_1058/model_data.zarr`
- Root 3: `seismic__2026.55728007__synthoseis_run_1059/model_data.zarr`
- Root 3: `seismic__2026.55734525__synthoseis_run_1060/model_data.zarr`
- Root 3: `seismic__2026.55740899__synthoseis_run_1061/model_data.zarr`
- Root 3: `seismic__2026.55747692__synthoseis_run_1062/model_data.zarr`
- Root 3: `seismic__2026.55752927__synthoseis_run_1063/model_data.zarr`
- Root 3: `seismic__2026.55759561__synthoseis_run_1064/model_data.zarr`
- Root 3: `seismic__2026.55764208__synthoseis_run_1065/model_data.zarr`
- Root 3: `seismic__2026.55769983__synthoseis_run_1066/model_data.zarr`
- Root 3: `seismic__2026.55774107__synthoseis_run_1067/model_data.zarr`
- Root 3: `seismic__2026.55779024__synthoseis_run_1068/model_data.zarr`
- Root 3: `seismic__2026.55785831__synthoseis_run_1069/model_data.zarr`
- Root 3: `seismic__2026.55792481__synthoseis_run_1070/model_data.zarr`
- Root 3: `seismic__2026.55799982__synthoseis_run_1071/model_data.zarr`
- Root 3: `seismic__2026.55806229__synthoseis_run_1072/model_data.zarr`
- Root 3: `seismic__2026.55813827__synthoseis_run_1073/model_data.zarr`
- Root 3: `seismic__2026.55818117__synthoseis_run_1074/model_data.zarr`
- Root 3: `seismic__2026.55826154__synthoseis_run_1075/model_data.zarr`
- Root 3: `seismic__2026.55832797__synthoseis_run_1076/model_data.zarr`
- Root 3: `seismic__2026.55837522__synthoseis_run_1077/model_data.zarr`
- Root 3: `seismic__2026.55843143__synthoseis_run_1078/model_data.zarr`
- Root 3: `seismic__2026.55849882__synthoseis_run_1079/model_data.zarr`
- Root 3: `seismic__2026.55856356__synthoseis_run_1080/model_data.zarr`
- Root 3: `seismic__2026.55861365__synthoseis_run_1081/model_data.zarr`
- Root 3: `seismic__2026.55865636__synthoseis_run_1082/model_data.zarr`
- Root 3: `seismic__2026.55869867__synthoseis_run_1083/model_data.zarr`
- Root 3: `seismic__2026.55876744__synthoseis_run_1084/model_data.zarr`
- Root 3: `seismic__2026.55883755__synthoseis_run_1085/model_data.zarr`
- Root 3: `seismic__2026.55887811__synthoseis_run_1086/model_data.zarr`
- Root 3: `seismic__2026.55894225__synthoseis_run_1087/model_data.zarr`
- Root 3: `seismic__2026.55900440__synthoseis_run_1088/model_data.zarr`
- Root 3: `seismic__2026.55906851__synthoseis_run_1089/model_data.zarr`
- Root 3: `seismic__2026.55915960__synthoseis_run_1090/model_data.zarr`
- Root 3: `seismic__2026.55925234__synthoseis_run_1091/model_data.zarr`
- Root 3: `seismic__2026.55933127__synthoseis_run_1092/model_data.zarr`
- Root 3: `seismic__2026.55938287__synthoseis_run_1093/model_data.zarr`
- Root 3: `seismic__2026.55943336__synthoseis_run_1094/model_data.zarr`
- Root 3: `seismic__2026.55949285__synthoseis_run_1095/model_data.zarr`
- Root 3: `seismic__2026.55954249__synthoseis_run_1096/model_data.zarr`
- Root 3: `seismic__2026.55959629__synthoseis_run_1097/model_data.zarr`
- Root 3: `seismic__2026.55970803__synthoseis_run_1098/model_data.zarr`
- Root 3: `seismic__2026.55975550__synthoseis_run_1099/model_data.zarr`
- Root 3: `seismic__2026.55980563__synthoseis_run_1100/model_data.zarr`
- Root 3: `seismic__2026.55985718__synthoseis_run_1101/model_data.zarr`
- Root 3: `seismic__2026.55990770__synthoseis_run_1102/model_data.zarr`
- Root 3: `seismic__2026.55995415__synthoseis_run_1103/model_data.zarr`
- Root 3: `seismic__2026.56000028__synthoseis_run_1104/model_data.zarr`
- Root 3: `seismic__2026.56009525__synthoseis_run_1105/model_data.zarr`
- Root 3: `seismic__2026.56013592__synthoseis_run_1106/model_data.zarr`
- Root 3: `seismic__2026.56019860__synthoseis_run_1107/model_data.zarr`
- Root 3: `seismic__2026.56025395__synthoseis_run_1108/model_data.zarr`
- Root 3: `seismic__2026.56031354__synthoseis_run_1109/model_data.zarr`
- Root 3: `seismic__2026.56038121__synthoseis_run_1110/model_data.zarr`
- Root 3: `seismic__2026.56044585__synthoseis_run_1111/model_data.zarr`
- Root 3: `seismic__2026.56049817__synthoseis_run_1112/model_data.zarr`
- Root 3: `seismic__2026.56055905__synthoseis_run_1113/model_data.zarr`
- Root 3: `seismic__2026.56061132__synthoseis_run_1114/model_data.zarr`
- Root 3: `seismic__2026.56064862__synthoseis_run_1115/model_data.zarr`
- Root 3: `seismic__2026.56071894__synthoseis_run_1116/model_data.zarr`
- Root 3: `seismic__2026.56077949__synthoseis_run_1117/model_data.zarr`
- Root 3: `seismic__2026.56083437__synthoseis_run_1118/model_data.zarr`
- Root 3: `seismic__2026.56089675__synthoseis_run_1119/model_data.zarr`
- Root 3: `seismic__2026.56096908__synthoseis_run_1120/model_data.zarr`
- Root 3: `seismic__2026.56104435__synthoseis_run_1121/model_data.zarr`
- Root 3: `seismic__2026.56113120__synthoseis_run_1122/model_data.zarr`
- Root 3: `seismic__2026.56119063__synthoseis_run_1123/model_data.zarr`
- Root 3: `seismic__2026.56126576__synthoseis_run_1124/model_data.zarr`
- Root 3: `seismic__2026.56132292__synthoseis_run_1125/model_data.zarr`
- Root 3: `seismic__2026.56136194__synthoseis_run_1126/model_data.zarr`
- Root 3: `seismic__2026.56142556__synthoseis_run_1127/model_data.zarr`
- Root 3: `seismic__2026.56151509__synthoseis_run_1128/model_data.zarr`
- Root 3: `seismic__2026.56157445__synthoseis_run_1129/model_data.zarr`
- Root 3: `seismic__2026.56160841__synthoseis_run_1130/model_data.zarr`
- Root 3: `seismic__2026.56168074__synthoseis_run_1131/model_data.zarr`
- Root 3: `seismic__2026.56174639__synthoseis_run_1132/model_data.zarr`
- Root 3: `seismic__2026.56179768__synthoseis_run_1133/model_data.zarr`
- Root 3: `seismic__2026.56186239__synthoseis_run_1134/model_data.zarr`
- Root 3: `seismic__2026.56192225__synthoseis_run_1135/model_data.zarr`
- Root 3: `seismic__2026.56198621__synthoseis_run_1136/model_data.zarr`
- Root 3: `seismic__2026.56205358__synthoseis_run_1137/model_data.zarr`
- Root 3: `seismic__2026.56213927__synthoseis_run_1138/model_data.zarr`
- Root 3: `seismic__2026.56220514__synthoseis_run_1139/model_data.zarr`
- Root 3: `seismic__2026.56228494__synthoseis_run_1140/model_data.zarr`
- Root 3: `seismic__2026.56233920__synthoseis_run_1141/model_data.zarr`
- Root 3: `seismic__2026.56239876__synthoseis_run_1142/model_data.zarr`
- Root 3: `seismic__2026.56245558__synthoseis_run_1143/model_data.zarr`
- Root 3: `seismic__2026.56252250__synthoseis_run_1144/model_data.zarr`
- Root 3: `seismic__2026.56258253__synthoseis_run_1145/model_data.zarr`
- Root 3: `seismic__2026.56261740__synthoseis_run_1146/model_data.zarr`
- Root 3: `seismic__2026.56267034__synthoseis_run_1147/model_data.zarr`
- Root 3: `seismic__2026.56273085__synthoseis_run_1148/model_data.zarr`
- Root 3: `seismic__2026.56280144__synthoseis_run_1149/model_data.zarr`
- Root 3: `seismic__2026.56285784__synthoseis_run_1150/model_data.zarr`
- Root 3: `seismic__2026.56291577__synthoseis_run_1151/model_data.zarr`
- Root 3: `seismic__2026.56298204__synthoseis_run_1152/model_data.zarr`
- Root 3: `seismic__2026.56303059__synthoseis_run_1153/model_data.zarr`
- Root 3: `seismic__2026.56307499__synthoseis_run_1154/model_data.zarr`
- Root 3: `seismic__2026.56312567__synthoseis_run_1155/model_data.zarr`
- Root 3: `seismic__2026.56317572__synthoseis_run_1156/model_data.zarr`
- Root 3: `seismic__2026.56321877__synthoseis_run_1157/model_data.zarr`
- Root 3: `seismic__2026.56329232__synthoseis_run_1158/model_data.zarr`
- Root 3: `seismic__2026.56336776__synthoseis_run_1159/model_data.zarr`
- Root 3: `seismic__2026.56343799__synthoseis_run_1160/model_data.zarr`
- Root 3: `seismic__2026.56350941__synthoseis_run_1161/model_data.zarr`
- Root 3: `seismic__2026.56358337__synthoseis_run_1162/model_data.zarr`
- Root 3: `seismic__2026.56365415__synthoseis_run_1163/model_data.zarr`
- Root 3: `seismic__2026.56371809__synthoseis_run_1164/model_data.zarr`
- Root 3: `seismic__2026.56378600__synthoseis_run_1165/model_data.zarr`
- Root 3: `seismic__2026.56384982__synthoseis_run_1166/model_data.zarr`
- Root 3: `seismic__2026.56394820__synthoseis_run_1167/model_data.zarr`
- Root 3: `seismic__2026.56400145__synthoseis_run_1168/model_data.zarr`
- Root 3: `seismic__2026.56407506__synthoseis_run_1169/model_data.zarr`
- Root 3: `seismic__2026.56419037__synthoseis_run_1170/model_data.zarr`
- Root 3: `seismic__2026.56424623__synthoseis_run_1171/model_data.zarr`
- Root 3: `seismic__2026.56431068__synthoseis_run_1172/model_data.zarr`
- Root 3: `seismic__2026.56436181__synthoseis_run_1173/model_data.zarr`
- Root 3: `seismic__2026.56439991__synthoseis_run_1174/model_data.zarr`
- Root 3: `seismic__2026.56446329__synthoseis_run_1175/model_data.zarr`
- Root 3: `seismic__2026.56451385__synthoseis_run_1176/model_data.zarr`
- Root 3: `seismic__2026.56457514__synthoseis_run_1177/model_data.zarr`
- Root 3: `seismic__2026.56464690__synthoseis_run_1178/model_data.zarr`
- Root 3: `seismic__2026.56469525__synthoseis_run_1179/model_data.zarr`
- Root 3: `seismic__2026.56475284__synthoseis_run_1180/model_data.zarr`
- Root 3: `seismic__2026.56480261__synthoseis_run_1181/model_data.zarr`
- Root 3: `seismic__2026.56486862__synthoseis_run_1182/model_data.zarr`
- Root 3: `seismic__2026.56493954__synthoseis_run_1183/model_data.zarr`
- Root 3: `seismic__2026.56498627__synthoseis_run_1184/model_data.zarr`
- Root 3: `seismic__2026.56504435__synthoseis_run_1185/model_data.zarr`
- Root 3: `seismic__2026.56510121__synthoseis_run_1186/model_data.zarr`
- Root 3: `seismic__2026.56515746__synthoseis_run_1187/model_data.zarr`
- Root 3: `seismic__2026.56522588__synthoseis_run_1188/model_data.zarr`
- Root 3: `seismic__2026.56528962__synthoseis_run_1189/model_data.zarr`
- Root 3: `seismic__2026.56534062__synthoseis_run_1190/model_data.zarr`
- Root 3: `seismic__2026.56538485__synthoseis_run_1191/model_data.zarr`
- Root 3: `seismic__2026.56542936__synthoseis_run_1192/model_data.zarr`
- Root 3: `seismic__2026.56551113__synthoseis_run_1193/model_data.zarr`
- Root 3: `seismic__2026.56557181__synthoseis_run_1194/model_data.zarr`
- Root 3: `seismic__2026.56562134__synthoseis_run_1195/model_data.zarr`
- Root 3: `seismic__2026.56569270__synthoseis_run_1196/model_data.zarr`
- Root 3: `seismic__2026.56573275__synthoseis_run_1197/model_data.zarr`
- Root 3: `seismic__2026.56578709__synthoseis_run_1198/model_data.zarr`
- Root 3: `seismic__2026.56584647__synthoseis_run_1199/model_data.zarr`
- Root 3: `seismic__2026.56591748__synthoseis_run_1200/model_data.zarr`
- Root 3: `seismic__2026.56595177__synthoseis_run_1201/model_data.zarr`
- Root 3: `seismic__2026.56599982__synthoseis_run_1202/model_data.zarr`
- Root 3: `seismic__2026.56605050__synthoseis_run_1203/model_data.zarr`
- Root 3: `seismic__2026.56613975__synthoseis_run_1204/model_data.zarr`
- Root 3: `seismic__2026.56619645__synthoseis_run_1205/model_data.zarr`
- Root 3: `seismic__2026.56626453__synthoseis_run_1206/model_data.zarr`
- Root 3: `seismic__2026.56630485__synthoseis_run_1207/model_data.zarr`
- Root 3: `seismic__2026.56635043__synthoseis_run_1208/model_data.zarr`
- Root 3: `seismic__2026.56640519__synthoseis_run_1209/model_data.zarr`
- Root 3: `seismic__2026.56644802__synthoseis_run_1210/model_data.zarr`
- Root 3: `seismic__2026.56650130__synthoseis_run_1211/model_data.zarr`
- Root 3: `seismic__2026.56657048__synthoseis_run_1212/model_data.zarr`
- Root 3: `seismic__2026.56667915__synthoseis_run_1213/model_data.zarr`
- Root 3: `seismic__2026.56674553__synthoseis_run_1214/model_data.zarr`
- Root 3: `seismic__2026.56679515__synthoseis_run_1215/model_data.zarr`
- Root 3: `seismic__2026.56686066__synthoseis_run_1216/model_data.zarr`
- Root 3: `seismic__2026.56690147__synthoseis_run_1217/model_data.zarr`
- Root 3: `seismic__2026.56695710__synthoseis_run_1218/model_data.zarr`
- Root 3: `seismic__2026.56700464__synthoseis_run_1219/model_data.zarr`
- Root 3: `seismic__2026.56707148__synthoseis_run_1220/model_data.zarr`
- Root 3: `validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr`
- Root 3: `validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr`
- Root 3: `validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr`
- Root 3: `validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr`
- Root 3: `validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr`
- Root 3: `validation/seismic__2026.56713658__synthoseis_run_1221/model_data.zarr`
- Root 3: `validation/seismic__2026.56719363__synthoseis_run_1222/model_data.zarr`
- Root 3: `validation/seismic__2026.56725661__synthoseis_run_1223/model_data.zarr`
- Root 3: `validation/seismic__2026.56733317__synthoseis_run_1224/model_data.zarr`
- Root 3: `validation/seismic__2026.56738951__synthoseis_run_1225/model_data.zarr`
- Root 3: `validation/seismic__2026.56745992__synthoseis_run_1226/model_data.zarr`
- Root 3: `validation/seismic__2026.56749584__synthoseis_run_1227/model_data.zarr`
- Root 3: `validation/seismic__2026.56754446__synthoseis_run_1228/model_data.zarr`
- Root 3: `validation/seismic__2026.56758041__synthoseis_run_1229/model_data.zarr`
- Root 3: `validation/seismic__2026.56761906__synthoseis_run_1230/model_data.zarr`
- Root 3: `validation/seismic__2026.56767780__synthoseis_run_1231/model_data.zarr`
- Root 3: `validation/seismic__2026.56773490__synthoseis_run_1232/model_data.zarr`
- Root 3: `validation/seismic__2026.56779472__synthoseis_run_1233/model_data.zarr`
- Root 3: `validation/seismic__2026.56785579__synthoseis_run_1234/model_data.zarr`
- Root 3: `validation/seismic__2026.56792013__synthoseis_run_1235/model_data.zarr`
- Root 3: `validation/seismic__2026.56798438__synthoseis_run_1236/model_data.zarr`
- Root 3: `validation/seismic__2026.56804549__synthoseis_run_1237/model_data.zarr`
- Root 3: `validation/seismic__2026.56809936__synthoseis_run_1238/model_data.zarr`
- Root 3: `validation/seismic__2026.56815746__synthoseis_run_1239/model_data.zarr`
- Root 3: `validation/seismic__2026.56822189__synthoseis_run_1040/model_data.zarr`
- Root 5: `seismic__2026.55209454__synthoseis_run_1230/model_data.zarr`
- Root 5: `seismic__2026.55215129__synthoseis_run_1231/model_data.zarr`
- Root 5: `seismic__2026.55220631__synthoseis_run_1232/model_data.zarr`
- Root 5: `seismic__2026.55226235__synthoseis_run_1233/model_data.zarr`
- Root 5: `seismic__2026.55232050__synthoseis_run_1234/model_data.zarr`
- Root 5: `seismic__2026.55238294__synthoseis_run_1235/model_data.zarr`
- Root 5: `seismic__2026.55244465__synthoseis_run_1236/model_data.zarr`
- Root 5: `seismic__2026.55250314__synthoseis_run_1237/model_data.zarr`
- Root 5: `seismic__2026.55255344__synthoseis_run_1238/model_data.zarr`
- Root 5: `seismic__2026.55260908__synthoseis_run_1239/model_data.zarr`
- Root 5: `validation/seismic__2026.41133304__synthoseis_run_0570/model_data.zarr`
- Root 5: `validation/seismic__2026.41138579__synthoseis_run_0571/model_data.zarr`
- Root 5: `validation/seismic__2026.41143924__synthoseis_run_0572/model_data.zarr`
- Root 5: `validation/seismic__2026.41151562__synthoseis_run_0573/model_data.zarr`
- Root 5: `validation/seismic__2026.41158562__synthoseis_run_0574/model_data.zarr`

Content verification: 13 matching-layout candidate groups; 13 confirmed duplicate groups.
