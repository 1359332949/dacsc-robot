# libero_10 in-suite calibration validation (Phase 2A follow-up)

## Hypothesis

V1 with Spatial-only calibration loses 5 Clean games on libero_10 because
`libero_10` contains new rooms (living room, study, kitchen variants) while the
Spatial calibration saw only kitchen-table scenes. The tight `tau_b3` then
misreads a clean new room as a shift.

## Code fixes applied before this run

1. `phase2_suite_eval.py` language now always uses `c['base_task'] + '.bddl'`;
   Light/Background trial JSON records the real `sub_group` (no index guessing).
2. `analyze_phase2.py` reads `sub_group` from the trial JSON and no longer
   double-multiplies McNemar p-values.
3. `docs/data/phase15/spatial_shared_calib.npz` synced to the corrected
   `[0,1]` pixel version.
4. New `libero10_init0_calib.npz` from 10 Clean init0 trajectories; test uses
   init_index 1..10 (strict isolation).

## Clean activation diagnostic (100 Clean conditions, one t=0 forward)

| Calibration | Mean active channel rate |
|---|---:|
| Spatial-only (cross-room) | **51.99%** |
| libero_10 init0 in-suite | **3.61%** |

This is direct evidence: the Spatial calibration fires on ~52% of channels for
clean new rooms; in-suite calibration fires on ~3.6%.

## Clean and Lighting/Background, fixed-language 2-arm rerun

300 conditions x 2 arms = 600 rollouts.

| Group | Frozen | V1 Spatial calib | V1 in-suite init0 |
|---|---:|---:|---:|
| Clean | 95/100 | 90/100 | **98/100** |
| Lighting | 54/100 | — (old run contaminated) | **59/100** |
| Background | 28/100 | — | 28/100 |
| – Table | 28/100 | — | 28/100 |
| – Scene | 0/0 (no scene BDDL exists) | — | 0/0 |

V1 in-suite Clean: 98/100 vs Frozen 95 (+3, McNemar p=.375), and +8 over
V1-Spatial 90/100. No Clean regression on this rerun.

Lighting: 59/100 vs Frozen 54 (+5, p=.227).

Background: exactly flat 28/100. **Important:** libero_10's official background
pool contains only table-texture variants (28 per task, 0 scene variants); the
earlier 50/50 table/scene split was a fake index-based split and is retracted.

## Conclusion

- The cross-room calibration hypothesis is supported: Clean activation drops
  from 52% to 3.6%, and Clean success recovers 90 -> 98.
- In-suite calibration removes the Clean regression and gives +5 Lighting.
- Background is flat; no method signal.
- Old Phase2A Lighting/Background 5-arm numbers are marked invalid/superseded
  because the repo script and run script were not identical and the background
  table/scene split was fabricated.
- The remaining honest claim: V1 is **not** universally zero-shot across rooms;
  it needs per-suite isolated clean calibration. With that, it is Clean-safe on
  libero_10, but total-suite gains are still mixed.

## Files

- `docs/data/phase2/libero10_insuite_v1.json`
- `scripts/phase2_insuite_fix.py`
- `scripts/activation_clean.py`
- `scripts/build_libero10_init0_calib.py`
