# TTA Evaluation Protocol (frozen 2026-11-17)

This document freezes the evaluation protocol for the target-only visual TTA
study on frozen CoRP. It exists because earlier experiments mixed source/target
domains, target development, and target test, which invalidated comparisons.

## 1. Terms

- **suite**: a LIBERO task collection (`libero_spatial`, `libero_object`,
  `libero_goal`, `libero_10`).
- **task**: one manipulation instruction inside a suite.
- **domain**: a visual condition for the same task (`Clean`, `Lighting`,
  `Background`, `Camera`, `Sensor`).
- **initial state**: one 50-way starting scene configuration. This is the main
  sampling unit, not a visual style.
- **seed**: repeat execution of one initial state. It controls flow/environment
  randomness.
- **episode**: one `(initial state, seed)` execution.

## 2. TTA taxonomy used here

Unsupervised VLM adaptation survey (arXiv:2508.05547) separates:

1. Data-Free Transfer (no target data);
2. Unsupervised Domain Transfer (abundant unlabeled target data);
3. Episodic Test-Time Adaptation (batch/episode data);
4. Online Test-Time Adaptation (streaming data).

Online TTA survey (arXiv:2310.20199) warns that ambiguous settings,
inconsistent hyperparameter tuning, and target-stream order make results
non-comparable.

**This project is Episodic TTA**: the adapter is reset at each episode and may
use only that episode's unlabeled observations. It is not DG and not continual
online TTA.

## 3. Frozen pilot split

Suite: `libero_spatial`
Task: `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
Target domain: `lighting_2292` (`light_1`)

Source train (Clean, source calibration):

```
0,1,2,4,6,8,10,13,14,17
```

Source val (Clean, source calibration):

```
18,20,21,22,25,27,29,31,32,33
```

Target dev (Lighting, method development only):

```
34,35,36,37,39,43,44,45,48,49
```

Target test (Lighting, evaluation only):

```
3,5,7,9,11,12,15,16,19,23,24,26,28,30,38,40,41,42,46,47
```

All four state sets are disjoint. The canonical JSON is
`docs/data/tta_split_libero_spatial.json`.

## 4. Information budget

Allowed:

- source Clean features/states for source calibration;
- target observations, proprioception, language, and action history at test;
- target dev success labels for **method development only**.

Forbidden:

- target Clean images as adapter input;
- target test success/reward/perturbation ID for adaptation;
- target test results for method selection or threshold tuning;
- paired Clean features as deployment inputs (offline diagnostics only).

## 5. Evaluation rules

- Episode is the statistical unit; seeds are repeated measures within state.
- Report Clean and Shift separately.
- Report all-heldout target test, not only easy states.
- Report paired discordant success counts (gain/loss) and exact binomial test.
- Use cluster bootstrap over initial states for action metrics.
- Never compare runs with different `executed_k`, precision, sampler, or
  concurrency mode as if they were the same experiment.

## 6. Code entry points

- `scripts/closed_loop_tta_pilot.py`:
  - `--tta-mode episodic`
  - `--suite libero_spatial`
  - `--lighting-id 2292`
  - `--executed-k 12`
  - `--precision bf16` for official `policy.predict` reference;
    `--precision fp32` when comparing `sample_z` methods.
- Frozen split JSON: `docs/data/tta_split_libero_spatial.json`.
- Source capture: `scripts/build_source_capture.py`.

## 7. Provenance

- matrix SHA256: `d51806a933d82ac5d78e7e9ab310cbe7a888fa6ef49ceace0438df95052db205`
- checkpoint SHA256: `f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6`
- source capture script: `scripts/build_source_capture.py`
- split file: `docs/data/tta_split_libero_spatial.json`

## 8. Leakage status

The 20 spatial target-test states have been used once for exploratory
Frozen/First/Online comparisons. They are therefore **not confirmatory test
states anymore**. Any final claim must use a new target domain or a new
suite/task that has not been inspected.

The retired `libero_object` heldout set is development-only. The attempted
`libero_goal` split failed Clean validation (0/20) and is not usable.

## 9. Approval gate

Do not run 20-episode confirmatory experiments until:

1. a fresh target test set is frozen;
2. Frozen Clean/Shift is validated on it;
3. only target dev is used for method selection;
4. target test is evaluated once for the selected method.

## 10. Enforcement (P0 fixes, 2026-11-17)

The split is no longer advisory. The runner must be invoked with
`--split-json` and `--role` for formal runs:

```
--split-json docs/data/tta_split_libero_spatial.json
--role target_dev|target_test|source_fit|source_validation
```

The runner:

- resolves the state list from the split role, not from `--init-indices`;
- rejects states outside the requested role unless `--allow-role-override`
  is passed (diagnostic only, recorded as such);
- fits source calibration only on `source_fit_state_ids`, never on
  `source_validation_state_ids`;
- uses `evaluation_seeds` from the split JSON as actual episode seeds;
- records `seed`, `split_role`, `split_sha256`, and `z0_schedule_sha256`
  in every record and in the run payload;
- computes hashes for checkpoint, matrix, split, BDDL, initial-state file,
  capture manifest, and capture features; `input_files_exist` is only a
  file-presence flag, not provenance.

### Inference-path rule

Frozen and First must use the same inference path in the main paired
comparison: `sample_z` with the same normalization, precision, and cached
`z0` schedule. `frozen_predict` is a separate official-reference check and
must not be mixed into the First-vs-Frozen pair.

### Clean retention

For every target test state/seed, run both:

- target Shift episode: adapter may use Shift observations only;
- Clean twin episode in a separate environment.

The Clean twin is evaluation-only. Clean images are never passed to the
Shift adapter.

### Target-test status

The 20 spatial target-test states have been used for exploratory
Frozen/First/Online comparisons. They are not confirmatory. A final claim
requires a fresh perturbation configuration or a fresh task/suite.

### Statistics

Seeds are repeated measures within an initial state. Report paired
state-level results and cluster bootstrap over initial states; do not treat
`20 states x 3 seeds = 60` as 60 independent scenes.

## 11. Action/failure diagnostic (target dev, 2026-11-17)

Frozen Clean vs Shift on the 10 target-dev states, seed 2014:

- Clean: 9/10 success;
- Shift: 4/10 success;
- Shift failures: 5 `no_lift`, 1 `lift_failed`;
- Paired action L2: 0.28–0.57 in successes, 0.64–1.17 in most failures;
- Action cosine: 0.78–0.95 in successes, 0.30–0.46 in several failures;
- Largest mean absolute action difference is the gripper dimension (0.399),
  followed by eef x/y/z (0.18–0.25).

Interpretation: under this Lighting shift the frozen policy often diverges
before lift/grasp completion, and its action trajectory changes strongly.
A single global visual-mean correction is not aligned with this state- and
phase-dependent action change. This is evidence against further global
visual-statistics adaptation, not evidence for a new adapter.

The diagnostic is exploratory: one seed, 10 states, one Lighting ID.

## 12. Execution-horizon diagnostic (target dev, 2026-11-17)

Frozen, same 10 target-dev states, seed 2014:

| E | Clean | Shift |
|---|---:|---:|
| 6 | 6/10 | 4/10 |
| 12 | 9/10 | 5/10 |

Shorter chunking is worse on both conditions in this pilot. There is no
evidence here that the Lighting failure is caused by an overly long
execution horizon; E=12 remains the frozen execution setting. This is a
single seed, 10 states, one Lighting ID, so it is diagnostic only.

## 13. Privileged action oracle (target dev, seed 2014)

Frozen Shift baseline: 5/10.

- Paired visual-gap correction (`paired_first`): 2/10.
- Clean-action oracle (query Frozen on the paired Clean twin at every chunk,
  execute those actions in the Shift environment): 8/10.

The clean-action oracle is **privileged and not deployable**. It is used only
to ask whether an action-level correction has headroom. Result: the same
Lighting shift that defeats global visual-mean correction is largely
recoverable when the action interface receives clean-view actions. This
supports action-level adaptation as a research direction; it does not provide
a deployable target-only signal.

First-chunk-only action oracle: **9/10**. Only chunk 0 is queried on the
Clean twin; all later chunks use the Shift policy. This shows that the
critical action error is decided in the first chunk, not spread over the
whole episode.

First-chunk action differences are concentrated in end-effector x/y
(mean |delta| ~0.08-0.16); gripper dims are nearly unchanged in chunk 0.
A deployable method would need to predict this early translation correction
without access to the Clean twin.

## 14. Source-paired first-chunk residual (target dev, seed 2014)

Source paired first chunks were collected on source fit+val states (20 states).
A kNN over shift state8 + pooled visual retrieves the source Clean-Shift action
delta; it is added to the target Shift first chunk, then the unmodified Shift
policy continues.

| method | target-dev Shift |
|---|---:|
| Frozen | 5/10 |
| kNN k=1, alpha=1 | 6/10 |
| kNN k=3, alpha=1 | 5/10 |
| kNN k=1, alpha=0.5 | 3/10 |
| Clean-action oracle | 9/10 |

A weak +1 gain is not evidence of a working method. Source-state retrieval does
not predict the target first-chunk correction well enough even though the
privileged action oracle has large headroom. This rules out the simplest
source-paired-kNN version, not the whole action-correction hypothesis.

## 15. First-chunk residual identifiability audit (offline, 2026-11-17)

Pairing sanity: source and target Clean/Shift state8 max abs diff = 0.0.
Only 8D proprio was checked; full MuJoCo/camera pairing is not yet audited.

Source residual mean norm: 0.508; target residual mean norm: 0.455;
source vs target mean cosine: 0.993.

Direction audit against privileged target residual:

| predictor | cosine median | orthogonal error median |
|---|---:|---:|
| zero | 0.000 | 1.000 |
| global source mean | **0.903** | **0.427** |
| source kNN k=1 | 0.843 | 0.538 |
| source kNN k=3 | 0.869 | 0.495 |
| shuffled source pairs | 0.703 | 0.710 |

The residual is mostly a shared global direction; kNN adds noise relative to
the global mean. But a 0.427 orthogonal error is substantial, and a high
cosine does not imply closed-loop benefit: the global-mean first-chunk
correction scored 4/10 vs Frozen 5/10 and kNN 6/10.

Therefore the source-paired first-chunk residual route is stopped in this
form. Commit 282104c is reclassified as **known-perturbation paired simulator
calibration**, not target-only TTA. No new regressor, no k/alpha search, no
20-episode run.


## 16. Correction to residuals audit (2026-11-17)

The earlier `residual_identifiability_audit.py` had an index-direction bug:
the distance matrix was `[source, target]` but `argsort(axis=1)` was used as if
it were `[target, source]`. Its kNN numbers are invalid. The corrected audit
(v2) uses `[target, source]`, both state-only and joint state+visual metrics,
and a 5000-permutation shuffled distribution.

Corrected: global mean direction cosine 0.903 / orth 0.427; joint k=1 0.741 /
0.671; joint k=3 0.855 / 0.516; joint k=5 0.894 / 0.447. Shuffled median
0.771, p95 0.843. Thus k=1 is weak, k=3/5 show some local information but do
not beat the global mean. The stop decision for source-paired residual remains.


## 17. Formal execution guard (2026-11-17)

Formal runs now require `--split-json` and `--role`. The legacy no-split path
requires `--allow-legacy-diagnostic`. `target_test` is rejected unless
`confirmatory_test=true`, `target_test_used_for` is empty, and
`target_test_status=locked_unused`.

Role seed pools: target_dev uses `screening_seeds`; target_test uses
`evaluation_seeds`. Previous seed-2014 target-dev runs are legacy/exploratory,
not protocol-compliant formal evidence. The current spatial target test is
`exploratory_accessed`; it is not confirmatory.


## 18. Perturbation-level pilot split

`docs/data/tta_split_libero_spatial_perturbation.json` separates perturbation
configurations:

- target dev: Lighting 2292, dev states 34,35,36,37,39,43,44,45,48,49;
- target test: Lighting 2312, heldout states 3,5,7,9,11,12,15,16,19,23,24,
  26,28,30,38,40,41,42,46,47.

The target-test perturbation config 2312 was not used before. The test state
IDs were used under 2292 exploratory evaluation, so this is a perturbation-
level holdout, not a fully fresh state holdout. Status after Frozen admission:
`admission_accessed`; not confirmatory.

Admission baseline (Frozen, seed 2014): Clean 18/20, Shift 1/20. Target-dev
Frozen with screening seeds: 11/30. These are not TTA method results.


## 19. Protocol-compliant target-dev result (2026-11-18)

Lighting 2292, 10 target-dev states, screening seeds 7777/8888/9999:

- Frozen 11/30;
- First-observation 8/30;
- paired gain 5 / loss 8.

First-observation is negative under the protocol-compliant seed pool. The old
seed-2014 target-dev positive result is legacy and must not be cited.

Target-test admission on Lighting 2312 (Frozen, seed 2014): Clean 18/20,
Shift 1/20. Status is `admission_accessed`, not confirmatory.
