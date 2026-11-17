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
