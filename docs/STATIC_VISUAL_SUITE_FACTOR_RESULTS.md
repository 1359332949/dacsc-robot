# Frozen Spatial development results

Date: 2026-11-23. Fixed CoRP, no TTA or parameter update.

## Results

The corrected Clean compatibility campaign completed 50/50 successes, five predetermined states per task. This is empirical compatibility, not training coverage or a population guarantee.

The frozen development manifest completed 100/100 rollouts without exceptions: 90 Shift and 10 unique Clean controls. The server verifier reopened all results and matched their hashes to execution receipts, checked physical pairing and timing. Clean controls succeeded 10/10.

| Factor | Success | Rate | Shift minus paired Clean |
|---|---:|---:|---:|
| Lighting | 24/30 | 80.00% | -20.00 pp |
| Background | 17/30 | 56.67% | -43.33 pp |
| Sensor Noise | 17/30 | 56.67% | -43.33 pp |

The primary equal-factor mean is 64.44% success and -35.56 percentage points relative to paired Clean. All 32 losses are Clean-success/Shift-failure. Lighting failures span four base tasks, Background six, Sensor eight. Coverage is respectively nine, nine and ten tasks; task composition differs across factors.

The old summary also emitted a task-factor mean of -37.14 pp and a stale n=3/cell note. These are superseded: the primary estimand is the factor mean, while the task-factor mean is explicitly descriptive. The original summary is retained for provenance.

## Interpretation and limits

Sampling is suite-by-factor, 30 instances per factor, seed 2026112301, without replacement after equivalence processing. Target initialization index is zero. Shared Clean controls are not 90 independent samples. No uncertainty or significance claim is made from independent-instance assumptions. These results support a development failure landscape, not a TTA improvement or risk detection claim.

A historical full10030 Frozen benchmark was accessed. The current 90 cases reuse official assets for development. Frozen visibility is distinct from method-development access: a future official method-held-out evaluation may be possible after auditing and excluding all historical method/threshold/feature/diagnostic accesses. It must disclose prior Frozen visibility and cannot be described as zero-contact Blind. Missing method-access records remain unknown, not certified clean. The current Blind manifest remains empty and locked; no new test set is authorized by this report.

Source Fit/Calibration and eventual state-disjoint Clean evaluation are separate roles. Disjoint indices must also be checked against actual physical-state duplicates and history; they do not imply task-disjoint generalization. Ten-task cluster bootstrap would be exploratory given few clusters. Instance exact McNemar additionally requires independence of discordant pairs; removal of shared Clean alone does not establish that assumption.

## Invalid environment batches

Earlier twin-environment OSMesa execution closed a secondary renderer while the policy renderer was live and corrupted RGB. Those batches are invalid environment diagnostics, not policy performance. The corrected lifecycle closes the reference environment before creating the policy environment. Regression reproduced single-environment RGB exactly. Official canonical language was also restored. Historical studies are not automatically invalidated by this runner-specific defect.

## Next experiment

First audit coverage by task, difficulty and sub-dimension. Then predefine a small development-only, training-free target-signal diagnostic with both successful and failed episodes, Source/Clean false alarms and explicit timing. Background and Sensor are candidates, but equal observed failure counts do not prove equal intrinsic difficulty. Do not train Adapter or unlock confirmation data based on these baseline results.

Evidence: docs/data/static_visual_v2_suite/final/ contains frozen manifests, original and corrected summaries, per-episode results, server verification and Clean smoke summary. No TTA method was evaluated.
