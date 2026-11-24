# Flow-Time Geometry V1: positive diagnostic, adaptation unproven

Date: 2026-11-24 (Asia/Shanghai). All results are development-only.

## Decision

**CONTINUE_SIGNAL_VALIDATION; DO_NOT_CLAIM_TTA_SUCCESS.** The prespecified exploratory forward gate passes on Sensor and Background, not Lighting. A stronger statement that flow straightening repairs perception or improves execution is not supported. The work completed here is the proposed forward litmus test plus a seven-case one-step gradient mechanism test. No candidate action was executed, no policy weight was changed, no Blind or target-test was accessed. A 3–5-step closed-loop adaptation campaign is not justified merely by optimizing its own surrogate.

## Scope and execution integrity

Exactly the frozen V2 development manifest: 10 unique Clean controls + 30 Sensor + 30 Background + 30 Lighting. Official Frozen outcomes remain 10/10, 17/30, 17/30, 24/30 respectively. At environment time zero after the original settling steps, each observation is evaluated using the original policy z0 plus three independent augmentation-stream z0 draws. The same diagnostic seed derivation is used within a pair group. Canonical language, raw two-view RGB and original z0 are checked against the original baseline receipts. Policy-state inputs are recreated from the deterministic reset/settling procedure; baseline files do not contain an independently stored t=0 state8 hash.

- 100/100 cases complete, no exceptions.
- Both raw RGB hashes match for all cases.
- Original policy z0 hashes match for all cases.
- Native manual Euler controller vs official predict: maximum absolute difference **0.0** across all 100 cases.
- 4 initial noises × 10 native steps × 100 observations recorded; official predict adds a separate parity forward.
- Independent verifier recomputed every saved velocity metric, baseline/result hash and append-only result receipt.
- The OSMesa runner uses one live environment. Runtime wrapper noise is disabled before reset and official step-keyed sensor_transform is applied once. Four original seed streams remain separate; added probes use augmentation-derived generators.
- Summed per-case elapsed time was 1,699.6 seconds over four concurrent workers. This is not wall-clock latency or a deployment inference benchmark; model loading and map re-rendering are separate.

Protocol: FLOW_STRAIGHTNESS_PROTOCOL.md, fixed before inspecting full-cohort scores. Primary score is raw all-coordinate velocity variance on the original z0. Prespecified normalized variance, turning and other channel/noise aggregations are reported as checks rather than retrospectively replacing the primary.

## Primary outcome association at t=0

Larger velocity variance is the prespecified risk direction. Coordinates are normalized action coordinates; raw score averages across coordinates (equivalent to the proposed squared norm up to a fixed dimensional constant). The metric is discrete velocity nonconstancy, not geometric curvature in the strict differential-geometric sense.

| Factor | Eventual failures / cases | Failure AUROC | Within-task fail>success pairs | Task-cluster bootstrap AUROC 95% |
|---|---:|---:|---:|---:|
| Sensor | 13/30 | **0.8733** | **23/23** | [0.7266, 1.0000] |
| Background | 13/30 | **0.7873** | **15/21** | [0.4428, 0.9304] |
| Lighting | 6/30 | 0.6319 | 1/6 | [0.2000, 1.0000] |

Bootstrap uses 1,000 resamples of the 10 base-task clusters, seed 20261124; factor/task coverage varies. Degenerate single-class resamples are omitted and their count retained (995 valid Lighting resamples). These small-cluster exploratory intervals do not certify population generalization. Within-task pairs are overlapping comparisons, not 23 independent experimental units.

Clean-vs-Shift AUROC is only **0.6400**. The score is not a clean/shift separator that detects all shifted images, yet its Sensor within-domain outcome association is appreciably stronger. This does not establish causal risk detection or effective intervention lead time: the label is the eventual episode outcome, measured only once at t=0.

## Scale and initial-noise checks

| Prespecified signal | Sensor AUROC | Background AUROC | Lighting AUROC |
|---|---:|---:|---:|
| Raw variance, original z0 (primary) | 0.8733 | 0.7873 | 0.6319 |
| Variance / mean squared speed, original z0 | 0.8597 | 0.8009 | 0.6736 |
| Adjacent-velocity angular turning, original z0 | 0.8643 | 0.7873 | 0.6389 |
| Raw variance averaged over 4 z0 | 0.8733 | 0.7738 | 0.7986 |
| XYZ endpoint diversity over 4 z0 | 0.6425 | 0.6968 | 0.6042 |

Normalized and angular checks retain the Sensor trend; it is not eliminated by dividing out velocity energy. The endpoint-diversity comparator is weaker on Sensor here, but shared small Dev data do not justify a definitive superiority claim. Lighting is sensitive to noise aggregation and weak in original-z0 within-task comparison. Do not choose its best secondary score as a validated result.

Sensor raw-variance channel checks: XYZ AUROC 0.6652; rotation 0.8371; gripper 0.8416. The aggregate is not an XYZ-only translational instability measure; rotation/gripper contributions matter. Rotation-coordinate variation is not reported as a physical SO(3) geodesic metric.

## Clean threshold is not validated No-Op safety

Threshold = maximum raw primary score on the same 10 Dev Clean observations, alarm if strictly greater. Threshold 0.0073125435451238665. Empirical Clean resubstitution false alarms = 0/10 by construction.

| Factor | Failure recall | False alarms among successful Shift |
|---|---:|---:|
| Sensor | 12/13 (92.3%) | **10/17 (58.8%)** |
| Background | 8/13 (61.5%) | **6/17 (35.3%)** |
| Lighting | 2/6 (33.3%) | 2/24 (8.3%) |

This threshold was not calibrated on independent Source Calibration. It must not be described as a statistically valid 0% FPR guarantee or a deployable selective-adaptation gate. In particular the Sensor alarm still activates on many successful shifted episodes.

## Subtype/severity confounding remains material

Sensor subtype AUROCs: motion blur 1.000 (n=6, 2 failures); Gaussian blur .700 (n=7, 5 failures); zoom blur 1.000 (n=3, 1 failure); glass blur .778 (n=13, 4 failures); fog not estimable (n=1, failure).

Sensor same-task-and-subtype comparisons are 6/6 concordant, but there are **zero** failure/success pairs matched on both task and exact noise code. Noise severity independence is therefore not established. Subtype samples are too small for subgroup performance claims.

Background scene texture AUROC .972 (15 cases, 3 failures), table texture .480 (15 cases, 10 failures). Same-task-and-subtype concordance is only 6/10. The aggregate Background association can depend strongly on subtype composition; this is a reason to prioritize Sensor validation, not to claim a general visual-risk invariant.

## Post-hoc simple-output controls (not used for the primary gate)

A further explanatory check asks whether curvature merely recovers an already accessible small-action signal. Lower first-chunk clipped XYZ command RMS has AUROC .629 on Sensor, .796 on Background and .257 on Lighting. Initial-velocity energy (larger-is-risk) gives .285/.624/.701; final-velocity energy gives .719/.498/.389. These controls were specified after seeing the primary result and are exploratory. Sensor curvature retains a useful distinction from these simple summaries in this sample; Background curvature does not outperform the small-action summary. No superiority inference or extra score selection is made.

## Seven-case gradient mechanism test

Predefined illustrative selection: lexicographically first success and failure in each Shift factor plus first Clean success. Selection uses existing baseline labels for illustration only; it is not an independent risk-validation set. Exact original resampler, frozen policy weights. Agentview multiplicative token gate exp(delta), delta=0 identity, wrist unchanged. One RMS-normalized gradient step with alpha=.01 and absolute gate bound .05; no learning-rate search. Objective is raw velocity variance through all 10 native Euler steps.

- Identity controller mismatch: 0.0 in every selected case.
- Central finite-difference vs analytic directional derivative relative error: maximum **0.000619** (0.0619%).
- Optimized-z0 variance falls in all seven, by **0.59%–1.35%**, including Clean and successful Shift.
- Held-out-z0 variance falls in six, but **increases 0.340% on the selected Sensor failure** despite a 1.272% training-z0 decrease.
- XYZ controller RMS changes only -0.081% to +0.332%; no substantive action-amplitude recovery demonstrated. No catastrophic norm collapse observed in this one small step either.
- Exact-RGB gradient maps are saved. Sensitive patches include robot and task-object areas as well as scene content. There are no nuisance masks or causal localization labels. Maps cannot establish identification of "pathogenic" background/blur tokens.

The gradient is implementable and numerically valid. Descent is not failure-specific, is imperfectly transferable across initial noise, and does not prove restoration of a correct action. The proposed inference from "curvature goes down" to "task performance recovers" remains untested.

## Relation to previous retained development results

The latest remote report has these results (same development pool; repeated method selection, not confirmation):

| Retained intervention | Shift successes /90 | Dev Clean | Interpretation |
|---|---:|---:|---|
| Frozen | 58 | 10/10 | reference |
| LPE eps .1 | 62 | 10/10 | weak positive |
| Single-prototype spatial beta4 | 63 | 10/10 | +5 net development candidate |
| Multi-prototype spatial beta4 | 58 | 10/10 | no net gain |
| Multi-prototype spatial beta6 | 53 | 10/10 | negative ablation |
| 10-task channel mean shrink | 58 | 9/10 | factor tradeoff and one Clean loss |
| Space×Channel | 59 | 10/10 | combined gains largely cancel |

These are prior report results, not rerun in this experiment. Flow V1 adds a promising t=0 diagnostic, not another closed-loop success-rate row. Earlier length-leaking reversal claims remain retracted; corrected fixed-prefix hypokinesia signals also had many successful-Shift false alarms.

## Next bounded experiment

Keep the primary score fixed. First calibrate a gate using independent Source Calibration and characterize false alarms on successful Shift, with subtype/severity controls. Then test a small paired closed-loop intervention with fixed gate/update/rollback rules and report rescued AND harmed cases, including Clean. Compare gradient straightening against equal-size random/zero gate changes and the retained spatial candidate; require held-out-noise and task-preservation checks. Do not adapt just because a same-sample surrogate decreases, unlock held-out assets, claim originality from this experiment, or automatically start a learned suffix-alignment fallback.

Evidence: docs/data/flow_geometry_v1/{summary.json,verification.json,gradient_summary.json,raw/,gradients/,maps/,flow_geometry.png}. Raw results preserve traces and append-only receipts. Tests: 5 new numerical tests and 24 existing V2 tests pass.
