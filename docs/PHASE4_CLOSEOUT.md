# Phase 4 evidence closeout

SOH v2 is an offline-trained residual with a frozen SmolVLA backbone and frozen inference-time parameters. It is distinct from the original training-free moment-alignment V1.

## Spatial development: three-seed paired evaluation

| Group | Frozen | SOH v2 | Rescue / harm | Delta pp | Exact p | Task-cluster 95% CI pp |
|---|---:|---:|---:|---:|---:|---|
| Sensor | 12/60 | 26/60 | 19/5 | +23.33 | 0.00661075 | [-3.3, +46.7] |
| Background | 13/30 | 12/30 | 3/4 | -3.33 | 1 | [-30.0, +16.7] |

| Seed | Sensor Frozen → SOH | Sensor net | Background Frozen → SOH | Background net |
|---|---:|---:|---:|---:|
| 0 | 3/20 → 9/20 | +6 | 4/10 → 3/10 | -1 |
| 1 | 5/20 → 9/20 | +4 | 4/10 → 5/10 | +1 |
| 2 | 4/20 → 8/20 | +4 | 5/10 → 4/10 | -1 |

Sensor is positive in the paired episode test, but the task-cluster interval crosses zero. Background is unresolved. The SOH residual and sensitivity basis were trained/estimated on Spatial init0 tasks; reference statistics are task-LOTO, but this is not an entirely task-held-out adapter-training result. This table contains no three-seed Clean or Lighting results and is not untouched confirmation.

## Beta-zero diagnostic (t=0 only)

| Group | Beta 1 pooled XYZ error reduction % | Beta 0 pooled XYZ error reduction % |
|---|---:|---:|
| Sensor | +52.02 | +45.99 |
| Background | -5.27 | -5.60 |

These are distances in policy action components relative to the same-state Clean Frozen action, not measured EEF pose errors, success rates or a closed-loop beta ablation. Beta-zero was a separately trained checkpoint; its training command/receipt are retained. The previously uncommitted --damping-beta training option is now included, with the same default 1 and an exact hash match to the server implementation.

## libero_10 transfer with new-suite Clean calibration

Spatial SOH weights and the source action-sensitivity basis remain frozen. New-suite init0 Clean trajectories calibrate moments and the deep gate under task LOTO; init1 evaluates Clean10 and bilateral Sensor20 severity3, seed0. This is calibrated transfer, not fully zero-shot. The native BF16/CPU preprocessing path is unchanged. The horizon is 520 policy steps plus 10 settling steps.

| Group | Frozen | SOH v2 | Rescue / harm | Delta pp | Exact p | Task-cluster 95% CI pp | Gate accept % |
|---|---:|---:|---:|---:|---:|---|---:|
| Clean | 4/10 | 4/10 | 0/0 | +0.00 | 1 | [+0.0, +0.0] | 6.99 |
| Sensor | 0/20 | 1/20 | 1/0 | +5.00 | 1 | [+0.0, +15.0] | 75.66 |

| Sensor family | Frozen | SOH | Rescue / harm | Gate accept % |
|---|---:|---:|---:|---:|
| motion_blur | 0/4 | 0/4 | 0/0 | 29.28 |
| gaussian_blur | 0/4 | 0/4 | 0/0 | 91.54 |
| zoom_blur | 0/4 | 0/4 | 0/0 | 99.04 |
| glass_blur | 0/4 | 1/4 | 1/0 | 100.00 |
| fog | 0/4 | 0/4 | 0/0 | 60.82 |

Source init0 native Frozen Clean: 6/10; all ten full trajectories, including failures, entered the reference collection. Each evaluated task is excluded from its fitted reference.

If all observed paired differences are zero, resampling those differences gives a degenerate [0,0] bootstrap interval. This is not an equivalence test or proof of zero population harm, especially in these small cohorts.

The run is a single-seed exploratory transfer check. Cross-suite transfer refers to the SOH residual and sensitivity basis, not a claim that the frozen SmolVLA backbone never saw libero_10 during its original training; its pinned model card does not specify the training dataset. There is no new-suite Lighting/Background result, no direct contemporaneous DeepGate third arm, and no universal geometry guarantee. A per-frame 95th-percentile gate is not an episode-level safety guarantee.

## Audit and provenance

Independent raw audit: 30 pairs, 29094 replans, 1930 equal-input rejected-gate action checks; status PASS.
Bounded paired RGB replacements: 2 events, maximum 1 pixels/view and 1 uint8 LSB. Reuse is restricted to identical physical/render state.
SOH checkpoint SHA256: `b4fe1ac8c4d206fa13f445c93b351682a488170de8ac117fb9c799d7145a8ceb`.
libero_10 calibration SHA256: `a7e305c04d723eeaed4d13bf0ab141a485c7be610e4af4b382ece1566b73abe3`.
The retained Spatial records hard-coded arm.policy_seed=0 in the reused base rollout. Independent comparison against the actual Torch CPU/CUDA RNG confirms that seeds 0, 1, 2 really executed different streams, with paired streams equal within each condition. The original records are unchanged; the release corrects future Spatial arm seed metadata. The archived runtime source hashes refer to the original executed files.
The current transfer summary retains the legacy internal count key deepgate for its SOH arm; the exported table correctly labels it SOH v2.
Full trajectories are kept in the separate evidence archive. Summary, preflight, source folds, source gate, runtime configuration and code hashes are retained beside this report.

## Appropriate claim

Spatial provides an encouraging Sensor result across three seeds, with uncertain task-level generalization and unresolved Background. The current libero_10 pilot does not establish a stable cross-suite improvement: Frozen is at the Sensor floor and 1 rescue(s) were observed. Gate triggering alone does not guarantee task recovery. The next evidence gap is native baseline validation and a prespecified evaluation with enough non-floor cases, before further SOH parameter changes. The result must not be replaced by the t=0 diagnostic or described as fully zero-shot.
