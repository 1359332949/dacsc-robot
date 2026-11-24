# High-energy Flow repair probe: Sensor-specific alignment, no general GO

Date: 2026-11-24, Asia/Shanghai. Selected V2 Dev t=0 mechanism experiment.

## Decision

**NO_GENERAL_GO; SENSOR_VALUE_TRUST_MECHANISM_IS_PROMISING.** No configuration passes the predeclared 6/8 material-improvement screen. Effective post-LayerNorm Value attenuation produces substantial paired-Clean XYZ alignment on all five selected Sensor failures, but worsens all three selected Background cases. Much of the Sensor benefit is reproduced by fixed view attenuation. There is no evidence here of a universal flow-straightening task-repair algorithm, nor a proof of zero-action collapse.

No policy weights changed, no adapted action executed, no full rollout, no Blind/Held-out access. All action-distance percentages below are prediction-reference error changes, NOT closed-loop success-rate changes.

## Audit of the supplied interpretation

The reviewer correctly identified the top five Sensor and top three Background failure IDs by four-noise paired-Clean curvature ratio. The five Sensor ratios are 47.74, 41.78, 37.05, 22.58, 16.98; Background 24.34, 20.96, 4.49. The initial lexicographic seven-case probe did not cover these extremes; it was an illustrative numerical-gradient check, not a conclusive repair test.

Recomputing from all 100 saved cases, with the distance defined as the mean over four identical z0 of flattened full-chunk UNCLIPPED controller XYZ L2:

| Factor | Four-z0 raw-variance failure AUC | Paired-Clean normalized AUC | XYZ reference-distance AUC | Raw curvature vs XYZ distance Spearman |
|---|---:|---:|---:|---:|
| Sensor | .8733 | .8778 | .8507 | .9177 |
| Background | .7738 | .8416 | .7738 | .4318 |
| Lighting | .7986 | .4653 | .5486 | .6814 |

Background normalized-curvature correlation is .6165. The earlier .7873/.6319 raw AUCs use the original single z0, not the four-z0 mean. Mixing those two estimators obscures interpretation. The slight Spearman difference from the supplied .916 depends on the exact L2/noise aggregation definition.

The k=7 group-mean velocity-deviation fail/success ratios reproduce 11.41 (Sensor), 24.35 (Background). They are descriptive group mean ratios influenced by extreme cases, not evidence that a physical prior causally "pulls trajectories back" at that flow time.

Successful Clean policy actions are privileged prediction references, NOT expert true actions or unique correct trajectories. Lighting aggregate statistics do not identify a specific physical failure cause. A Dev-selected threshold such as .024 remains retrospective and cannot be promoted to a calibrated safety gate.

## Frozen design and implementation

Selection: five highest-ratio Sensor failures, three highest-ratio Background failures, plus two lexicographic Clean controls from different tasks. Outcome/paired-Clean-based extreme-case selection is explicit and limits generalization. The full selection and hashes were frozen before updates.

Each full grid has 3 knobs x 3 objectives x 2 learning rates = 18 configurations per case, 180 per phase. Four original V1 train noises, four independent held-out noises, Adam .3/.8 for ten steps; checkpoints 0/5/10. Relative, late-weighted and target-speed-anchored objectives; squared-L2 regularizer .001*(sum(delta^2)+s^2). Five matched-norm nulls per checkpoint. The optimize function has no Clean/reference/outcome arguments. Paired Clean is evaluated after all update trajectories of the target are complete. No configuration is selected for production using reference distance.

Original view knob: resampler agentview output times exp(s), s=0 identity. Uniform per-view resampler logit offsets would cancel in softmax, so were not used. Spatial bias has nonpositive logits with max(delta)=0 gauge and [-8,0] bounds; this constrains logits, not every normalized attention probability.

Preflight correction: an initial timing/correctness pilot used mean-squared rather than sum-squared spatial regularization. An initial full launcher was stopped with zero completed cases, receipts retained; both final full grids use the specified squared-L2 sum. Pilot data are excluded from final summaries.

## Phase A: original resampler-output view knob

All 180 configurations completed. Spatial relative/lr .8/step10 reduces held-out raw variance by 28.87% on average across eight failures, but XYZ reference distance by only 3.47%. Seven cases improve by any amount; none reach 10%. Average matched-null improvement is .03%. All-command norm remains >=.988 of baseline over failure/noise combinations.

All optimized resampler-output view weights remain 1, and joint essentially matches spatial alone. A fixed trust sweep found that even trust .1 changes average XYZ reference distance by only .0166%. Therefore this placement is practically ineffective, not evidence that all view weighting is ineffective.

## Architecture issue and Phase B correction

This checkpoint has **one DiT block**. Its prefix Keys/Values are computed from per-token LayerNorm. Multiplying a positive scalar into a whole visual token is approximately canceled by LayerNorm (apart from epsilon). Prefix residual updates do not feed another block's cache in this one-block checkpoint.

A separately labelled second full grid moves only the view-trust operation: multiply agentview prefix Values AFTER LayerNorm/projection by exp(s), leaving Keys, wrist Values and language/state unchanged. This changes agentview contribution, not attention probability. Spatial and all optimization/evaluation settings remain identical. This follow-up is adaptive development, not an independent confirmation.

Both grids pass independent integrity verification:
- 10/10 cases, 180 configurations, 360 optimized checkpoint evaluations and 1,800 matched-null evaluations EACH.
- Native sampler vs predict maximum error: 0.0.
- Zero-gate controller maximum error: 3.532e-6; batched-vs-saved V1 error: 3.785e-6.
- Null spatial L2 norm mismatch <4.8e-6, constraints and scalar matches verified.
- Four held-out noise hashes are disjoint from four training hashes.
- Source hashes, selection hashes and result end receipts verified; no exceptions.

## Effective Value-trust results

For the fixed illustrative configuration **relative objective, lr .3, step10** (not selected by reference-distance ranking):

| Knob | Sensor mean XYZ distance reduction | Background mean XYZ distance reduction | All eight | Cases with >=10% reduction |
|---|---:|---:|---:|---:|
| Spatial | 2.83% | 0.38% | 1.91% | 0/8 |
| Post-norm Value trust | 25.04% | -10.27% | 11.80% | 5/8 |
| Spatial + Value trust | **25.42%** | **-10.29%** | **12.03%** | **5/8** |

Negative reduction means the action moved farther from paired Clean.

Joint relative/lr .3/step10 per-case details:

| Case prefix | Factor | XYZ reference distance reduction | Held-out relative curvature reduction |
|---|---|---:|---:|
| 94221db7af99 | Sensor | 24.28% | 87.79% |
| 537c44cf5378 | Sensor | 12.77% | 75.49% |
| c799396aefd5 | Sensor | 21.64% | 84.88% |
| 3d16d19504b3 | Sensor | 52.36% | 81.48% |
| 8515f621f7ce | Sensor | 16.07% | 76.65% |
| db52f74d7afc | Background | -14.53% | -11.04% |
| f769d7749054 | Background | -16.01% | -19.24% |
| ea09068ddd4e | Background | -0.33% | 7.59% |

Sensor mean held-out relative variance falls 81.26%; Background mean rises 7.57%. Four train noises do not eliminate train-to-held-out discrepancy. One Sensor case's rotation-coordinate reference distance worsens 15.15% while XYZ improves: translation alignment alone is not full action correctness.

All-command norm across held-out failure/noise combinations is >=.930 of initial, but minimum XYZ-only norm ratio is .693. There is no zero-action collapse, yet substantial translational magnitude change in some samples must not be hidden by gripper/rotation contributions to the total norm.

The two Clean controls move by absolute flattened XYZ L2 .00475 and .01085 (controller units); these were forced-update diagnostics, not gated rollout tests. No Clean success or No-Op guarantee follows.

## Nulls, fixed attenuation and necessity of optimization

For joint relative/lr .3/step10, learned XYZ improvement averaged 12.03%, versus 11.73% for matched random spatial suppressions with the SAME learned scalar. Learned beats the null median on 7/8 cases, but the incremental mean gain is only .30 percentage points of error reduction. Most benefit is attributable to Value trust, not learned spatial localization.

A scalar constrained <=0 has only one equal-L2 direction; its equal-norm null is necessarily identical. View-only null ties cannot be treated as evidence against a useful scalar intervention, or as proof that gradient optimization adds value.

A final fixed post-norm Value-trust control uses the already declared levels, all reported without picking a production level:

| Fixed trust | Sensor mean XYZ reduction | Background mean XYZ reduction | >=10% reductions among eight |
|---|---:|---:|---:|
| .75 | 9.73% | -4.79% | 2/8 |
| .50 | 18.52% | -9.83% | 4/8 |
| .25 | 24.27% | -15.11% | 5/8 |
| .10 | **25.05%** | -18.35% | 5/8 |

Fixed .1 nearly matches optimized view-only Sensor benefit (25.04%). Thus the data support a usable Sensor-specific intervention mechanism, but do not establish that flow-energy optimization is necessary. Fixed attenuation also changes the two Clean references substantially (XYZ L2 .798 and .472 at .1), underscoring the need for selection/calibration rather than unconditional attenuation.

## Verdict and next bounded question

All 18 corrected configurations reach material improvement on only 5/8 cases, not the required 6/8. Do not lower the threshold or relabel a Sensor-only post-hoc subgroup as passing the original cross-factor GO rule. Do not conclude that all curvature minimization is a cheating shortcut: there is nonzero directional alignment on Sensor, and no observed zero-action collapse. Background provides specific harm and generalization evidence.

The next useful experiment is a separately frozen **Sensor-only** evaluation covering successful as well as failed Dev instances, comparing a curvature gate plus fixed post-norm Value attenuation against the same gate plus optimized attenuation. First calibrate any gate using independent Source Calibration; keep held-out assets locked. This tests selection and the necessity of optimization, rather than another larger spatial/channel combination. No learned suffix-alignment fallback or 100-case closed-loop campaign was started here.

## Literature correction

The checked CoRe primary text (arXiv:2608.14822, Online Suffix OOD Detection, Eq.2) defines a per-task Mahalanobis OOD score over stochastic-forward-averaged action-expert suffix latents: the last action-head hidden state before motor decoding. Eq.2 is not a Suffix/Token manifold-projection repair objective. Its detector therefore provides a source-calibrated anomaly-score reference, not direct support for the proposed suffix-alignment fallback. No novelty or top-conference acceptance claim is made.

## Evidence and reproduction

- `docs/data/flow_geometry_v1/causal_repair_probe.json`: corrected post-norm grid, all per-case/configuration metrics.
- `causal_repair_token_probe.json`: original placement, preserved.
- `repair_verification.json`, `repair_kv_verification.json`: integrity checks.
- `repair_claims_audit.json`: 100-case retrospective claim audit.
- `view_control_summary.json`, `value_control_summary.json`: fixed-placement controls.
- `repair_selection.json`: frozen 10-case selection and full grid.
- Raw JSONs and append-only receipts are distributed in compressed archives; see REPAIR_ARCHIVES.md.

Tests: seven new intervention tests, five geometry tests and 24 V2 regression tests; actual native-parity checks in both 10-case grids. No unrelated local deletions were included.
