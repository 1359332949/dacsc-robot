# Flow geometry litmus test, V1

Frozen 2026-11-24 before inspecting new diagnostic outputs. Only frozen V2 100 Dev cases, no protected roles. No environment actions beyond original 10 settling actions; no policy updates.

## Hypothesis and cautions

Straight stochastic training interpolants do not guarantee straight integral curves of a learned conditional vector field. Conditional expectations can mix endpoint pairs; rectification/reflow is a separate operation. This is a falsifiable association study, not a proven invariance or novelty claim. A constant zero vector field also minimizes velocity variance; lowering this energy does not certify correct action.

## Stage 1

Exactly reproduce both raw initial RGB hashes, official canonical language and first policy z0 from baseline receipts. Native 10-step Euler and original policy.predict parity must be <=1e-5. Explicit environment RNG and step-keyed sensor transform once, only one live simulator. Asset, manifest and checkpoint hashes checked before inference.

Four trajectories per observation: original baseline first z0 plus three augmentation-stream z0 with fixed derived seeds common within a pair group. No lambda-combined energy fitted on Dev. Record raw velocities, normalized-space velocity variance, variance divided by mean squared speed, adjacent cosine turning and path excess. XYZ, rotation, gripper reported separately. Controller-space endpoint variance is secondary and not called SO(3) geodesic variance.

Primary direction: larger nonstraightness predicts failure. Primary score: original-z0 all-channel raw velocity variance as proposed; relative variance and angle are prespecified checks. Additional-z0 averages are robustness checks, not candidates selected by maximum AUC. No lower-is-risk reorientation after seeing results.

Report Clean-vs-Shift and, separately, failure-vs-success per factor, plus same-task failure/success pairs and noise subtype/severity coverage. Clean-derived threshold is only descriptive 10-state Dev calibration: > maximum observed Clean gives empirical zero FPR by construction, not validated FPR control. No final-risk labels copied to chunks: only t=0 episode-outcome association is measured, no intervention lead-time claim. Failure to discriminate Clean vs Shift alone does not rule out within-domain risk discrimination.

Progression: a factor must have primary AUC >= .70 and same-task concordance >= .60 (at least 5 eligible pairs), failure recall >= .50 at > maximum Clean, and relative variance must have AUC > .50. These are exploratory development gates, not confidence guarantees. Otherwise STOP for flow-energy-driven adaptation V1; retain all secondary results. No switch to another learned manifold method without a separate design.

## Stage 2: illustrative gradient diagnostic

One lexicographically first success and failure for each Shift factor plus first Clean success (7 cases), selected by the existing baseline outcome for illustration only. Freeze all policy weights; exact original resampler. Multiplicative agentview token gate exp(delta), delta=0 identity, wrist unchanged. Full 10-step gradient of raw velocity variance, one fixed .01 RMS-normalized negative-gradient step clipped to .05; no learning-rate sweep. Independent held-out z0 evaluation; finite-difference gradient check. Save patch-gradient norms and gate changes. No spatial ground-truth mask means heatmaps cannot establish causal nuisance localization. No controller norm recovery is counted as success, and no candidate action is executed. Even if Stage 1 stops, gradients may be computed as a mechanism/collapse control, not evidence for deployment.

## Literature verification

Rectified Flow: arXiv:2209.03003, primary source. The supplied SALT/RECLAIM/T3R/CoRe summaries and claims of universal applicability or first-ever novelty are not assumed established. This experiment cannot establish originality or generalize to all flow policies. Hardware timings will be measured rather than promised as 10 minutes/milliseconds.
