# Global t=0 diagnostic V1 (2026-11-24)

Only the frozen 100-case V2 Dev, no full rollouts, no policy updates and no protected data. This replaces the not-yet-launched closed-loop request. Four original V1 noises per task/pair, exact original RGB and first z0 hashes, native FP32 10-step Euler; one live OSMesa environment. Independent inference identity checks and complete failure retention.

## Representation definitions

DINO patches (256x384/view), patch means, resampler (48x768/view), one-layer post-LN prefix K/V (48 concatenated-head 768-d channels/view), output_norm suffix (12x768 after averaging 10 Euler steps and four noises). Suffix is fused and cannot be honestly assigned to either view. Hook is immediately before velocity_projection. Also retain prepared RGB pixel moments, state8 and native prefix caches for controlled probes.

## Source fitting

For each evaluated base task, use exactly the other nine Dev Clean tasks, one t=0 state each. No Shift, outcome or evaluated-task Clean in fitting. This is LOTO **Dev Clean reference fitting**, not independent Source Calibration or a calibrated safety guarantee. Input moment sketches and fit source IDs are retained.

Primary diagonal distance uses source token/channel mean and variance pooled across nine equal-size Clean token sets, 10% variance shrinkage to the source average channel variance and a numerical floor. DINO-mean distance instead uses the nine image-level patch means. Report RMS diagonal z-distance and pooled-mean cosine distance, not unregularized full covariance inversion. Nine independent scenes remain nine despite many correlated patches/tokens. No dimensions, shrinkage or thresholds are tuned on failure labels.

For unilateral/bilateral diagnosis: source-only thresholds are maxima of inner leave-one-out scores (each of the nine source Clean images scored against the other eight). Outer target task remains excluded throughout. Strictly above threshold is abnormal; classify agent-only, wrist-only, both or neither. This is descriptive small-sample classification, not validated false-alarm control. Report raw asym=agent z-distance/(wrist z-distance+1e-6), all group distributions and predictions. Do not assume Background/Lighting corrupt both views equally.

## Outcome analysis

After fitting/scoring, attach eventual outcomes and paired-Clean prediction-reference distances. All per-location scores and factors/subtypes reported: Sensor, all Background, scene/table texture separately, Lighting. Failure AUROC, within-task concordance, XYZ/rotation Spearman correlations with SAME four-noise native prediction-reference L2. Compare to four-noise raw Flow variance, not a mixture of single/four-noise estimators. Ten-task paired cluster bootstrap differences may be exploratory; many scores/subgroups and few clusters prevent an unqualified significance claim. No learned fusion or best-score claim.

## Selected probe population

All 13 Background failures + 6 Lighting failures + 10 Clean controls =29. Selection by existing outcomes is explicitly privileged diagnostic design, not method evaluation on a representative pool. Four identical original noises for current/paired Clean/probes. Paired Clean reference is a successful policy prediction, not expert ground truth. It enters ONLY oracle replacements and after-the-fact evaluation.

Oracle variants: agent Value, wrist Value, agent K+V, wrist K+V, both-view K+V. Other prefix tokens and conditions remain current. Since this is one block and paired nonvisual states/language should match, both-view K+V replacement should approximately reproduce paired Clean actions; verify this implementation ceiling, do not sell it as an online algorithm.

Target-only variants: both-view photometric channel mean/std alignment alpha .25/.50 (own-image moments allowed, target-task source excluded), and both-view visual K/V soft pullback tau 2/3 x beta .3/.6. Source moments are per-view pooled token-channel stats. Components with |z|<=tau are bitwise unchanged; flagged components move beta of the way to the threshold boundary. Policy weights unchanged. RGB operator is mixed in prepared uint8 pixel space then clipped/rounded. Do not optimize alpha/beta/tau using reference actions.

Report native/clipped XYZ/rotation/all L2 changes, Clean absolute drift, altered-component fractions and standardized before/after Flow relative variance. Descriptive nonworsening-curvature guard = relative_after <= relative_before*(1+1e-6), no label-based threshold tuning. It is not a validated safety certificate.

No 10–15 minute guarantee: runtime measured; full feature caches stay on server/local artifacts with hashes, while compact moment sketches, source fits, probe actions and analysis outputs are committed. No automatic fallback training or promotion to a unified method.

Matched-coverage oracle addendum (before examining probe outcomes): also replace both-view Values while preserving both-view Keys. This separates a Key intervention effect from the confound of replacing more views. It is an extra privileged ceiling, not a target-only method.
