# Selected high-energy t=0 intervention probe V1

Frozen 2026-11-24 before new intervention outcomes. This is an exploratory surrogate/action-reference experiment, not proof of causal task repair. Paired successful Clean policy actions are reference predictions, not expert ground-truth actions or unique correct solutions.

## Selection and roles

From existing 100 V2 Dev cases, select top 5 Sensor failures and top 3 Background failures by four-noise mean raw variance divided by paired Clean four-noise mean. This deliberately uses outcomes and Clean to select extreme diagnostic cases. It is not an independent risk evaluation. Select lexicographically first 2 Clean controls on different tasks. Freeze full IDs, selection hashes and grid before model updates. Ratios used in selection never enter optimization.

New probe sees t=0 only. Reset/settle using original environment RNG, noise stream exactly once, canonical language and both RGB hash checks. Only one live environment. Source code/checkpoint/asset/manifest hashes retained, policy eval and requires_grad_(False), no adapted action execution, no protected-role access.

## Fixed complete grid, no oracle hyperparameter selection

3 knobs x 3 objectives x 2 learning rates = 18 configurations per selected case. Adam, checkpoints 0/5/10; learning rates .3 and .8; regularization .001*(sum(delta^2)+s^2). Reset gates and optimizer for every configuration. Train on original V1 four z0. Four new held-out z0 are identical for all configurations and paired Clean, generated from an independent augmentation-seed namespace.

Knobs:
- spatial: manual native-equivalent resampler with additive agentview delta <=0. Eye-in-hand unchanged. Fix softmax gauge max(delta)=0; bounds [-8,0]. Relative attention can increase for less-suppressed patches: nonpositive logits do not mean every attention probability decreases.
- view: agentview output tokens multiplied by exp(s), s in [-4.60517,0], exact identity at s=0 and trust in [.01,1]. Wrist tokens unchanged. This is input trust ablation, not a guarantee that wrist carries adequate task information. Uniform per-view attention logit offsets cannot serve this role because softmax cancels them.
- joint: both.

Objectives (per-noise, averaged across four noises):
- relative: velocity variance / mean squared velocity, all 10 native steps.
- late: weighted mean, variance and energy with weights 2 for k=6..9 and 1 otherwise.
- anchored: relative + .1*mean((L2 norm of mean velocity - original target-observation mean-velocity norm)^2). Anchor is detached and contains no Clean information.

Clean is rendered/evaluated only after all update trajectories for a target are complete. optimize() accepts only target representation pieces and noise, never Clean observations/actions/outcome labels. All 18 configurations are reported, with no best-Clean-distance configuration selected for a production method.

## Nulls and checks

Five fixed random spatial suppressions per step/configuration, with identical learned spatial L2 norm, same nonpositive gauge/bounds. Match scalar block separately: a one-dimensional nonpositive scalar has no alternate equal-norm direction, so view-only random controls are identically the learned intervention. For joint controls, preserve learned scalar and randomize spatial component. This limitation prevents claiming learned scalar optimization beats a random-direction scalar control.

Zero-gate manual vs exact resampler controller parity <=1e-5. Native single-noise sampler vs policy.predict <=1e-5. Batched original-noise output also checked against recorded V1 controller actions (tolerance stated by measured parity). Null norm and constraints independently verified.

## Offline evaluation and exploratory screen

Evaluate steps 0/5/10 on four held-out z0: raw/relative flow variance, controller and clipped controller XYZ/rotation/all L2 distances to the exact native paired Clean outputs; controller magnitude ratios. L2 uses flattened 12-step chunks per noise, then averages noises. Rotation is coordinate L2, not an SO(3) geodesic; all-coordinate L2 mixes units. When initial Clean distance is zero, percent distance change is undefined and report absolute drift instead.

Prespecified material-change screen for each full configuration at step10: at least 6/8 selected failures have >=10% clipped XYZ distance reduction, mean reduction >=10%, learned direction beats median of five matched nulls on >=6/8, and all held-out clipped action-norm ratios across failure cases stay >=.8. This is an exploratory existence screen across 18 configurations, not a significance test or permission to select hyperparameters using Clean. Clean-control drift is reported separately and must be addressed before any closed-loop run. Passing only motivates a separately frozen calibrated experiment.

If the screen fails, do not declare that all curvature minimization is impossible. If curvature falls but reference distance does not improve or nulls match it, retain that specific shortcut/ineffectiveness evidence. Likewise, negative repair evidence does not validate a deployable curvature gate; independent Source calibration, successful-Shift false alarms and severity confounding remain unresolved.

No 15-minute timing guarantee: pilot records actual runtime and GPU memory before full deployment. No full100 closed-loop campaign, no blind unlock and no automatic learned manifold fallback.

Preflight correction: initial pilot used mean(delta^2); the requested squared-L2 sum is used for the full grid. An initial full launcher was stopped before any completed case, its directory/receipts retained separately. The pilot is a correctness/timing run, not evidence used in the final decision.

## Mechanism follow-up: post-LayerNorm Value trust

After the full original grid, all optimized view trusts remained 1. A separate fixed trust sweep (1,.75,.5,.25,.1) on the same four held-out z0 showed almost no output change. Architecture inspection found one DiT block: prefix keys/values are computed from per-token LayerNorm, which nearly cancels positive resampler-token scaling (apart from epsilon). This is evidence about intervention location, not proof that view reweighting cannot work.

A second, separately labelled full fixed grid preserves the original outputs and all settings. Only view trust location changes: multiply the first 48 agentview prefix **Values** by exp(s) AFTER prefix LayerNorm/projection; keep Keys, wrist Values, language/state and policy weights unchanged. Apply to every DiT prefix cache (one in this checkpoint). This attenuates the agentview contribution without changing softmax normalization or claiming a change in attention probabilities. Initialization s=0 is identity. Spatial manipulation remains resampler-logit suppression; joint combines it with Value trust. Same 18 configurations, same 10 selected cases, same 4 train and 4 held-out noises, same five nulls, same GO screen. No change based on Clean alignment or learning-rate search. This follow-up is adaptive development, not an independent replication.

A final diagnostic repeats the same already fixed trust levels (1,.75,.5,.25,.1) at the effective Value location, on the same 10 cases/noises. Purpose: distinguish view-attenuation benefit from the necessity of energy-based optimization, since equal-norm scalar nulls are degenerate. All levels are reported, with no best-reference-distance choice. This is a post-grid mechanism control, not a new GO criterion or a source-calibrated policy.
