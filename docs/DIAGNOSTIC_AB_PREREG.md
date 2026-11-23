# Diagnostic A/B pre-registration

## Fixed facts

- H=E=12, N=10 Euler steps, action dim 7.
- Target-dev states: 34,35,36,37,39,43,44,45,48,49.
- Baseline double peak:
  - immune: 36,39,49 -> Shift 9/9
  - wipeout: 34,35,43,45 -> Shift 0/12
  - edge: 37,44,48 -> Shift 2/9
- All diagnostics are privileged target-dev exploratory diagnostics, not deployable TTA.
- Do not use target-test.
- Do not use Clean twin as a deployment input.
- All state statistics use initial state as the cluster.

## Diagnostic A: t=0 z0 sensitivity

Script: `scripts/diagnostic_a_z0_sensitivity.py`.

For each state:
1. Fix the initial observation after settling.
2. Sample K=8 z0 from fixed seeds.
3. Run frozen Shift CoRP for each z0.
4. Measure:
   - per-action std of XYZ, rotation, gripper;
   - pairwise RMS across z0;
   - gripper sign flip rate.

Primary question:
> Does the wipeout group have significantly higher action dispersion / gripper
> ambiguity at t=0 than the immune group and Clean?

Decision:
- If yes: candidate zero-training selective-abstention signal.
- If no: reject global t=0 sensitivity as discriminator.

## Diagnostic B: chunk-boundary physical anomaly

Only after Gate1 reproducibility is fixed.

Collect Frozen CoRP rollouts with H=E=12, 3 chunks, Clean and Shift, same state/seed/z0 schedule.
Record:
- executed clipped actions;
- executed steps;
- state8 before/after;
- per-view Resampler tokens;
- command displacement vs actual state8 displacement residual;
- action stagnation / sign reversal across chunks;
- agentview/eye token change ratio.

Primary question:
> Do low-dimensional target-only physical/control anomalies separate the wipeout
> and immune groups before task failure?

Decision:
- If yes: use as a candidate abstention feature; no residual regression yet.
- If no: stop appearance-based global TTA for this benchmark.

## Gate 1 repro prerequisites

- max_steps=220, matching P0 ledger;
- fixed renderer/GPU;
- per-step action/state8 hashes;
- 30/30 shift_success agreement with P0 before using 7/3/3;
- if not, report Gate1 as non-reproducible, not as a recovery rate.

## Hard stop conditions

- Any diagnostic using target-test / target success / Clean at deployment: invalid.
- Any correction trained on target-dev oracle residual: privileged diagnostic only.
- Direct 7D flow-residual regression remains stopped:
  - leakage-fixed Constant G_agg(all)=0.058 beats learned models;
  - unseen-7 learned G_row all <= 0.
