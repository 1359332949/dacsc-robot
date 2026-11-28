# Frozen Spatial SOH v2 transfer with libero_10 Clean calibration

This is **not fully zero-shot**: the SOH weights and Spatial action-sensitivity
basis are unchanged, but new-suite init0 Clean trajectories calibrate the
moments and deep gate. Test-time parameters stay frozen. This one-seed run is
an exploratory transfer check, not a blind confirmation.

## Locked protocol

- SOH: server `runs/phase4_soh/soh_v2_ckpt.pt`, SHA256
  `b4fe1ac8c4d206fa13f445c93b351682a488170de8ac117fb9c799d7145a8ceb`.
- Native SmolVLA BF16 visual path, original CPU pixel normalization,
  H=50, E=1, Euler10, EGL, env.noise=0, policy seed0, EMA beta0.
- Vanilla LIBERO task/init assets; 520 policy steps plus 10 settling steps.
  Installed roboeval supplies only the Spatial adapter (default220).
  The new suite inherits its IO/reset and changes only task inventory/init/horizon.
- Horizon reference: OpenVLA official evaluation, revision
  `c8f03f48af692657d3060c19588038c7220e9af9`,
  `experiments/robot/libero/run_libero_eval.py`, libero_10 branch520;
  a verbatim source snapshot is retained in docs/data/phase4/.
- Calibration: all actual native Frozen frames from ten init0 Clean episodes,
  no success filtering. Layer3/6/9/12 collected for reuse of the existing
  reference schema; only Layer3 and12 are used by SOH.
- Each target task is excluded from its moment reference (outer9/inner8+1)
  and scalar gate calibration (held source fit8/channel threshold inner7+1).
- Evaluation: init1 Clean10 + bilateral Sensor20 severity3. Five families each
  four conditions; cyclic task schedule gives each of ten tasks two Sensor cases.
  No Lighting/Background/Blind evaluation, no weight training.
- Reject branch must be bit-exact Frozen; forced accept must have encoder4,
  one VLM prefill, ten Flow steps, finite actions, unchanged parameter versions.
- Paired trajectories use contemporaneous Frozen, identical initial physical
  state, bilateral pixels and Torch CPU/CUDA RNG prefix. Existing bounded RGB
  cache is reused only at identical physical/render state (not between diverged
  trajectories). All replacements remain logged.
- Full trajectories, source SHA256, checkpoint SHA256, per-task reference folds,
  per-case rescue/harm, exact McNemar p, task-cluster bootstrap and gate rates
  are retained. Output directory: runs/phase4_libero10_transfer_v1/.

Compatibility aliases under `runs/phase4_soh/` expose
`libero10_init0_calib.npz` and `libero10_source_gate.json`; their actual files
remain isolated in the run directory above.

## Execution

`bash scripts/run_phase4_libero10_transfer.sh` on the configured GPU server
runs calibration, fit, preflight, and sixty evaluation episodes in that order,
with four workers and resumable per-task/per-case outputs. Any failed phase
stops the driver and writes FAILED. A completed run writes COMPLETE.

The scalar gate is a per-frame empirical quantile, not a guarantee of 95%
episode safety. New-suite calibration changes SOH input fingerprints and
can therefore change its behavior even with all learned weights frozen.

The first campaign invoked the independent encoder-call and raw-data audits
separately from the original driver. The released driver now includes those
audits in sequence and creates/validates the code lock; inference code and
method parameters are unchanged.

Performance results are reported in PHASE4_CLOSEOUT.md after completion.

The release also corrects the older Spatial worker arm.policy_seed output
field. The original executed helper and raw records are preserved in the
evidence archive; actual seeds0/1/2 were independently verified from RNG states.
This metadata correction does not alter inference or historical outcomes.
