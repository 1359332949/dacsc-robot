# Reproducing this paired development experiment

The wrist stress deliberately changes which *raw sensor pixels* receive the benchmark kernel; the model, camera masks and source views are not relabelled. No third-party camera-fault classifier, Clean twin, sample labels or parameter update in policy weights. The square penalty is summed over the two log coefficients, so it equals the original scalar penalty on feasible points. Projected two-dimensional Adam is NOT the same algorithm as enumerating and fitting one model per eye. Ties are not broken in favor of agentview. Extra validation searches after observing failure are not permitted in this experiment.

The server entry point is `scripts/run_egl_symmetry.sh`; dependencies, protected manifest SHA and EGL environment are the same as in `5795eb8`. `evidence.tar.xz` contains all 1260 individual closed-loop receipts. To reproduce the summary after unpacking this archive into a new directory:

```sh
D=docs/data/egl_trust_symmetry
python scripts/summarize_egl_trust_symmetry.py --raw WORK --manifest docs/data/static_visual_v2_suite/final/dev_manifest.json --previous docs/data/egl_low_dof/trust_three_seeds.json --out WORK/summary.json
python scripts/report_egl_trust_symmetry.py WORK/summary.json WORK/report.md
```

Gate acceptance for the fixed intervention is **defined by the saved continuous proposal**. The fixed action is deliberately not checked against a separate gate: doing so would confound selection with correction magnitude. Actual independent rollouts are executed for all arms even when the gate rejects. Rejected branches must equal fresh Frozen in complete executed-action hash, steps and outcome. `steps` denotes actual controller commands, not flow integration steps. Paired exact p-values and intervals are descriptive on reused Dev data; seed repetitions do not triple the number of independent initial scenes or visual assets. The stress difference is in rendered observations only; nominal simulation conditions, kernel strength and RNG namespace stay paired across arms.
