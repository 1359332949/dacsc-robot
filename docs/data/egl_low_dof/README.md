# Reproduce the low-DOF Dev summaries

The `seed0_evidence.tar.gz` and `repeat_evidence.tar.gz` files contain individual rollout receipts, per-chunk plan, image and state hashes, per-episode success and execution length, EGL Clean moments, learned trust values and the 10-step gate fit history. `preflight.tar.gz` contains the superseded pilot directories (not counted). See `protocol.md` for the matching protocol and reasons it changed during pilot checks. Historical EGL controls lack full action digests; use actual same-campaign Frozen for comparisons.

From the repo root (replace `WORK` with a new local directory):

```sh
D=docs/data/egl_low_dof
mkdir -p WORK
tar -xzf "$D/seed0_evidence.tar.gz" -C WORK
tar -xzf "$D/repeat_evidence.tar.gz" -C WORK
python scripts/check_low_dof_pilot.py WORK/egl_low_dof_04
python scripts/summarize_egl_low_dof.py --raw WORK/egl_low_dof_04 --baseline docs/data/egl_baseline_seed0.json --out WORK/seed0_summary.json
for seed in 1 2; do
  python scripts/summarize_egl_low_dof.py --raw WORK/egl_low_dof_repeat_01 --baseline docs/data/egl_baseline_seed${seed}.json --out WORK/seed${seed}_summary.json --seed "$seed" --methods trust
done
python scripts/aggregate_egl_low_dof_seeds.py WORK/egl_low_dof_04 WORK/egl_low_dof_repeat_01 WORK/egl_low_dof_repeat_01 --out WORK/trust_three_seeds.json
python scripts/report_egl_low_dof_repeats.py WORK WORK/report.md
```

Only `trust` qualified for follow-up. Three policy seeds do not yield 300 independent instances: reset states, the candidate Dev list and gate-noise derivation are shared. Clean is an unshifted condition under a fixed task/start, not a guarantee of population Clean retention.
