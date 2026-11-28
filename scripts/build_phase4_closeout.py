"""Generate the Phase4 closeout from committed machine-readable summaries."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/data/phase4'
load=lambda name:json.loads((DATA/name).read_text())
fmt_ci=lambda bounds:f'[{bounds[0]:+.1f}, {bounds[1]:+.1f}]'
main=load('soh_v2_3seed_summary.json')
beta1=load('soh_v2_diag_t0_summary.json')['groups']
beta0=load('soh_v2_beta0_diag_summary.json')['groups']
transfer=load('libero10_transfer/summary.json')
audit=load('libero10_transfer/independent_audit.json')
legacy=load('libero10_transfer/spatial_beta0_audit.json')
assert legacy['status']=='PASS'
assert audit['status']=='PASS'
lines=['# Phase 4 evidence closeout','',
    'SOH v2 is an offline-trained residual with a frozen SmolVLA backbone and frozen inference-time parameters. It is distinct from the original training-free moment-alignment V1.','',
    '## Spatial development: three-seed paired evaluation','',
    '| Group | Frozen | SOH v2 | Rescue / harm | Delta pp | Exact p | Task-cluster 95% CI pp |',
    '|---|---:|---:|---:|---:|---:|---|']
for name,x in main['groups'].items():
    lines.append(f"| {name} | {x['frozen']}/{x['n']} | {x['soh']}/{x['n']} | {x['rescued']}/{x['harmed']} | {100*x['net']/x['n']:+.2f} | {x['p']:.6g} | {fmt_ci(x['ci'])} |")
lines+=['', '| Seed | Sensor Frozen → SOH | Sensor net | Background Frozen → SOH | Background net |', '|---|---:|---:|---:|---:|']
for seed,groups in legacy['per_seed'].items():
    a,b=groups['Sensor'],groups['Background']
    lines.append(f"| {seed} | {a['frozen']}/{a['n']} → {a['soh']}/{a['n']} | {a['net']:+d} | {b['frozen']}/{b['n']} → {b['soh']}/{b['n']} | {b['net']:+d} |")
lines+=['',
    'Sensor is positive in the paired episode test, but the task-cluster interval crosses zero. Background is unresolved. The SOH residual and sensitivity basis were trained/estimated on Spatial init0 tasks; reference statistics are task-LOTO, but this is not an entirely task-held-out adapter-training result. This table contains no three-seed Clean or Lighting results and is not untouched confirmation.','',
    '## Beta-zero diagnostic (t=0 only)','',
    '| Group | Beta 1 pooled XYZ error reduction % | Beta 0 pooled XYZ error reduction % |',
    '|---|---:|---:|']
for name in ('Sensor','Background'):
    a=beta1[name]['soh_pooled_reduction_pct']['xyz'];b=beta0[name]['soh_pooled_reduction_pct']['xyz']
    lines.append(f'| {name} | {a:+.2f} | {b:+.2f} |')
lines+=['',
    'These are distances in policy action components relative to the same-state Clean Frozen action, not measured EEF pose errors, success rates or a closed-loop beta ablation. Beta-zero was a separately trained checkpoint; its training command/receipt are retained. The previously uncommitted --damping-beta training option is now included, with the same default 1 and an exact hash match to the server implementation.','',
    '## libero_10 transfer with new-suite Clean calibration','',
    'Spatial SOH weights and the source action-sensitivity basis remain frozen. New-suite init0 Clean trajectories calibrate moments and the deep gate under task LOTO; init1 evaluates Clean10 and bilateral Sensor20 severity3, seed0. This is calibrated transfer, not fully zero-shot. The native BF16/CPU preprocessing path is unchanged. The horizon is 520 policy steps plus 10 settling steps.','',
    '| Group | Frozen | SOH v2 | Rescue / harm | Delta pp | Exact p | Task-cluster 95% CI pp | Gate accept % |',
    '|---|---:|---:|---:|---:|---:|---|---:|']
for name,x in transfer['groups'].items():
    lines.append(f"| {name} | {x['frozen']}/{x['n']} | {x['deepgate']}/{x['n']} | {x['rescued']}/{x['harmed']} | {x['delta_pp']:+.2f} | {x['mcnemar_exact_two_sided']:.6g} | {fmt_ci(x['task_cluster_bootstrap95_pp'])} | {100*x['gate']['micro_rate']:.2f} |")
lines+=['', '| Sensor family | Frozen | SOH | Rescue / harm | Gate accept % |', '|---|---:|---:|---:|---:|']
for name,x in transfer['subgroups'].items():
    lines.append(f"| {name} | {x['frozen']}/{x['n']} | {x['deepgate']}/{x['n']} | {x['rescued']}/{x['harmed']} | {100*x['gate']['micro_rate']:.2f} |")
n=transfer['calibration']['source_clean_success']
lines+=['',f'Source init0 native Frozen Clean: {n}/10; all ten full trajectories, including failures, entered the reference collection. Each evaluated task is excluded from its fitted reference.','',
    'If all observed paired differences are zero, resampling those differences gives a degenerate [0,0] bootstrap interval. This is not an equivalence test or proof of zero population harm, especially in these small cohorts.', '',
    'The run is a single-seed exploratory transfer check. Cross-suite transfer refers to the SOH residual and sensitivity basis, not a claim that the frozen SmolVLA backbone never saw libero_10 during its original training; its pinned model card does not specify the training dataset. There is no new-suite Lighting/Background result, no direct contemporaneous DeepGate third arm, and no universal geometry guarantee. A per-frame 95th-percentile gate is not an episode-level safety guarantee.','',
    '## Audit and provenance','',
    f"Independent raw audit: {audit['cases']} pairs, {audit['replans']} replans, {audit['same_input_off_action_checks']} equal-input rejected-gate action checks; status {audit['status']}.",
    f"Bounded paired RGB replacements: {audit['rounding_events']} events, maximum {audit['max_rounding_pixels']} pixels/view and {audit['max_rounding_lsb']} uint8 LSB. Reuse is restricted to identical physical/render state.",
    f"SOH checkpoint SHA256: `{audit['soh_sha256']}`.",
    f"libero_10 calibration SHA256: `{audit['calib_sha256']}`.",
    'The retained Spatial records hard-coded arm.policy_seed=0 in the reused base rollout. Independent comparison against the actual Torch CPU/CUDA RNG confirms that seeds 0, 1, 2 really executed different streams, with paired streams equal within each condition. The original records are unchanged; the release corrects future Spatial arm seed metadata. The archived runtime source hashes refer to the original executed files.',
    'The current transfer summary retains the legacy internal count key deepgate for its SOH arm; the exported table correctly labels it SOH v2.',
    'Full trajectories are kept in the separate evidence archive. Summary, preflight, source folds, source gate, runtime configuration and code hashes are retained beside this report.',
    '', '## Appropriate claim', '',
    f"Spatial provides an encouraging Sensor result across three seeds, with uncertain task-level generalization and unresolved Background. The current libero_10 pilot does not establish a stable cross-suite improvement: Frozen is at the Sensor floor and {transfer['groups']['Sensor']['rescued']} rescue(s) were observed. Gate triggering alone does not guarantee task recovery. The next evidence gap is native baseline validation and a prespecified evaluation with enough non-floor cases, before further SOH parameter changes. The result must not be replaced by the t=0 diagnostic or described as fully zero-shot."]
(ROOT/'docs/PHASE4_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
print('WROTE docs/PHASE4_CLOSEOUT.md')
