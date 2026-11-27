"""Summarize completed SOH diagnostic case JSON without rerunning forward passes."""
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--dir',type=Path,required=True); a=p.parse_args()
rows=[json.loads(f.read_text()) for f in sorted((a.dir/'cases').glob('*.json'))]
assert len(rows)==50, len(rows)
groups={}
for g in ('Clean','Lighting','Background','Sensor'):
    grp=[r for r in rows if r['case']['group']==g]
    def mean(vals):
        vals=[v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    groups[g]=dict(n=len(grp), soh_accepted=sum(bool(r['soh']['accepted']) for r in grp),
        soh_mean_reduction_pct={k:mean([r['soh']['reduction_vs_frozen_pct'][k] for r in grp]) for k in ('xyz','rotation','action7')},
        deepgate_mean_reduction_pct={k:mean([100.0*(1-r['methods']['deep_gate_L3_a0.75']['errors'][k]/r['baseline_errors'][k]) for r in grp if r['baseline_errors'][k]>0]) for k in ('xyz','rotation','action7')},
        frozen_mean_error={k:mean([r['baseline_errors'][k] for r in grp]) for k in ('xyz','rotation','action7')})
summary=dict(schema='phase4_soh_diag_summary_v1',groups=groups,scope='t=0 only; 50 previously accessed development conditions; not a success rate; no closed loop')
(a.dir/'soh_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# SOH t=0 diagnostic','','| group | n | accepted | SOH xyz red% | SOH rot red% | DeepGate xyz red% | DeepGate rot red% |','|---|---:|---:|---:|---:|---:|---:|']
def fmt(v):
    return 'N/A' if v is None else f'{v:+.2f}'
for g,x in groups.items():
    lines.append(f"| {g} | {x['n']} | {x['soh_accepted']} | {fmt(x['soh_mean_reduction_pct']['xyz'])} | {fmt(x['soh_mean_reduction_pct']['rotation'])} | {fmt(x['deepgate_mean_reduction_pct']['xyz'])} | {fmt(x['deepgate_mean_reduction_pct']['rotation'])} |")
lines += ['', 'Negative values mean worse than Frozen at the same t=0 state.','']
(a.dir/'PHASE4_SOH_DIAG.md').write_text('\n'.join(lines))
print(json.dumps(groups,indent=2))
