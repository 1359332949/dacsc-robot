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
    def pooled(errs, bases):
        num=sum(e for e in errs if e is not None); den=sum(b for b in bases if b is not None)
        return 100.0*(1.0-num/den) if den>0 else None
    def mean(vals):
        vals=[v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    soh_err=[r['soh']['errors'] for r in grp]
    base_err=[r['baseline_errors'] for r in grp]
    dg_err=[r['methods']['deep_gate_L3_a0.75']['errors'] for r in grp]
    groups[g]=dict(n=len(grp), soh_accepted=sum(bool(r['soh']['accepted']) for r in grp),
        soh_pooled_reduction_pct={k:pooled([r[k] for r in soh_err],[r[k] for r in base_err]) for k in ('xyz','rotation','action7')},
        soh_mean_case_reduction_pct={k:mean([100.0*(1.0-e[k]/b[k]) for e,b in zip(soh_err,base_err) if b[k]>0]) for k in ('xyz','rotation','action7')},
        deepgate_pooled_reduction_pct={k:pooled([r[k] for r in dg_err],[r[k] for r in base_err]) for k in ('xyz','rotation','action7')},
        deepgate_mean_case_reduction_pct={k:mean([100.0*(1.0-e[k]/b[k]) for e,b in zip(dg_err,base_err) if b[k]>0]) for k in ('xyz','rotation','action7')},
        frozen_mean_error={k:mean([r[k] for r in base_err]) for k in ('xyz','rotation','action7')})
summary=dict(schema='phase4_soh_diag_summary_v1',groups=groups,scope='t=0 only; 50 previously accessed development conditions; not a success rate; no closed loop')
(a.dir/'soh_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# SOH t=0 diagnostic','','| group | n | accepted | SOH xyz pooled% | SOH xyz mean% | SOH rot pooled% | SOH rot mean% | DeepGate xyz pooled% | DeepGate xyz mean% |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
def fmt(v):
    return 'N/A' if v is None else f'{v:+.2f}'
for g,x in groups.items():
    lines.append(f"| {g} | {x['n']} | {x['soh_accepted']} | {fmt(x['soh_pooled_reduction_pct']['xyz'])} | {fmt(x['soh_mean_case_reduction_pct']['xyz'])} | {fmt(x['soh_pooled_reduction_pct']['rotation'])} | {fmt(x['soh_mean_case_reduction_pct']['rotation'])} | {fmt(x['deepgate_pooled_reduction_pct']['xyz'])} | {fmt(x['deepgate_mean_case_reduction_pct']['xyz'])} |")
lines += ['', 'Negative values mean worse than Frozen at the same t=0 state.','']
(a.dir/'PHASE4_SOH_DIAG.md').write_text('\n'.join(lines))
print(json.dumps(groups,indent=2))
