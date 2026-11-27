import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
p=argparse.ArgumentParser(); p.add_argument('--dir',type=Path,required=True); a=p.parse_args()
rows=[json.loads(f.read_text()) for f in sorted((a.dir/'pairs').glob('*.json'))]
def calc(sub):
    n=len(sub); f=np.array([r['arms']['frozen']['success'] for r in sub],bool); g=np.array([r['arms']['soh']['success'] for r in sub],bool)
    res=int((~f&g).sum()); harm=int((f&~g).sum())
    tasks=sorted({r['case']['task_id'] for r in sub}); nums=[]; dens=[]
    for t in tasks:
        rr=[r for r in sub if r['case']['task_id']==t]
        nums.append(sum(int(r['arms']['soh']['success'])-int(r['arms']['frozen']['success']) for r in rr))
        dens.append(len(rr))
    nums=np.array(nums); dens=np.array(dens); rng=np.random.default_rng(2026112801)
    draws=rng.integers(0,len(tasks),size=(10000,len(tasks)))
    eff=100*nums[draws].sum(1)/dens[draws].sum(1)
    acc=sum(sum(t['gate']['accepted'] for t in r['arms']['soh']['trace']) for r in sub); tot=sum(len(r['arms']['soh']['trace']) for r in sub)
    return dict(n=n,frozen=int(f.sum()),soh=int(g.sum()),rescued=res,harmed=harm,net=res-harm,
        p=float(binomtest(res,res+harm,.5).pvalue) if res+harm else 1.0,
        ci=[float(x) for x in np.percentile(eff,[2.5,97.5])], gate_accepted=acc, gate_total=tot)
summary=dict(pairs=len(rows), groups={}, subgroups={})
for g in ('Sensor','Background'):
    sub=[r for r in rows if r['case']['group']==g]; summary['groups'][g]=calc(sub)
    summary['subgroups'][g]={sg:calc([r for r in sub if r['case'].get('subgroup')==sg]) for sg in sorted({r['case'].get('subgroup') for r in sub})}
(a.dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Clean5 closed-loop seed 0','',f'pairs: {len(rows)}','','| group | n | Frozen | SOH | rescued/harmed | net | p | task-cluster 95% CI pp |','|---|---:|---:|---:|---:|---:|---:|---:|']
for g,x in summary['groups'].items():
    lines.append(f"| {g} | {x['n']} | {x['frozen']}/{x['n']} | {x['soh']}/{x['n']} | {x['rescued']}/{x['harmed']} | {x['net']:+d} | {x['p']:.6g} | [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}] |")
for g,subs in summary['subgroups'].items():
    lines += ['',f'## {g} subgroups','','| subgroup | n | Frozen | SOH | net | p |','|---|---:|---:|---:|---:|---:|']
    for sg,x in subs.items(): lines.append(f"| {sg} | {x['n']} | {x['frozen']}/{x['n']} | {x['soh']}/{x['n']} | {x['net']:+d} | {x['p']:.4g} |")
(a.dir/'PHASE4_SOH_CLEAN5_EVAL.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
