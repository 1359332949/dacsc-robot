import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest

p=argparse.ArgumentParser(); p.add_argument('--dir',type=Path,required=True); a=p.parse_args()
rows=[json.loads(f.read_text()) for f in sorted((a.dir/'pairs').glob('*.json'))]
assert len(rows)==20, len(rows)
def calc(sub):
    n=len(sub); f=[bool(r['arms']['frozen']['success']) for r in sub]; g=[bool(r['arms']['soh']['success']) for r in sub]
    rescue=sum((not x) and y for x,y in zip(f,g)); harm=sum(x and not y for x,y in zip(f,g))
    return dict(n=n, frozen=sum(f), soh=sum(g), rescued=rescue, harmed=harm, net=rescue-harm,
                p=float(binomtest(rescue, rescue+harm, .5).pvalue) if rescue+harm else 1.0,
                frozen_rate=sum(f)/n, soh_rate=sum(g)/n)
allstats=calc(rows)
subs={}
for sg in sorted({r['case'].get('subgroup','?') for r in rows}):
    subs[sg]=calc([r for r in rows if r['case'].get('subgroup')==sg])
gate=[]
for r in rows:
    tr=r['arms']['soh']['trace']; acc=sum(x['gate']['accepted'] for x in tr)
    gate.append((acc,len(tr)))
summary=dict(n_pairs=20, overall=allstats, subgroups=subs,
             gate_accepted=int(sum(x for x,_ in gate)), gate_total=int(sum(y for _,y in gate)),
             mean_episode_accept_rate=float(np.mean([x/y for x,y in gate])),
             scope='Sensor-only 20 edbac60 conditions; Frozen vs SOH; 220 steps; paired state/RNG; not a full 100-condition campaign')
(a.dir/'sensor_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# SOH Sensor-only paired closed loop','',f"20 Sensor conditions, {2*len(rows)} episodes.",'',
       '| arm | success | rate |','|---|---:|---:|',
       f"| Frozen | {allstats['frozen']}/20 | {100*allstats['frozen_rate']:.1f}% |",
       f"| SOH | {allstats['soh']}/20 | {100*allstats['soh_rate']:.1f}% |",'',
       f"rescued/harmed = {allstats['rescued']}/{allstats['harmed']}, net {allstats['net']:+d}, exact p={allstats['p']:.6g}",'',
       '| subgroup | n | Frozen | SOH | net |','|---|---:|---:|---:|---:|']
for sg,x in subs.items(): lines.append(f"| {sg} | {x['n']} | {x['frozen']}/{x['n']} | {x['soh']}/{x['n']} | {x['net']:+d} |")
lines += ['', f"gate accepted replans {summary['gate_accepted']}/{summary['gate_total']}; mean episode accept rate {100*summary['mean_episode_accept_rate']:.2f}%",'']
(a.dir/'PHASE4_SOH_SENSOR_CL.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
