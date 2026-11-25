import json,glob,collections,random,math
from pathlib import Path
from scipy.stats import binomtest
T=Path('/hy-tmp/patch-diag/runs/phase2/libero10/trials'); ARMS=['frozen','pixel_align','deep_block12','actmad_true','ours_v1']
rows=[]
for f in glob.glob(str(T/'*.json')):
 for r in json.load(open(f)): rows.append(r)
by=collections.defaultdict(dict)
for r in rows: by[r['case_id']][r['arm']]=r
print('conditions',len(by),'rows',len(rows))
def groupof(cid):
 r=next(iter(by[cid].values())); g=r['group']
 if cid.startswith('sensor_'):
  parts=cid.split('_'); return 'sensor_'+'_'.join(parts[2:-1])
 if cid.startswith('bg_'):
  parts=cid.split('_'); return 'background_table' if int(parts[2])<5 else 'background_scene'
 return g
def members(g):
 if g=='total': return list(by)
 if g=='sensor': return [k for k in by if groupof(k).startswith('sensor')]
 if g=='background': return [k for k in by if groupof(k).startswith('background')]
 if g=='background_table': return [k for k in by if groupof(k)=='background_table']
 if g=='background_scene': return [k for k in by if groupof(k)=='background_scene']
 return [k for k in by if groupof(k)==g]

G=['clean','sensor','sensor_motion_blur','sensor_gaussian_blur','sensor_zoom_blur','sensor_fog','sensor_glass_blur','lighting','background','background_table','background_scene','total']
out={}
for g in G:
 ks=members(g); d={}
 for a in ARMS:
  n=sum(1 for k in ks if a in by[k]); s=sum(1 for k in ks if a in by[k] and by[k][a]['success']); d[a]={'success':s,'n':n}
 def net(base,arm):
  b=c=0
  for k in ks:
   if base not in by[k] or arm not in by[k]: continue
   x=by[k][base]['success']; y=by[k][arm]['success']; b+= (not x and y); c+= (x and not y)
  p=1.0 if b+c==0 else min(1.0,2*binomtest(min(b,c),b+c,0.5).pvalue)
  return b,c,b-c,p
 d['_vs_frozen']={a:net('frozen',a) for a in ARMS[1:]}; d['_vs_actmad']={a:net('actmad_true',a) for a in ARMS if a!='actmad_true'}
 out[g]=d
 print('===',g,{a:(d[a]['success'],d[a]['n']) for a in ARMS})
 for a in ARMS[1:]: print(a,'vsF',d['_vs_frozen'][a])
# cluster bootstrap total
rng=random.Random(20261125); tasks=sorted({next(iter(by[k].values()))['base_task'] for k in by})
for g in ['clean','sensor','lighting','background','total']:
 ks=members(g)
 for arm in ARMS[1:]:
  per=[]
  for t in tasks:
   vals=[int(by[k][arm]['success'])-int(by[k]['frozen']['success']) for k in ks if next(iter(by[k].values()))['base_task']==t]
   if vals: per.append(sum(vals)/len(vals))
  boots=[]
  for _ in range(10000): boots.append(sum(per[rng.randrange(len(per))] for _ in per)/len(per))
  boots.sort(); out[g][arm]['ci95_vs_frozen']=[boots[250],boots[9750]]
json.dump(out,open('/hy-tmp/patch-diag/runs/phase2/libero10/summary.json','w'),indent=2)
# markdown
L=['# Phase 2A libero_10 5-arm table','','## Success','','| Group | n | Frozen | Pixel | Deep B12 | ActMAD | Ours V1 |','|---|---:|---:|---:|---:|---:|---:|']
for g in G:
 d=out[g]; n=d['frozen']['n']; L.append('| '+g+' | '+str(n)+' | '+' | '.join(f"{d[a]['success']}/{d[a]['n']}" for a in ARMS)+' |')
L+=['','## vs Frozen (rescued/harmed, exact McNemar p)','','| Group | Pixel | Deep B12 | ActMAD | Ours V1 |','|---|---:|---:|---:|---:|']
for g in G:
 d=out[g]; L.append('| '+g+' | '+' | '.join(f"{d['_vs_frozen'][a][2]:+d} ({d['_vs_frozen'][a][3]:.3g})" for a in ARMS[1:])+' |')
L+=['','## vs ActMAD (rescued/harmed, exact McNemar p)','','| Group | Frozen | Pixel | Deep B12 | Ours V1 |','|---|---:|---:|---:|---:|']
for g in G:
 d=out[g]; L.append('| '+g+' | '+' | '.join(f"{d['_vs_actmad'][a][2]:+d} ({d['_vs_actmad'][a][3]:.3g})" for a in ['frozen','pixel_align','deep_block12','ours_v1'])+' |')
open('/hy-tmp/patch-diag/runs/phase2/libero10/summary.md','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
