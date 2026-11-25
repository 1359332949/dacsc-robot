import json,glob,collections,random
T='/hy-tmp/patch-diag/runs/w2_ab_seed0/trials'; ARMS=['frozen','v1_hard075','w2_proj_rms_noclamp','js_shrink_rms_noclamp']
rows=[]
for f in glob.glob(T+'/*.json'):
 for r in json.load(open(f)): rows.append(r)
print('files',len(glob.glob(T+'/*.json')),'rows',len(rows))
by=collections.defaultdict(dict)
for r in rows: by[(r['case_id'],r['seed'])][r['arm']]=r
def group(r):
 return r['group']
def member(k,g):
 r=next(iter(by[k].values())); x=group(r)
 if g=='total100': return True
 if g=='background': return x.startswith('background')
 return x==g
out={}
for g in ['clean','sensor_bilateral','background','background_table_texture','background_scene_texture','lighting','total100']:
 d={}
 for arm in ARMS:
  n=sum(1 for k,v in by.items() if member(k,g) and arm in v); s=sum(1 for k,v in by.items() if member(k,g) and arm in v and v[arm]['success']); d[arm]={'success':s,'n':n}
 def net(base,arm):
  saved=lost=0
  for k,v in by.items():
   if not member(k,g) or base not in v or arm not in v: continue
   b=v[base]['success']; a=v[arm]['success']; saved+=(not b and a); lost+=(b and not a)
  return saved,lost,saved-lost
 d['_vs_frozen']={a:net('frozen',a) for a in ARMS[1:]}; d['_vs_v1']={a:net('v1_hard075',a) for a in ARMS[2:]}; out[g]=d
 print('===',g,{a:(d[a]['success'],d[a]['n']) for a in ARMS})
 for a in ARMS[1:]: print(a,'vsF',d['_vs_frozen'][a],'vsV1',d['_vs_v1'].get(a))
# cluster bootstrap total
rng=random.Random(20261125); tasks=sorted({next(iter(by[k].values()))['base_task'] for k in by})
for g in ['total100','clean','sensor_bilateral','background','lighting']:
 for arm in ARMS[1:]:
  per=[]
  for t in tasks:
   vals=[]
   for k,v in by.items():
    if not member(k,g) or next(iter(v.values()))['base_task']!=t or arm not in v: continue
    base='v1_hard075' if arm!='v1_hard075' else 'frozen'
    if base in v: vals.append(int(v[arm]['success'])-int(v[base]['success']))
   if vals: per.append(sum(vals)/len(vals))
  boots=[]
  for _ in range(10000): boots.append(sum(per[rng.randrange(len(per))] for _ in per)/len(per))
  boots.sort(); out[g][arm]['ci95_vs_frozen' if arm!='v1_hard075' else 'ci95_vs_frozen']=[boots[250],boots[9750]]
json.dump(out,open('/hy-tmp/patch-diag/runs/w2_ab_seed0/summary.json','w'),indent=2)
