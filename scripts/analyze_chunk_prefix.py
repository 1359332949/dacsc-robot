import json,math,glob,collections
files={'sensor_noise':'/tmp/chunk_anomaly_sensor_noise.json','background':'/tmp/chunk_anomaly_background.json','lighting':'/tmp/chunk_anomaly_lighting.json'}
def feats(r,K):
 ch=r['chunks'][:K]; ds=[math.dist(c['state8_start'][:3],c['state8_end'][:3]) for c in ch]
 cmd=[0.05*math.dist([0,0,0],c['cmd_xyz_sum']) for c in ch]
 rev=sum(1 for i in range(max(0,len(ch)-1)) if sum(ch[i]['cmd_xyz_sum'][j]*ch[i+1]['cmd_xyz_sum'][j] for j in range(3))<0)
 return {'mean_disp':sum(ds)/len(ds) if ds else 0,'mean_cmd':sum(cmd)/len(cmd) if cmd else 0,'rev':rev,'nchunks':len(ch)}
def auc_lower(pos,neg):
 # P(pos<neg), pos=fail, neg=succ; ties .5
 if not pos or not neg:return None
 s=sum(1 if p<n else (.5 if p==n else 0) for p in pos for n in neg)
 return s/(len(pos)*len(neg))
def paired(rows,key):
 # within base_task, all fail-success pairs, count fail<succ
 cnt=0; tot=0
 by=collections.defaultdict(list)
 for r,x in rows: by[r['base_task']].append((r,x))
 for task,v in by.items():
  for rf,xf in v:
   if rf['success']:continue
   for rs,xs in v:
    if not rs['success']:continue
    tot+=1; cnt+= xf[key] < xs[key]
 return cnt,tot
for K in (1,3,5):
 print('\n=== K',K)
 for f,path in files.items():
  rows=[(r,feats(r,K)) for r in json.load(open(path))['rows']]
  print(f,'n',len(rows))
  for key in ('mean_disp','mean_cmd','rev'):
   pos=[x[key] for r,x in rows if not r['success']]; neg=[x[key] for r,x in rows if r['success']]
   c,t=paired(rows,key); print(' ',key,'AUC_lower_fail',round(auc_lower(pos,neg),3),'paired_fail<succ',f'{c}/{t}',round(c/t,3) if t else None)
