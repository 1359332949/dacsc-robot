import json,glob
rows=[]
for f in glob.glob('/hy-tmp/patch-diag/runs/channel_v2_shift_p10g03_s*.json'): rows+=json.load(open(f))['rows']
ids=[r['case_id'] for r in rows]; print('n',len(rows),'unique',len(set(ids)),'dups',len(ids)-len(set(ids)))
print('factors',{f:sum(1 for r in rows if r['factor']==f) for f in ['sensor_noise','background','lighting']})
ref={r['case_id']:r['success'] for r in json.load(open('/hy-tmp/patch-diag/docs/data/static_visual_v2_suite/final/dev_verified.json'))['results']}
print('total',sum(r['success'] for r in rows))
for f in ['sensor_noise','background','lighting']:
 sub=[r for r in rows if r['factor']==f]; res=sum(r['success'] and not ref[r['case_id']] for r in sub); harm=sum((not r['success']) and ref[r['case_id']] for r in sub); print(f,len(sub),sum(r['success'] for r in sub),'res',res,'harm',harm,'net',res-harm)
json.dump({'schema':'channel_v2_merged','rows':rows},open('/hy-tmp/patch-diag/runs/channel_v2_shift_p10g03_merged.json','w'),indent=2)
