import json,glob,os
ref={r['case_id']:r['success'] for r in json.load(open('/hy-tmp/patch-diag/docs/data/static_visual_v2_suite/final/dev_verified.json'))['results']}
for b in ['b40','b60']:
 files=sorted(glob.glob(f'/hy-tmp/patch-diag/runs/spatial_shift_multi_{b}_shard*.json')); rows=[]
 for f in files: rows+=json.load(open(f))['rows']
 ids=[r['case_id'] for r in rows]; print(b,'files',len(files),'n',len(rows),'unique',len(set(ids)),'dups',len(ids)-len(set(ids)))
 print(' factors',{f:sum(1 for r in rows if r['factor']==f) for f in ['sensor_noise','background','lighting']})
 print(' success',sum(r['success'] for r in rows))
 out={'schema':'spatial_patch_bias_v2_merged','rows':rows,'n':len(rows)}
 json.dump(out,open(f'/hy-tmp/patch-diag/runs/spatial_shift_multi_{b}_merged.json','w'),indent=2)
