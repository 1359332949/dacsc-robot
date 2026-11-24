"""Fixed Lighting pilot repeated-reset audit; run from deployed project root. No rollout."""
import frozen_egl_baseline as b, numpy as np, torch, random,ctypes,json,hashlib
from pathlib import Path
from collections import Counter
c=next(x for x in json.load(open('docs/data/static_visual_v2_suite/final/dev_manifest.json'))['cases'] if x['case_id'].startswith('260f'));seeds=json.load(open('docs/data/static_visual_v2_suite/final/results/'+c['case_id']+'.json'))['seeds'];seed=seeds['environment_seed'];orig=None;rows=[]
for i in range(10):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);ctypes.CDLL(None).srand(ctypes.c_uint(seed))
 e=b.OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=0)
 try:
  e.noise=0;e.seed(seed);e.reset();obs=e.set_init_state(torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],weights_only=False)[0]);
  for k in range(10):obs=e.step(b.DUMMY_ACTION)[0]
  im=obs['agentview_image'].copy();orig=im if orig is None else orig;d=abs(im.astype(int)-orig.astype(int));rows.append(dict(hash=b.array_hash(im),max=int(d.max()),mean=float(d.mean()),pixels=int((d.max(-1)>0).sum()),physics=b.array_hash(e.sim.get_state().flatten()),locations=np.argwhere(d.max(-1)>0)[:10].tolist()));print(rows[-1],flush=True)
 finally:e.close()
json.dump(rows,open('runs/low_dof_frame_forensics.json','w'))
