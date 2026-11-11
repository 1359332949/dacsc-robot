import glob, json
import numpy as np
from scipy.stats import binomtest

def load(pat):
    d={}
    for f in glob.glob(pat, recursive=True):
        for line in open(f):
            if line.strip():
                r=json.loads(line); d[(r["suite"],r["task_id"],r["episode_id"])]=bool(r["success"])
    return d

base=load("/hy-tmp/validation/lpfull_*/**/*.jsonl")
for tag in ("asub_cam","asub_l3","rv_cam_topk","rv_cam_aff","rv_cam_aff_s5","rv_cam_aff_a05","rv_cam_aff_s1_a03","rv_cam_aff_s1_a10"):
    meth=load(f"/hy-tmp/validation/{tag}/*.jsonl"); keys=sorted(set(base)&set(meth)); n=len(keys)
    if n==0:
        print(f"{tag:18s} no pairs"); continue
    b=np.array([base[k] for k in keys],float); m=np.array([meth[k] for k in keys],float)
    ob=int(((b==1)&(m==0)).sum()); oa=int(((b==0)&(m==1)).sum())
    p=binomtest(ob,ob+oa,0.5).pvalue if ob+oa else 1.0
    rng=np.random.default_rng(0); ds=[(m[i].mean()-b[i].mean())*100 for i in (rng.integers(0,n,(2000,n)))]
    lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"{tag:18s} n={n:4d} base={b.mean()*100:5.1f}% method={m.mean()*100:5.1f}% delta={(m.mean()-b.mean())*100:+5.1f} [{lo:+5.1f},{hi:+5.1f}] disc={ob}/{oa} p={p:.4f}")
