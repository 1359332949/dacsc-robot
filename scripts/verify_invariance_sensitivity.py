"""Independently reopen caches and check source references/identity before interpreting screen."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root;s=json.loads(a.summary.read_text());rows=[r for p in sorted((root/'input').glob('index_*.json')) for r in json.loads(p.read_text())];allby={r['id']:r for r in rows};tasks={r['task']:r for r in rows if r['factor']=='clean' and not r['geometry']};directions=[];primary={k:[] for k in s['simultaneous_pass']};cache={};noise={};outputs={};sources=json.loads((root/'sources.json').read_text())
    for r in rows:
        i=r['id'];f=root/'input'/(i+'.npz');assert sha(f)==r['file_sha'];o=root/'outputs'/(i+'.npz');outputs[i]=sha(o)
        with np.load(o,allow_pickle=False) as d:cache[i]=np.clip(d['actions'].astype(float),-1,1);noise[i]=d['noise_hashes'].tolist();assert d['keys'].tolist()==s['candidate_order'] and np.isfinite(d['energy']).all() and np.array_equal(d['identity'],[3,6,9,12])
        assert len(set(noise[i]))==4
    for r in rows:
        i=r['id'];t=r['task'];assert set(sources[t])=={c['id'] for k,c in tasks.items() if k!=t} and noise[i]==noise[tasks[t]['id']]
        if r['geometry']:
            ref=allby[r['case']+'_move_zero'];assert (ref['lang'],ref['state'])==(r['lang'],r['state']);np.testing.assert_allclose(np.subtract(r['target_xyz'],ref['target_xyz']),r['offset_m'],atol=1e-9,rtol=0)
    # Additional directional sanity subset: positive world-coordinate dot (descriptive only).
    for pair in s['geometry_pairs']:
        if not pair['eligible']:continue
        delta=(cache[pair['x']]-cache[pair['y']])[...,:3];off=np.array(pair['offset']);u=delta[0].mean((0,1));dcos=float((u@off)/max(1e-12,np.linalg.norm(u)*np.linalg.norm(off)));directions.append(dcos)
        if dcos<=0:continue
        v=delta[0].flatten();n=np.linalg.norm(v)
        for k in primary:
            w=delta[s['candidate_order'].index(k)].flatten();ratio=float(np.linalg.norm(w)/n);cos=float(w@v/(np.linalg.norm(w)*n));primary[k].append((cos,ratio))
    assert len(directions)==s['pair_counts']['eligible'];app={}
    for k in primary:
        app[k]={}
        for f in ['sensor_noise','background','lighting']:
            rr=[r for r in rows if r['factor']==f and not r['geometry']];i=s['candidate_order'].index(k);acc={}
            for name,sl in [('xyz',slice(0,3)),('rot',slice(3,6))]:
                ds=[(np.linalg.norm((cache[r['id']][j]-cache[tasks[r['task']]['id']][0])[:, :, sl].reshape(4,-1),axis=1).mean()) for r in rr for j in (0,i)];base,after=np.array(ds).reshape(-1,2).T;val=100*(1-after.sum()/base.sum());np.testing.assert_allclose(val,s['candidates'][k][{'sensor_noise':'A1','background':'A2','lighting':'A3'}[f]][name]['pooled_gain_pct']);acc[name]=dict(closer=int((after<base-1e-6).sum()),farther=int((after>base+1e-6).sum()),n=len(rr))
            app[k][f]=acc
    v={'pass':True,'raw_inputs_checked':len(rows),'output_hashes':outputs,'paired_noise_separation':'four unique and identical within task','positive_world_dot_n':sum(x>0 for x in directions),'positive_dot_joint_pass':{k:sum(c>=.95 and .88<=r<=1.12 for c,r in v) for k,v in primary.items()},'appearance_case_counts':app,'archive_sha256':sha(root/'evidence.tar.gz'),'source_fits_sha256':sha(root/'references.npz'),'summary_sha256':sha(a.summary)};a.out.write_text(json.dumps(v,indent=2)+'\n');print('verified',len(rows),'frames; positive-direction subset',v['positive_dot_joint_pass'])
if __name__=='__main__':main()
