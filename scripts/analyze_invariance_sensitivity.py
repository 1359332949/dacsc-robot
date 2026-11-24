"""Offline only. Never used to set layers, thresholds, magnitudes or eligibility."""
import argparse,itertools,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np

CH={'xyz':slice(0,3),'rot':slice(3,6)}
def dist(a,b,sl):
    d=(a-b)[...,sl];return np.linalg.norm(d.reshape(d.shape[0],4,-1),axis=2).mean(1)

def geom(a,b,sl):
    d=(a-b)[...,sl].reshape(len(a),-1);n=np.linalg.norm(d,axis=1);dot=d@d[0]
    return np.divide(dot,n*n[0],out=np.zeros_like(n),where=n*n[0]>1e-12),n/max(n[0],1e-12),n[0]/2

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root
    rows=[r for f in sorted((root/'input').glob('index_*.json')) for r in json.loads(f.read_text())];original=[r for r in rows if not r['geometry']];clean={r['task']:r for r in original if r['factor']=='clean'};assert len(original)==100 and len(clean)==10 and len(rows)==len({r['id'] for r in rows});out=root/'outputs';files=list(out.glob('*.npz'));assert {p.stem for p in files}=={r['id'] for r in rows}
    ac={};energy={};hashes={};parity=[];keys=None
    for r in rows:
        i=r['id']
        with np.load(out/(i+'.npz'),allow_pickle=False) as z:
            kk=z['keys'].tolist();assert len(kk)==len(set(kk))==25 and (keys is None or kk==keys);keys=kk;assert np.array_equal(z['identity'],[3,6,9,12]);parity.append(float(z['parity']));ac[i]=np.asarray(z['actions'],dtype=float).clip(-1,1);energy[i]=z['energy'].astype(float).mean(1);hashes[i]=z['noise_hashes'].tolist();assert ac[i].shape==(25,4,12,7) and np.isfinite(ac[i]).all() and np.isfinite(energy[i]).all()
    source=json.loads((root/'sources.json').read_text());assert all(set(source[t])=={c['id'] for other,c in clean.items() if t!=other} for t in clean)
    for r in rows:assert hashes[r['id']]==hashes[clean[r['task']]['id']]
    a.out.mkdir(exist_ok=True,parents=True); raw={};
    for r in original:
        i=r['id'];ref=clean[r['task']];assert (r['sim'],r['state'],r['lang'])==(ref['sim'],ref['state'],ref['lang']);raw[i]={k:dist(ac[i],ac[ref['id']][:1],sl) for k,sl in CH.items()}
    groups={'A1':lambda r:r['factor']=='sensor_noise','A1_blur':lambda r:r['factor']=='sensor_noise' and r['sub']!='fog','A2':lambda r:r['factor']=='background','table':lambda r:r['sub']=='table_texture' and r['factor']=='background','scene':lambda r:r['sub']=='scene_texture' and r['factor']=='background','A3':lambda r:r['factor']=='lighting'}
    stats={k:{} for k in keys};tasks=sorted(clean);rng=np.random.default_rng(20261124);wt=rng.multinomial(10,[.1]*10,size=30000)
    for k in keys:
        j=keys.index(k);stats[k]['clean']={ch:{'mean':float(np.mean(v)),'max':float(np.max(v))} for ch in CH for v in [[raw[r['id']][ch][j] for r in clean.values()]]}
    for f,fn in groups.items():
        rr=[r for r in original if fn(r)];W=wt[:,[tasks.index(r['task']) for r in rr]]
        for ch in CH:
            data=np.stack([raw[r['id']][ch] for r in rr]); denom=W@data[:,0];good=denom>1e-8;boot=100*(1-(W[good]@data)/denom[good,None]);pct=100*(1-data.sum(0)/data[:,0].sum());valid=data[:,0]>1e-4;per=100*(1-data[valid]/data[valid,:1]);
            for j,k in enumerate(keys):
                d=stats[k].setdefault(f,{});d[ch]=dict(n=len(rr),eligible_pct=int(valid.sum()),pooled_gain_pct=float(pct[j]),median_case_gain_pct=float(np.median(per[:,j])),base_mean=float(data[:,0].mean()),after_mean=float(data[:,j].mean()),task_ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),simultaneous_lower=float(np.quantile(boot[:,j],.025/(24*3*2))))
    pairinfo=[];geometry={ch:[] for ch in CH};track=[]
    for t in tasks:
        q=[r for r in rows if r['geometry'] and r['task']==t];
        for x,y in itertools.combinations(q,2):
            assert x['state']==y['state'] and x['lang']==y['lang'];off=np.asarray(x['target_xyz'])-y['target_xyz'];d=np.linalg.norm(off)
            if d<.02-1e-9:continue
            i,k=x['id'],y['id'];cx,rx,mag=geom(ac[i],ac[k],CH['xyz']);cr,rr,magr=geom(ac[i],ac[k],CH['rot']);info=dict(x=i,y=k,task=t,offset=off.tolist(),base_xyz=mag,base_rot=magr,eligible=bool(mag>=.15));pairinfo.append(info)
            if mag<.15:continue
            geometry['xyz'].append((cx,rx));
            if magr>1e-4:geometry['rot'].append((cr,rr))
            delta=ac[i][0,:,:,:3].mean((0,1))-ac[k][0,:,:,:3].mean((0,1));track.append(float(delta@off/(max(np.linalg.norm(delta)*d,1e-12))))
    eligible=[q for q in pairinfo if q['eligible']];assert len(eligible)>0,'no interpretable geometry contrast'
    for ch,pairs in geometry.items():
        arr=np.stack([np.stack(x) for x in pairs]);
        for j,k in enumerate(keys):
            cc,rr=arr[:,:,j].T;stats[k].setdefault('geometry',{})[ch]=dict(n=len(cc),cos_med=float(np.median(cc)),cos_p10=float(np.quantile(cc,.1)),ratio_med=float(np.median(rr)),ratio_p10_p90=np.quantile(rr,[.1,.9]).tolist(),joint_pass=int(((cc>=.95)&(rr>=.88)&(rr<=1.12)).sum()))
    nominal=[];robust=[]
    for k,v in stats.items():
        if k=='frozen':continue
        geo=v['geometry']['xyz'];B=geo['n']>=20 and geo['cos_med']>=.95 and .88<=geo['ratio_med']<=1.12;C=v['clean']['xyz']['mean']<=.02 and v['clean']['xyz']['max']<=.05
        def good(f,strict):return all(v[f][ch]['pooled_gain_pct']>=10 and (v[f][ch]['simultaneous_lower'] if strict else v[f][ch]['task_ci95'][0])>0 for ch in CH)
        v['check']=dict(geometry=bool(B),clean=bool(C),families=[f for f in ('A1','A2','A3') if good(f,False)],simultaneous_families=[f for f in ('A1','A2','A3') if good(f,True)])
        if B and C and len(v['check']['families'])>=2:nominal.append(k)
        if B and C and len(v['check']['simultaneous_families'])>=2:robust.append(k)
    align={}
    for f,fn in groups.items():
        if f not in ['A1','A2','A3']:continue
        rr=[r for r in original if fn(r)];res={}
        for l,m in itertools.product((3,6,9,12),('uniform','dead')):
            ix=[0]+[j for j,k in enumerate(keys) if k.startswith(f'{l}_{m}_')];agree=[];loss=[];choices=Counter()
            for r in rr:
                i=r['id'];e=energy[i][ix];ds=raw[i]['xyz'][ix];c=int(e.argmin());choices[keys[ix[c]]]+=1;agree.append(bool(ds[c]<=ds.min()+1e-6));loss.append(float(ds[c]-ds.min()))
            res[f'{l}_{m}']={'min_energy_choice_counts':dict(choices),'xyz_near_min_agreement':sum(agree),'n':len(rr),'mean_excess_xyz_l2':float(np.mean(loss))}
        align[f]=res
    view_changes={}
    for f,fn in groups.items():
        rr=[r for r in original if fn(r)];both=sum(all(u!=v for u,v in zip(r['input_hash'],clean[r['task']]['input_hash'])) for r in rr);view_changes[f]={'n':len(rr),'both_differ_from_clean':both}
    geometry_checks={'n_reference_tasks':len(tasks),'moved_frames':sum(r['geometry'] and any(r['offset_m']) for r in rows),'excluded_offset_count':sum(len(r.get('excluded_offsets',[])) for r in rows),'true_displacement_verified_m':sorted({round(float(np.linalg.norm(r['offset_m'])),6) for r in rows if r['geometry']}),'eligible_pairs_per_task':dict(Counter(q['task'] for q in eligible)),'same_direction_world_offset_n':sum(v>0 for v in track)}
    payload=dict(view_changes=view_changes,geometry_checks=geometry_checks,status='offline_screen_only',n_original=len(original),n_geometry_frames=len(rows)-100,n_geometry_tasks=len({q['task'] for q in eligible}),pair_counts={'spatially_separated':len(pairinfo),'eligible':len(eligible)},frozen_delta_world_offset_cos_median=float(np.median(track)),max_official_action_error=max(parity),identity_hooks_verified=len(rows)*4,candidates=stats,nominal_pass=nominal,simultaneous_pass=robust,flow_diagnostic=align,geometry_pairs=pairinfo,per_case_energy={i:v.tolist() for i,v in energy.items()},candidate_order=keys)
    (a.out/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');print('eligible pairs',len(eligible),'nominal',nominal,'robust',robust)
if __name__=='__main__':main()
