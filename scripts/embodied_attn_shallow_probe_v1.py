"""EAGA 5-minute feasibility probe: frozen CoRP attention survival on cached t=0 Dev."""
from __future__ import annotations
import json,math,sys,os
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from corp import load_policy
from corp.preprocessing import prepare_observation
from dacsc.static_visual_v2 import file_hash
from dacsc.single_block_probe import moments
from dacsc.spatial_patch_bias import _manual_mha
INPUT=ROOT/'runs/invariance_sensitivity_v1b/input'
OUT=ROOT/'runs/eaga_probe_v1'

def load_row(r):
    z=np.load(INPUT/(r['id']+'.npz'),allow_pickle=False)
    raw={'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':[r['lang']]}
    return prepare_observation(raw,config=policy.config,normalizer=policy.normalizer,device='cuda:0')

def attn_for(policy,po):
    ex=policy.representation_extractor; rs=ex.cross_attention_resampler; patches=ex.visual_encoder(torch.cat([po['images'][v] for v in ex.view_names],dim=0)); B=po['state'].shape[0]
    pt={v:patches[i*B:(i+1)*B] for i,v in enumerate(ex.view_names)}; lang,pool=ex.language_encoder(po['instruction'])
    gamma,beta=rs.film(pool).unsqueeze(1).chunk(2,dim=-1); ws=[]; checks=[]
    for i,v in enumerate(ex.view_names):
        p=rs.input_projection(pt[v])+rs.view_embedding[i].unsqueeze(0); q=rs.queries[i].unsqueeze(0).expand(B,-1,-1); q=q*(1.0+gamma)+beta
        native,weights=rs.cross_attention(q,p,p,need_weights=True,average_attn_weights=False)
        # exact native attention: average heads then 48 queries -> 256 patch weights, sum=1
        w=weights.mean(dim=1).mean(dim=1)[0]
        manual=_manual_mha(rs.cross_attention,q,p,p)
        checks.append(float((manual-native).abs().max()))
        ws.append(w)
    assert max(checks)<1e-4, checks
    return torch.stack(ws), max(checks)

def metrics(w,q):
    ne=float((1.0/(w.square().sum(dim=-1)+1e-12)).mean()); cos=float(torch.nn.functional.cosine_similarity(w,q,dim=-1).mean()); k=max(1,int(round(0.25*w.shape[-1]))); ious=[]
    for a,b in zip(w,q):
        top=torch.topk(a,k).indices; topq=torch.topk(b,k).indices; inter=len(set(top.tolist())&set(topq.tolist())); ious.append(inter/(2*k-inter))
    return ne,cos,float(np.mean(ious))

def main():
    global policy
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False)
    rows=[]
    for f in sorted(INPUT.glob('index_*.json')): rows+=json.load(open(f))
    rows=[r for r in rows if not r.get('geometry')]; assert len(rows)==100
    rec={}
    for r in rows:
        po=load_row(r); w,par=attn_for(policy,po); rec[r['id']]={'task':r['task'],'factor':r['factor'],'sub':r['sub'],'w':w.cpu().numpy(),'parity':par}
    clean={r['task']:rec[r['id']]['w'] for r in rows if r['factor']=='clean'}; assert len(clean)==10
    out=[]
    for r in rows:
        x=rec[r['id']]; g=x['factor'];
        if g=='clean': ne=float(np.mean(1/(np.square(x['w']).sum(-1)+1e-12))); out.append({'id':r['id'],'task':r['task'],'factor':g,'sub':x['sub'],'n_eff':ne,'cos':1.0,'iou':1.0}); continue
        q=clean[r['task']]; ne,cos,iou=metrics(torch.from_numpy(x['w']),torch.from_numpy(q)); out.append({'id':r['id'],'task':r['task'],'factor':g,'sub':x['sub'],'n_eff':ne,'cos':cos,'iou':iou})
    # group summaries
    groups={}
    for g in ['clean','sensor_noise','background','lighting']:
        sub=[x for x in out if x['factor']==g]; groups[g]={'n':len(sub),'n_eff_mean':float(np.mean([x['n_eff'] for x in sub])),'cos_mean':float(np.mean([x['cos'] for x in sub])),'iou_mean':float(np.mean([x['iou'] for x in sub])),'n_eff_max':float(np.max([x['n_eff'] for x in sub])),'iou_min':float(np.min([x['iou'] for x in sub]))}
        if g=='background':
            for s in ['table_texture','scene_texture']:
                ss=[x for x in sub if s in x['sub']]; groups['background_'+s]={'n':len(ss),'n_eff_mean':float(np.mean([x['n_eff'] for x in ss])),'cos_mean':float(np.mean([x['cos'] for x in ss])),'iou_mean':float(np.mean([x['iou'] for x in ss])),'iou_min':float(np.min([x['iou'] for x in ss]))}
    gate={'pass':True,'reasons':[]}
    if groups['sensor_noise']['n_eff_max']>180: gate['pass']=False; gate['reasons'].append('sensor N_eff>180')
    if groups.get('background_table_texture',{}).get('iou_mean',1)<.25: gate['pass']=False; gate['reasons'].append('table IoU<0.25')
    OUT.mkdir(exist_ok=True); (OUT/'probe.json').write_text(json.dumps({'schema':'eaga_attention_probe_v1','groups':groups,'cases':out,'gate':gate,'max_parity':max(v['parity'] for v in rec.values())},indent=2)+'\n')
    print(json.dumps({'groups':groups,'gate':gate},indent=2))
if __name__=='__main__': main()
