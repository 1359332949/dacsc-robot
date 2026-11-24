"""Exact V2 t=0 multi-layer cache extraction. No rollout or policy updates."""
import argparse,json,sys,os,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import authorize,file_hash,array_hash,seed_for,append_event


def expand_cache(cache,n=4):
    from corp.action_generator import PrefixCache
    return [PrefixCache(c.key.expand(n,-1,-1,-1),c.value.expand(n,-1,-1,-1),tuple(v.expand(n,-1) for v in c.adaln)) for c in cache]


def sample_cache(policy,cache,z,suffix=False):
    import torch
    x=z.clone();vel=[];hidden=[]
    handle=policy.action_generator.output_norm.register_forward_hook(lambda mod,args,out:hidden.append(out.detach())) if suffix else None
    try:
        for k in range(10):
            t=torch.full((len(z),),k/10,device=z.device,dtype=z.dtype)
            v=policy.action_generator(x,t,cache);vel.append(v);x=x+v/10
    finally:
        if handle is not None:handle.remove()
    return policy.normalizer.unnormalize_action(x),torch.stack(vel),torch.stack(hidden).mean((0,1)) if hidden else None


def encode(policy,po):
    import torch
    from corp.preprocessing import prepare_observation
    from corp.representation.extractor import Representation
    ex=policy.representation_extractor
    with torch.no_grad():
        b=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
        p=ex.visual_encoder(torch.cat([b['images'][n] for n in ex.view_names],dim=0))
        patches={n:p[i:i+1] for i,n in enumerate(ex.view_names)}
        lt,pl=ex.language_encoder(b['instruction']);vis=ex.cross_attention_resampler(patches,pl);st=ex.state_encoder(b['state'])
        rep=Representation(tokens=torch.cat([vis,lt.to(vis.dtype),st],1),pooled_language=pl,action_dtype=vis.dtype)
        cache=policy.action_generator.prepare_prefix(rep)
    if len(cache)!=1 or tuple(ex.view_names)!=('agentview','eye_in_hand') or vis.shape[1]!=96:raise ValueError('checkpoint architecture changed')
    return patches,vis,cache


def cache_arrays(cache):
    c=cache[0];a={'prefix_key':c.key.cpu().numpy(),'prefix_value':c.value.cpu().numpy()}
    a.update({'adaln_'+str(i):v.cpu().numpy() for i,v in enumerate(c.adaln)})
    return a


def restore_cache(arrays,device='cuda:0'):
    import torch
    from corp.action_generator import PrefixCache
    return [PrefixCache(torch.tensor(arrays['prefix_key'],device=device),torch.tensor(arrays['prefix_value'],device=device),tuple(torch.tensor(arrays['adaln_'+str(i)],device=device) for i in range(4)))]


def visual_tokens(tensor,view):
    # prefix [B,heads,length,head_dim] -> 48 x concatenated-head channels
    start=0 if view=='agentview' else 48
    return tensor[0,:,start:start+48].transpose(1,0,2).reshape(48,-1)


def main():
    ap=argparse.ArgumentParser()
    for k in ['manifest','results','flow-results','checkpoint','libero-src','corp-src','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--num-shards',type=int,default=4);ap.add_argument('--shard-id',type=int,default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());mh=file_hash(a.manifest)
    if m.get('kind')!='dev' or m.get('status')!='FROZEN':raise PermissionError('frozen Dev only')
    expected=json.loads((a.manifest.parent/'manifest_hashes.json').read_text())[a.manifest.name]
    for c in m['cases']:authorize(m,c,'global t0 feature diagnostic',expected,a.manifest)
    cases=sorted(m['cases'],key=lambda c:c['case_id'])[a.shard_id::a.num_shards]
    if a.limit:cases=cases[:a.limit]
    sys.path[:0]=[str(a.libero_src),str(a.corp_src),str(a.corp_src/'scripts'),str(ROOT/'scripts')]
    os.environ.setdefault('MUJOCO_GL','osmesa');os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
    import torch
    from corp import load_policy
    from evaluate_libero import policy_observation
    from libero.libero.benchmark import grab_language_from_filename
    from flow_causal_repair_probe import observe
    from dacsc.global_shift_diag import moments
    from unittest.mock import patch
    ch=file_hash(a.checkpoint)
    if ch!='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6':raise ValueError('checkpoint drift')
    policy=load_policy(a.checkpoint,device='cuda:0');policy.eval();policy.requires_grad_(False)
    a.out.mkdir(parents=True,exist_ok=True)
    for c in cases:
        cid=c['case_id'];p=a.out/(cid+'.json');npz=a.out/(cid+'.npz')
        if p.exists() or npz.exists():raise FileExistsError(cid)
        began=time.monotonic();append_event(a.out/'access_ledger.jsonl',{'action':'global_extract_start','case_id':cid,'manifest_sha256':mh})
        ref=json.loads((a.results/(cid+'.json')).read_text());old=json.loads((a.flow_results/(cid+'.json')).read_text())
        if ref['manifest_sha256']!=mh or ref['checkpoint_sha256']!=ch:raise ValueError('baseline receipt drift')
        for field,key in [('resolved_bddl_hash','bddl'),('init_state_file_sha256','init_state_file')]:
            if file_hash(a.libero_src/c[key])!=c[field]:raise ValueError('asset drift')
        try:
            obs=observe(c,ref,a);lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl')
            if lang!=ref['language_instruction']:raise ValueError('language mismatch')
            po=policy_observation(obs,lang,policy)
            zz=[];hashes=[]
            for j in range(4):
                seed=ref['seeds']['policy_flow_seed'] if j==0 else seed_for(ref['seeds']['method_augmentation_seed'],'flow_geometry_v1',j)
                z=torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed));zz.append(z);hashes.append(array_hash(z.cpu().numpy()))
            if hashes!=old['z0_hashes']:raise ValueError('V1 noise mismatch')
            z=torch.cat(zz)
            with torch.no_grad():
                patches,vis,cache=encode(policy,po);actions,vel,suf=sample_cache(policy,expand_cache(cache),z,suffix=True)
                orig,_,_=sample_cache(policy,cache,z[:1])
                with patch('torch.randn',side_effect=lambda *args,**kw:z[:1].clone()):official=policy.predict(po)
                parity=float((orig-official).abs().max());v1=float((actions-torch.tensor(old['controller_actions'],device='cuda:0')).abs().max())
                if parity>1e-5 or v1>1e-4:raise ValueError('official/V1 action mismatch')
            arrays=cache_arrays(cache);arrays.update(suffix=suf.cpu().numpy(),actions=actions.cpu().numpy(),state=po['state'].cpu().numpy(),z0=z.cpu().numpy(),
                rgb_agentview=po['images']['agentview'][0].cpu().numpy(),rgb_eye_in_hand=po['images']['eye_in_hand'][0].cpu().numpy())
            arrays.update({'dino_'+v:patches[v][0].cpu().numpy() for v in ('agentview','eye_in_hand')})
            arrays['resampler_agentview']=vis[0,:48].cpu().numpy();arrays['resampler_eye_in_hand']=vis[0,48:].cpu().numpy()
            sketches={}
            for view in ('agentview','eye_in_hand'):
                sketches['dino__'+view]=moments(arrays['dino_'+view])
                sketches['dino_mean__'+view]=moments(arrays['dino_'+view].mean(0,keepdims=True))
                sketches['resampler__'+view]=moments(arrays['resampler_'+view])
                sketches['key__'+view]=moments(visual_tokens(arrays['prefix_key'],view))
                sketches['value__'+view]=moments(visual_tokens(arrays['prefix_value'],view))
                sketches['rgb__'+view]=moments(arrays['rgb_'+view].reshape(-1,3))
            sketches['suffix__joint']=moments(arrays['suffix'])
            arrays.update({'moment_'+k:v for k,v in sketches.items()});np.savez_compressed(npz,**arrays)
            row={'case_id':cid,'factor':c['factor'],'base_task':c['base_task'],'pair_group_id':c['pair_group_id'],'sub_dimension':c.get('sub_dimension'),
                'baseline_success':ref['success'],'noise_code':int(c['runtime_params']['noise']),'status':'completed','moments':{k:v.tolist() for k,v in sketches.items()},
                'flow_m4_variance':float(np.mean([x['all']['velocity_variance'] for x in old['metrics']])),
                'flow_m4_relative':float(np.mean([x['all']['relative_variance'] for x in old['metrics']])),
                'official_max_abs':parity,'v1_actions_max_abs':v1,'actions':actions.cpu().tolist(),'z0_hashes':hashes,
                'baseline_initial_rgb':ref['initial_observation'],'feature_npz_sha256':file_hash(npz),
                'baseline_result_sha256':file_hash(a.results/(cid+'.json')),'manifest_sha256':mh}
        except Exception as exc:
            import traceback
            row={'case_id':cid,'status':'exception','error':repr(exc),'traceback':traceback.format_exc()}
        row.update(script_sha256=file_hash(Path(__file__)),statistics_module_sha256=file_hash(ROOT/'dacsc/global_shift_diag.py'),wall_seconds=time.monotonic()-began,
                   policy_weights_updated=False,adapted_actions_executed=False,blind_accessed=False)
        p.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');append_event(a.out/'access_ledger.jsonl',{'action':'global_extract_end','case_id':cid,'status':row['status'],'result_sha256':file_hash(p)})
        print(json.dumps({'case_id':cid,'status':row['status'],'seconds':row['wall_seconds'],'error':row.get('error')}),flush=True)
        if row['status']!='completed':raise RuntimeError(row['error'])

if __name__=='__main__':main()
