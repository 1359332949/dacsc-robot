"""29-case offline oracle ceilings and strict other-task-Clean fixed pullbacks."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import authorize,file_hash,append_event
from dacsc.global_shift_diag import pullback


def main():
    ap=argparse.ArgumentParser()
    for k in ['manifest','features','stats','checkpoint','corp-src','libero-src','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--num-shards',type=int,default=4);ap.add_argument('--shard-id',type=int,default=0);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());mh=file_hash(a.manifest);cases={c['case_id']:c for c in m['cases']}
    expected=json.loads((a.manifest.parent/'manifest_hashes.json').read_text())[a.manifest.name]
    if m.get('kind')!='dev':raise PermissionError('Dev only')
    for c in m['cases']:authorize(m,c,'offline global pullback probe',expected,a.manifest)
    records=[json.loads(p.read_text()) for p in a.features.glob('*.json')]
    if len(records)!=100 or any(r['status']!='completed' for r in records):raise ValueError('100 feature cases needed')
    clean={r['pair_group_id']:r for r in records if r['factor']=='clean'}
    selected=[r for r in records if r['factor']=='clean' or (r['factor'] in ('background','lighting') and not r['baseline_success'])]
    if len(selected)!=29:raise ValueError('13 Background failures + 6 Lighting failures +10 Clean')
    selected=sorted(selected,key=lambda r:r['case_id'])[a.shard_id::a.num_shards]
    stat_meta=json.loads((a.stats/'loto_stats.json').read_text());stats=np.load(a.stats/'loto_stats.npz')
    for r in records:
        source=[cases[cid] for cid in stat_meta['source_case_ids'][r['base_task']]]
        if len(source)!=9 or any(c['factor']!='clean' or c['base_task']==r['base_task'] for c in source):raise ValueError('source leakage')
    sys.path[:0]=[str(a.libero_src),str(a.corp_src),str(a.corp_src/'scripts'),str(ROOT/'scripts')]
    os.environ.setdefault('MUJOCO_GL','osmesa');os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
    import torch
    from corp import load_policy
    from corp.action_generator import PrefixCache
    from libero.libero.benchmark import grab_language_from_filename
    from global_shift_diag_v1 import restore_cache,expand_cache,sample_cache,encode
    from dacsc.flow_repair_kv import energy
    ch=file_hash(a.checkpoint)
    if ch!='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6':raise ValueError('checkpoint drift')
    policy=load_policy(a.checkpoint,device='cuda:0');policy.eval();policy.requires_grad_(False);a.out.mkdir(parents=True,exist_ok=True)
    def eval_cache(cache,z):
        with torch.no_grad():actions,vel,_=sample_cache(policy,expand_cache(cache),z);_,e=energy(vel)
        return {'actions':actions.cpu().tolist(),'relative_curvature':e['relative'].cpu().tolist()}
    for r in selected:
        began=time.monotonic();cid=r['case_id'];p=a.out/(cid+'.json')
        if p.exists():raise FileExistsError(p)
        try:
            cr=clean[r['pair_group_id']];f=a.features/(cid+'.npz');cf=a.features/(cr['case_id']+'.npz')
            if file_hash(f)!=r['feature_npz_sha256'] or file_hash(cf)!=cr['feature_npz_sha256']:raise ValueError('feature hash drift')
            arr=np.load(f);ref=np.load(cf)
            if r['z0_hashes']!=cr['z0_hashes']:raise ValueError('paired noise mismatch')
            state_diff=float(np.max(np.abs(arr['state']-ref['state'])))
            if state_diff>1e-6:raise ValueError('nonvisual state differs; cannot claim visual-only ceiling')
            cache=restore_cache(arr);reference_cache=restore_cache(ref);z=torch.tensor(arr['z0'],device='cuda:0');base=eval_cache(cache,z)
            if np.max(np.abs(np.array(base['actions'])-arr['actions']))>1e-5:raise ValueError('cached native action mismatch')
            append_event(a.out/'access_ledger.jsonl',{'action':'global_probe_start','case_id':cid,'source_case_ids':stat_meta['source_case_ids'][r['base_task']]})
            variants=[]
            for name,parts,views in [('oracle_agent_value',('value',),('agentview',)),('oracle_wrist_value',('value',),('eye_in_hand',)),
                                     ('oracle_agent_kv',('key','value'),('agentview',)),('oracle_wrist_kv',('key','value'),('eye_in_hand',)),
                                     ('oracle_both_kv',('key','value'),('agentview','eye_in_hand'))]:
                k=cache[0].key.clone();v=cache[0].value.clone()
                for view in views:
                    start=0 if view=='agentview' else 48
                    if 'key' in parts:k[:,:,start:start+48]=reference_cache[0].key[:,:,start:start+48]
                    if 'value' in parts:v[:,:,start:start+48]=reference_cache[0].value[:,:,start:start+48]
                ev=eval_cache([PrefixCache(k,v,cache[0].adaln)],z);ev.update(name=name,privileged_oracle=True);variants.append(ev)
            ti=stat_meta['tasks'].index(r['base_task']);lang=grab_language_from_filename(cases[cid]['suite'],r['base_task']+'.bddl')
            for alpha in (.25,.5):
                images={}
                for view in ('agentview','eye_in_hand'):
                    im=arr['rgb_'+view].astype(np.float64);mu=im.mean((0,1));sd=im.std((0,1));sm=stats[f'{ti}__rgb__{view}__mean'];ss=stats[f'{ti}__rgb__{view}__std']
                    aligned=(im-mu)*(ss/np.maximum(sd,1e-6))+sm
                    mixed=np.clip(np.rint((1-alpha)*im+alpha*aligned),0,255).astype(np.uint8)
                    images[view]=torch.tensor(mixed[None])
                po={'images':images,'state':torch.tensor(arr['state']),'instruction':[lang]}
                with torch.no_grad():_,_,newcache=encode(policy,po)
                ev=eval_cache(newcache,z);ev.update(name=f'photo_alpha_{alpha}',privileged_oracle=False,alpha=alpha);variants.append(ev)
            for tau in (2.,3.):
                for beta in (.3,.6):
                    arrays={};altered=0;total=0
                    for kind in ('key','value'):
                        old=arr['prefix_'+kind]
                        flat=old.transpose(0,2,1,3).reshape(1,old.shape[2],-1).copy()
                        for view,start in [('agentview',0),('eye_in_hand',48)]:
                            srcmean=stats[f'{ti}__{kind}__{view}__mean'];srcstd=stats[f'{ti}__{kind}__{view}__std']
                            x=flat[0,start:start+48].copy();y,mask=pullback(x,srcmean,srcstd,tau,beta);yf=y.astype(np.float32)
                            if not np.array_equal(yf[~mask],x[~mask]):raise ValueError('normal component changed')
                            flat[0,start:start+48]=yf;altered+=int(mask.sum());total+=mask.size
                        arrays[kind]=torch.tensor(flat.reshape(1,old.shape[2],old.shape[1],old.shape[3]).transpose(0,2,1,3).copy(),device='cuda:0')
                    ev=eval_cache([PrefixCache(arrays['key'],arrays['value'],cache[0].adaln)],z)
                    ev.update(name=f'kv_tau_{tau}_beta_{beta}',privileged_oracle=False,tau=tau,beta=beta,altered_components=altered,total_components=total);variants.append(ev)
            row={'case_id':cid,'factor':r['factor'],'sub_dimension':r['sub_dimension'],'base_task':r['base_task'],'baseline_success':r['baseline_success'],
                'status':'completed','step0':base,'paired_clean_actions':ref['actions'].tolist(),'variants':variants,'state_pair_max_abs':state_diff,
                'source_case_ids':stat_meta['source_case_ids'][r['base_task']],'source_n':9,'z0_hashes':r['z0_hashes'],
                'feature_npz_sha256':file_hash(f),'paired_npz_sha256':file_hash(cf),'source_stats_sha256':file_hash(a.stats/'loto_stats.npz'),
                'manifest_sha256':mh,'script_sha256':file_hash(Path(__file__)),
                'policy_weights_updated':False,'adapted_actions_executed':False,'blind_accessed':False,'wall_seconds':time.monotonic()-began}
            p.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');append_event(a.out/'access_ledger.jsonl',{'action':'global_probe_end','case_id':cid,'result_sha256':file_hash(p)})
            print(json.dumps({'case_id':cid,'factor':r['factor'],'completed':True,'seconds':row['wall_seconds']}),flush=True)
        except Exception as exc:
            import traceback
            failed={'case_id':cid,'status':'exception','error':repr(exc),'traceback':traceback.format_exc()}
            p.write_text(json.dumps(failed,indent=2)+'\n');append_event(a.out/'access_ledger.jsonl',{'action':'global_probe_exception','case_id':cid,'result_sha256':file_hash(p)})
            raise

if __name__=='__main__':main()
