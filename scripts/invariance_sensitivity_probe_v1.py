"""Two-stage EGL t=0 diagnostic on pinned V2 Dev: collect, then single-block grid."""
import argparse,ctypes,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CORP=ROOT/'repos/CoRP';LIB=Path('/hy-tmp/LIBERO-plus');sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from dacsc.static_visual_v2 import file_hash,array_hash,authorize,seed_for,sensor_transform
VIEWS=('agentview','eye_in_hand'); KEYS=('agentview_image','robot0_eye_in_hand_image');LAYERS=(3,6,9,12)

def collect(a,cases,policy):
    import torch
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper
    receipts=[];out=a.out/'input';out.mkdir(exist_ok=True,parents=True)
    def write(cid,c,obs,raw,env,lang,extra):
        po=policy_observation(obs,lang,policy);arr={v:po['images'][v][0].cpu().numpy() for v in VIEWS};arr['state']=po['state'].cpu().numpy();np.savez_compressed(out/(cid+'.npz'),**arr)
        receipts.append(dict(id=cid,case=c['case_id'],task=c['base_task'],factor=c['factor'],sub=c['sub_dimension'],noise=c['runtime_params']['noise'],lang=lang,sim=array_hash(env.sim.get_state().flatten()),state=array_hash(arr['state']),file_sha=file_hash(out/(cid+'.npz')),raw_hash=[array_hash(np.asarray(raw[k])) for k in KEYS],input_hash=[array_hash(np.asarray(obs[k])) for k in KEYS],**extra))
    for c in sorted(cases,key=lambda c:c['case_id'])[a.shard::a.workers]:
        cid=c['case_id'];ref=json.loads((ROOT/'docs/data/static_visual_v2_suite/final/results'/(cid+'.json')).read_text());seed=ref['seeds']['environment_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);ctypes.CDLL(None).srand(ctypes.c_uint(seed));lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl')
        assert file_hash(LIB/c['bddl'])==c['resolved_bddl_hash'] and file_hash(LIB/c['init_state_file'])==c['init_state_file_sha256']
        env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
        try:
            env.noise=0;env.seed(seed);env.reset();initial=torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[c['init_state_index']];assert array_hash(np.asarray(initial))==c['raw_init_hash'];raw=env.set_init_state(initial)
            for _ in range(SETTLING_STEPS):raw=env.step(DUMMY_ACTION)[0]
            obs=raw;code=c['runtime_params']['noise'];ns=ref['seeds']['sensor_noise_seed'];
            if code:
                obs=dict(raw)
                for i,k in enumerate(KEYS):obs[k]=sensor_transform(dict(raw,agentview_image=raw[k]),code,ns if i==0 else seed_for(ns,'wrist'),0,wrapper)['agentview_image']
            write(cid,c,obs,raw,env,lang,{'geometry':False});eligible=c['factor']=='clean' and any(t in c['base_task'] for t in ('from_table_center','next_to_','between_the_plate'))
            if eligible:
                sim=env.sim; model=sim.model; obj=env.env.objects_dict['akita_black_bowl_1'];joint=obj.joints[0];body=model.body_name2id(obj.root_body);start,end=model.get_joint_qpos_addr(joint);assert end-start==7
                orig=sim.get_state().flatten().copy();pre_pos=sim.data.body_xpos[body].copy();zero_obs=env.set_init_state(orig.copy());pos=sim.data.body_xpos[body].copy();qbase=sim.data.qpos.copy();goal='target_xyz'
                write(cid+'_move_zero',c,zero_obs,zero_obs,env,lang,{'geometry':True,goal:pos.tolist(),'offset_m':[0.,0.,0.],'forward_refresh_delta_m':float(np.linalg.norm(pos-pre_pos))})
                def contacts():
                    d={}
                    for contact in sim.data.contact[:sim.data.ncon]:
                        i,j=int(contact.geom1),int(contact.geom2);bi,bj=model.geom_bodyid[[i,j]]
                        if body not in (bi,bj):continue
                        key=model.geom_id2name(j if bi==body else i);d[str(key)]=min(d.get(str(key),1.),float(contact.dist))
                    return d
                orig_contact=contacts(); base_state_hash=receipts[-1]['state'];base_receipt=receipts[-1];base_receipt['excluded_offsets']=[]
                for axis in (0,1):
                    for cm in (-35,-20,20,35):
                        state=orig.copy();state[1+start+axis]+=cm/1000;shift=env.set_init_state(state);q=sim.data.qpos.copy();q[start+axis]-=cm/1000
                        assert np.allclose(q,qbase,atol=1e-12,rtol=0),('non-target qpos',float(abs(q-qbase).max()));assert np.array_equal(sim.get_state().flatten()[1+model.nq:],orig[1+model.nq:]),'velocity mismatch';xyz=sim.data.body_xpos[body].copy();off=np.eye(3)[axis]*cm/1000;assert np.allclose(xyz-pos,off,rtol=0,atol=1e-9),('world XYZ mismatch',xyz-pos,off)
                        ct=contacts()
                        if set(ct)!=set(orig_contact) or any(ct[k]<orig_contact[k]-.001 for k in ct):
                            base_receipt['excluded_offsets'].append([axis,cm,ct]);continue
                        tag=f'{cid}_move_{axis}_{cm}';write(tag,c,shift,shift,env,lang,{'geometry':True,goal:xyz.tolist(),'offset_m':off.tolist(),'contacts':ct});assert receipts[-1]['state']==base_state_hash
            print('collected',cid,flush=True)
        finally:env.close()
    (out/f'index_{a.shard}.json').write_text(json.dumps(receipts,allow_nan=False)+'\n')

def main():
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['collect','probe']);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--shard',type=int,default=0);a=ap.parse_args()
    mp=ROOT/'docs/data/static_visual_v2_suite/final/dev_manifest.json';mh=file_hash(mp);assert mh=='d3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0' and 0<=a.shard<a.workers;assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl';m=json.loads(mp.read_text());cases=m['cases'];assert m['kind']=='dev' and len(cases)==100
    for c in cases:authorize(m,c,'invariance/sensitivity t0',mh,mp)
    from corp import load_policy
    cp=CORP/'checkpoints/corp_libero.pt';assert file_hash(cp)=='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6';policy=load_policy(cp,device='cuda:0').eval().requires_grad_(False)
    with torch.no_grad():
        if a.mode=='collect':collect(a,cases,policy)
        else:
            from dacsc.single_block_probe import run
            run(a,cases,policy)
if __name__=='__main__':main()
