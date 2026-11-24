"""P2 geometry audit: bowl body/geoms, rim-like vertices, finger pads."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from target_only_action_validation import configure  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa


def target_from_bddl(path):
    t=Path(path).read_text(); m=re.search(r'\(:goal\s*\(And\s*\(On\s+([^\s\)]+)',t,re.S)
    if not m: raise RuntimeError('cannot parse target')
    return m.group(1)


def quat_mat(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-id',type=int,default=37); ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--render-gpu',type=int,default=0); ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    env=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
    try:
        env.seed(7777); env.reset(); obs=env.set_init_state(states[args.state_id])
        body_names=[str(x) for x in env.sim.model.body_names]; geom_names=[str(x) for x in env.sim.model.geom_names]
        target=target_from_bddl(variant['base_bddl']); tids=[i for i,n in enumerate(body_names) if n==f'{target}_main']
        if len(tids)!=1: raise RuntimeError('target body not unique')
        tid=tids[0]; gids=[i for i in range(len(geom_names)) if int(env.sim.model.geom_bodyid[i])==tid]
        cols=[i for i in gids if int(env.sim.model.geom_contype[i])!=0 or int(env.sim.model.geom_conaffinity[i])!=0]
        visual=[i for i in gids if i not in cols]
        # aggregate collision mesh vertices in target body frame
        verts=[]; geom_info=[]
        for gi in cols:
            did=int(env.sim.model.geom_dataid[gi]); info={'name':geom_names[gi],'type':int(env.sim.model.geom_type[gi]),'dataid':did,'contype':int(env.sim.model.geom_contype[gi]),'conaffinity':int(env.sim.model.geom_conaffinity[gi])}
            if did>=0:
                adr=int(env.sim.model.mesh_vertadr[did]); num=int(env.sim.model.mesh_vertnum[did]); mv=np.asarray(env.sim.model.mesh_vert[adr:adr+num],float)
                R=quat_mat(env.sim.model.geom_quat[gi]); p=env.sim.model.geom_pos[gi]; v=mv@R.T+p; verts.append(v); info['nverts']=num
            geom_info.append(info)
        verts=np.concatenate(verts) if verts else np.zeros((0,3))
        if len(verts):
            z=verts[:,2]; rad=np.linalg.norm(verts[:,:2],axis=1)
            top=verts[z>=np.quantile(z,.9)]; toprad=np.linalg.norm(top[:,:2],axis=1)
            rim={'z_min':float(z.min()),'z_max':float(z.max()),'z_top10_min':float(np.quantile(z,.9)),
                 'radius_all_mean':float(rad.mean()),'radius_top10_mean':float(toprad.mean()),'radius_top10_std':float(toprad.std()),
                 'radius_top10_p10':float(np.quantile(toprad,.1)),'radius_top10_p90':float(np.quantile(toprad,.9))}
        else: rim=None
        finger_bodies=[i for i,n in enumerate(body_names) if 'finger' in n and 'gripper0' in n]
        finger_pos=env.sim.data.body_xpos[finger_bodies] if finger_bodies else np.zeros((0,3))
        # closing axis: vector between left/right finger bodies
        closing=None
        if len(finger_bodies)>=2:
            v=finger_pos[1]-finger_pos[0]; closing=(v/(np.linalg.norm(v)+1e-8)).tolist()
        out={'schema':'bowl_geometry_audit_v1','target_object':target,'target_body':f'{target}_main','target_body_id':tid,
             'body_origin':np.asarray(env.sim.data.body_xpos[tid],float).tolist(),
             'n_geoms_total':len(gids),'n_collision_geoms':len(cols),'n_visual_geoms':len(visual),
             'collision_geoms':geom_info,'rim_vertex_stats':rim,'n_rim_vertices':int(len(verts)),
             'finger_bodies':[body_names[i] for i in finger_bodies], 'finger_positions':finger_pos.tolist(),'closing_axis':closing,
             'note':'rim vertex stats are mesh geometry diagnostics, not a validated grasp-point distance; collision mesh may include non-rim vertices.'}
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)[:4000])
    finally: env.close()

if __name__=='__main__': main()
