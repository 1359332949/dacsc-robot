"""P2: exact point-to-OBB grasp geometry on the 40 bowl collision boxes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from target_only_action_validation import configure  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='37,43,45'); ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--settle',type=int,default=10); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False)
    env=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
    out={'schema':'bowl_obb_geometry_v1','states':{}}
    try:
        for sid in [int(x) for x in args.state_ids.split(',') if x.strip()]:
            env.seed(7777); env.reset(); obs=env.set_init_state(states[sid])
            from evaluate_libero import DUMMY_ACTION
            for _ in range(args.settle): obs,_,_,_=env.step(DUMMY_ACTION)
            body_names=[str(x) for x in env.sim.model.body_names]; geom_names=[str(x) for x in env.sim.model.geom_names]
            tid=next(i for i,n in enumerate(body_names) if n=='akita_black_bowl_1_main')
            boxes=[]
            for gi in range(len(geom_names)):
                if int(env.sim.model.geom_bodyid[gi])==tid and int(env.sim.model.geom_type[gi])==6:
                    boxes.append({'name':geom_names[gi],'center':np.asarray(env.sim.data.geom_xpos[gi],float).copy(),'R':np.asarray(env.sim.data.geom_xmat[gi],float).reshape(3,3).copy(),'size':np.asarray(env.sim.model.geom_size[gi],float).copy()})
            pads=[]
            for gi,n in enumerate(geom_names):
                if 'finger' in n and 'pad_collision' in n:
                    pads.append({'name':n,'center':np.asarray(env.sim.data.geom_xpos[gi],float).copy(),'R':np.asarray(env.sim.data.geom_xmat[gi],float).reshape(3,3).copy(),'size':np.asarray(env.sim.model.geom_size[gi],float).copy()})
            if len(boxes)!=40: raise RuntimeError(f'expected 40 boxes, got {len(boxes)}')
            if len(pads)!=2: raise RuntimeError(f'expected 2 finger pads, got {len(pads)}')
            pad_results=[]
            for pad in pads:
                best=None
                for b in boxes:
                    loc=b['R'].T@(pad['center']-b['center'])
                    q=np.clip(loc,-b['size'],b['size']); delta=loc-q; dist=float(np.linalg.norm(delta))
                    if best is None or dist<best['distance']:
                        normal_local=delta/(np.linalg.norm(delta)+1e-12) if np.linalg.norm(delta)>1e-12 else np.zeros(3)
                        best={'distance':dist,'box':b['name'],'local':loc.tolist(),'normal_world':(b['R']@normal_local).tolist()}
                pad_results.append({'pad':pad['name'],'center':pad['center'].tolist(),'nearest':best})
            closing=pads[1]['center']-pads[0]['center']; closing=closing/(np.linalg.norm(closing)+1e-12)
            angles=[]
            for pr in pad_results:
                n=np.asarray(pr['nearest']['normal_world'],float); angles.append(float(np.arccos(np.clip(abs(np.dot(closing,n)),-1,1))))
            out['states'][str(sid)]={'target_body': 'akita_black_bowl_1_main','n_boxes':len(boxes),'n_pads':len(pads),
                'pad_results':pad_results,'closing_axis':closing.tolist(),'closing_to_nearest_normal_angle_rad':angles,
                'box_extent_z':[float(b['center'][2]-b['size'][2]) for b in boxes][:3],
                'note':'point-to-OBB distance; no mesh approximation; pads are pad_collision geoms'}
            print(json.dumps({'state':sid,'pad_distances':[p['nearest']['distance'] for p in pad_results],'angles_deg':[a*180/np.pi for a in angles]}))
    finally: env.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__': main()
