"""Re-render exact t=0 RGB for illustrative flow-gradient overlays; no actions."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import authorize,file_hash,array_hash,sensor_transform,EnvironmentRNG

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--gradients',type=Path,required=True)
    ap.add_argument('--libero-src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());h=json.loads((a.manifest.parent/'manifest_hashes.json').read_text())[a.manifest.name]
    records=[json.loads(p.read_text()) for p in sorted(a.gradients.glob('*.json'))]
    if len(records)!=7 or any(r['status']!='completed' for r in records):raise ValueError('seven completed illustrative gradients required')
    cases={c['case_id']:c for c in m['cases']}
    for r in records:authorize(m,cases[r['case_id']],'gradient map render',h,a.manifest)
    sys.path[:0]=[str(a.libero_src),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
    import torch
    from PIL import Image,ImageDraw
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.env_wrapper as wrapper
    a.out.mkdir(parents=True,exist_ok=True);panels=[]
    # Seeds read from original V2 result receipts, not inferred from class labels.
    for r in records:
        c=cases[r['case_id']];ref=json.loads((ROOT/'docs/data/static_visual_v2_suite/final/results'/(r['case_id']+'.json')).read_text())
        if file_hash(ROOT/'docs/data/static_visual_v2_suite/final/results'/(r['case_id']+'.json'))!=r['baseline_result_sha256']:raise ValueError('baseline changed')
        seeds=ref['seeds'];rng=EnvironmentRNG(seeds['environment_seed']);env=None
        try:
            with rng.use():
                env=OffScreenRenderEnv(bddl_file_name=str(a.libero_src/'libero/libero/bddl_files'/c['suite']/(c['logical_task_name']+'.bddl')),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
                env.noise=0;env.seed(seeds['environment_seed']);env.reset()
                initial=torch.load(a.libero_src/c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
                obs=env.set_init_state(initial)
                for _ in range(SETTLING_STEPS):obs,_,_,_=env.step(DUMMY_ACTION)
            obs=sensor_transform(obs,int(c['runtime_params']['noise']),seeds['sensor_noise_seed'],0,wrapper)
            if array_hash(np.asarray(obs['agentview_image']))!=r['initial_rgb_hashes']['agentview']:raise ValueError('RGB mismatch')
            original=Image.fromarray(np.asarray(obs['agentview_image'])[::-1,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR)
            grad=np.array(r['gradient']['patch_gradient_norm']).reshape(16,16);grad=grad/(grad.max()+1e-12)
            heat=np.stack([grad,np.zeros_like(grad),1-grad],-1)
            heat=Image.fromarray((heat*255).astype(np.uint8)).resize((224,224),Image.Resampling.NEAREST)
            overlay=Image.blend(original,heat,.45)
            step=np.array(r['gradient']['gate_step']).reshape(16,16);step=step/max(np.abs(step).max(),1e-12)
            rgb=np.ones((16,16,3));rgb[:,:,0]-=np.maximum(-step,0);rgb[:,:,1]-=np.abs(step);rgb[:,:,2]-=np.maximum(step,0)
            gate=Image.fromarray((rgb*255).astype(np.uint8)).resize((224,224),Image.Resampling.NEAREST)
            panel=Image.new('RGB',(672,264),'white');panel.paste(original,(0,40));panel.paste(overlay,(224,40));panel.paste(gate,(448,40))
            d=ImageDraw.Draw(panel);d.text((5,3),f"{r['factor']} / {'success' if r['baseline_success'] else 'failure'} / {r['case_id'][:12]}",fill='black')
            d.text((5,20),'Original          Patch gradient (blue low / red high)          Gate step (red up / blue down)',fill='black')
            panel.save(a.out/(r['case_id']+'.png'));panels.append(panel)
        finally:
            if env is not None:env.close()
    atlas=Image.new('RGB',(672,264*len(panels)),'white')
    for i,p in enumerate(panels):atlas.paste(p,(0,264*i))
    atlas.save(a.out/'gradient_atlas.png');print('rendered seven exact-RGB gradient maps; no causal mask annotation')
if __name__=='__main__':main()
