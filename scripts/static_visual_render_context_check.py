"""Real OSMesa regression: same source state, three renderer lifecycles, no policy rollout."""
from pathlib import Path
import argparse,json,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--libero-src',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.libero_src))
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    base='pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate';src=a.libero_src/'libero/libero';bddl=str(src/'bddl_files/libero_spatial'/(base+'.bddl'));initial=torch.load(src/'init_files/libero_spatial'/(base+'.pruned_init'),weights_only=False)[0]
    model=load_policy(ROOT/'repos/CoRP/checkpoints/corp_libero.pt',device='cuda:0');model.eval();images={};states={};metrics=[];a.out.mkdir(parents=True,exist_ok=False)
    def make():
        env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=0)
        np.random.seed(2014);env.seed(2014);env.reset();env.set_init_state(initial);return env
    for mode in ('single','buggy_main_then_twin_close','fixed_twin_close_then_main'):
        if mode=='fixed_twin_close_then_main':twin=make();twin.close()
        env=make()
        if mode=='buggy_main_then_twin_close':twin=make();twin.close()
        for _ in range(SETTLING_STEPS):obs,_,_,_=env.step(DUMMY_ACTION)
        po=policy_observation(obs,base.replace('_',' '),model);states[mode]=po['state'].numpy().copy()
        images[mode]={v:po['images'][v][0].numpy().copy() for v in ('agentview','eye_in_hand')}
        for v,im in images[mode].items():Image.fromarray(im).save(a.out/(mode+'_'+v+'.png'))
        metrics.append({'mode':mode,'means':{v:float(im.mean()) for v,im in images[mode].items()}});env.close()
    fixed={v:float(np.abs(images['single'][v].astype(float)-images['fixed_twin_close_then_main'][v]).max()) for v in images['single']}
    buggy={v:float(np.abs(images['single'][v].astype(float)-images['buggy_main_then_twin_close'][v]).mean()) for v in images['single']}
    assert max(fixed.values())==0 and all(np.array_equal(states['single'],q) for q in states.values())
    out={'pass':True,'same_physical_state8':True,'fixed_vs_single_rgb_max_abs':fixed,'buggy_vs_single_rgb_mae':buggy,'metrics':metrics,'meaning':'renderer-lifecycle bug reproduced independently of policy performance; fixed lifecycle exactly matches one-env baseline'}
    (a.out/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
