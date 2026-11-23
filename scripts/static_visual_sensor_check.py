"""Native official sensor functions parity with step-keyed RNG; no benchmark episodes."""
import sys,argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import sensor_transform,isolated_seed,seed_for

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--libero-src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();sys.path.insert(0,str(a.libero_src))
    import libero.libero.envs.env_wrapper as w
    from PIL import Image
    # Real image already recorded in old Source; resize only to make the standalone kernel check cheap.
    z=np.load(ROOT/'runs/v2_migration_inputs/source_fit_state0_seed2014/episode.npz')
    im=np.asarray(Image.fromarray(z['rgb_agentview'][0]).resize((32,32)));eye=np.full_like(im,77)
    row=[];functions=[w.motion_blur,w.gaussian_blur,w.zoom_blur,w.fog,w.glass_blur]
    for code in (1,11,21,31,41):
        family=(code-1)//10;obs={'agentview_image':im,'robot0_eye_in_hand_image':eye}
        actual=sensor_transform(obs,code,123,12,w)
        with isolated_seed(seed_for(123,12,'agentview')):expected=functions[family](Image.fromarray(im),severity=1)
        if family!=0:expected=expected.astype(np.uint8)
        error=float(np.abs(actual['agentview_image'].astype(float)-expected.astype(float)).max())
        with isolated_seed(4):np.random.rand(1000)
        repeat=sensor_transform(obs,code,123,12,w)
        assert error==0 and np.array_equal(actual['agentview_image'],repeat['agentview_image']) and np.array_equal(actual['robot0_eye_in_hand_image'],eye)
        row.append({'noise_code':code,'native_kernel_max_abs':error,'repeat_exact':True,'eye_unchanged':True})
    a.out.write_text(json.dumps({'pass':True,'sensor_native_function_checks':row,'image_size':32,'scope':'kernel/RNG parity only, not benchmark severity-performance validation'},indent=2)+'\n');print(row)
if __name__=='__main__':main()
