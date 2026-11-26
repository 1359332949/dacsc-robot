"""Real-image oracle against verbatim upstream AST; no policy outcomes used."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from phase3_smolvla_shift import corruption_functions,sha
from dacsc.plus_glass_exact import glass_blur_exact


def state_equal(a,b):return all(np.array_equal(x,y) for x,y in zip(a,b))


def main(a):
    original=corruption_functions()['glass_blur'];checks=[]
    # Compile on a disposable image, before timing or recording test state.
    glass_blur_exact(Image.fromarray(np.zeros((256,256,3),dtype=np.uint8)),5)
    samples=[]
    for task in range(10):
        with np.load(a.source/f'task{task}_initial.npz',allow_pickle=False) as z:
            for key in ('agentview_image','robot0_eye_in_hand_image'):
                samples.append((f'task{task}_{key}',z[key].copy(),5))
    for severity in range(1,11):
        for size in (224,256):
            pixels=np.random.default_rng(severity*1000+size).integers(0,256,size=(size,size,3),dtype=np.uint8)
            samples.append((f'random{size}',pixels,severity))
    for index,(name,pixels,severity) in enumerate(samples):
        seed=2026112600+index;image=Image.fromarray(pixels)
        np.random.seed(seed);t=time.perf_counter();expected=original(image,severity);slow=time.perf_counter()-t;s0=np.random.get_state()
        np.random.seed(seed);t=time.perf_counter();actual=glass_blur_exact(image,severity);fast=time.perf_counter()-t;s1=np.random.get_state()
        assert np.array_equal(expected,actual),(name,severity,float(np.max(np.abs(expected-actual))))
        assert state_equal(s0,s1),(name,severity,'MT19937 state changed')
        checks.append(dict(name=name,severity=severity,shape=list(pixels.shape),seed=seed,
                           float_output_bit_exact=True,uint8_output_bit_exact=True,rng_state_exact=True,
                           original_seconds=slow,accelerated_seconds=fast))
        print('EXACT',index+1,len(samples),name,severity,round(slow,3),round(fast,3),flush=True)
    result={'status':'PASS','checks':checks,'accelerator_sha256':sha(Path(__file__).parents[1]/'dacsc/plus_glass_exact.py'),
            'oracle':'verbatim official Plus AST, including aliased pixel assignment; both floating outputs and RNG states exact',
            'median_speedup':float(np.median([c['original_seconds']/c['accelerated_seconds'] for c in checks]))}
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
