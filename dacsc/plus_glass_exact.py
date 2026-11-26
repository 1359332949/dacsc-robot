"""Integer-loop acceleration of official Plus glass_blur, not a new corruption.

Keep skimage filtering and numpy MT19937 draws unchanged. The official numpy
tuple assignment uses aliased pixel views (copy then copy back, not a true
swap); the compiled loop deliberately reproduces that behavior exactly.
Use only after the executable oracle compares pixels AND final RNG state.
"""
import numpy as np
from numba import njit
from skimage.filters import gaussian


@njit(cache=True)
def _copy_pixels(x,offsets,delta):
    height,width=x.shape[:2]
    for iteration in range(offsets.shape[0]):
        for hi in range(height-2*delta):
            h=height-delta-hi
            for wi in range(width-2*delta):
                w=width-delta-wi
                dx,dy=offsets[iteration,hi,wi,0],offsets[iteration,hi,wi,1]
                hp,wp=h+dy,w+dx
                for channel in range(3):
                    x[h,w,channel]=x[hp,wp,channel]
                    x[hp,wp,channel]=x[h,w,channel]
    return x


def glass_blur_exact(image,severity=1):
    sigma,delta,iterations=[(.5,1,3),(.7,1,3),(.9,2,3),(1.,2,2),(1.1,3,2),
                          (1.3,3,2),(1.5,4,2),(1.8,4,2),(2.2,5,1),(2.5,5,1)][severity-1]
    x=np.uint8(gaussian(np.array(image)/255.,sigma=sigma,channel_axis=-1)*255)
    h,w=x.shape[:2]
    offsets=np.random.randint(-delta,delta,size=(iterations,h-2*delta,w-2*delta,2))
    _copy_pixels(x,offsets,delta)
    return np.clip(gaussian(x/255.,sigma=sigma,channel_axis=-1),0,1)*255
