"""Descriptive flow geometry; training interpolation is not a straightness guarantee."""
import numpy as np


def geometry(velocities, eps=1e-12):
    v=np.asarray(velocities,dtype=np.float64)
    if v.ndim!=3 or v.shape[0]<2 or not np.isfinite(v).all():
        raise ValueError('expected finite [flow_steps,horizon,channels]')
    mean=v.mean(0); curvature=float(np.mean((v-mean)**2)); energy=float(np.mean(v*v))
    f=v.reshape(len(v),-1); norms=np.linalg.norm(f,axis=1)
    valid=(norms[:-1]>eps)&(norms[1:]>eps)
    cos=np.sum(f[:-1]*f[1:],axis=1)/np.maximum(norms[:-1]*norms[1:],eps)
    path=float(norms.mean()); chord=float(np.linalg.norm(f.mean(0)))
    return {'velocity_variance':curvature,'relative_variance':curvature/max(energy,eps),
            'velocity_energy':energy,'turning':float(np.mean(1-np.clip(cos[valid],-1,1))) if valid.any() else None,
            'valid_turn_pairs':int(valid.sum()),'path_excess':path/max(chord,eps)-1 if path>eps else None,
            'zero_energy':energy<=eps}


def auc(y,score):
    y=np.asarray(y,bool);s=np.asarray(score,float)
    if not np.isfinite(s).all():raise ValueError('nonfinite score')
    a=s[y];b=s[~y]
    if not len(a) or not len(b):return None
    d=a[:,None]-b[None,:]
    return float(np.mean((d>0)+.5*(d==0)))
