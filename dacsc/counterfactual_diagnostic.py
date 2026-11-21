"""Pure array contracts/metrics; controller-command differences are not risks."""
import numpy as np


def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))


def require_current_pair(clean,shift,index):
    q=np.asarray(clean['state8'][index])
    if q.shape!=(8,) or not np.isfinite(q).all():raise ValueError('invalid current proprio')
    if not np.array_equal(q,shift['state8'][index]):raise ValueError('unequal current proprio')
    if clean['boundary_step'][index]!=shift['boundary_step'][index]:raise ValueError('unequal boundary time')
    return q.copy()


def command_difference(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=(12,7) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid command')
    out={}
    for name,x,y in [('preclip',a,b),('postclip',np.clip(a,-1,1),np.clip(b,-1,1))]:
        d=x-y;out[name]={'all_rms':rms(d),'xyz_l2_mean':float(np.linalg.norm(d[:,:3],axis=1).mean()),
            'rotation_channels_l2_mean':float(np.linalg.norm(d[:,3:6],axis=1).mean()),
            'gripper_abs_mean':float(np.abs(d[:,6]).mean()),'per_position_l2':np.linalg.norm(d,axis=1).tolist()}
    out['candidate_preclip_fraction_outside_unit']=float((np.abs(a)>1).mean())
    out['reference_preclip_fraction_outside_unit']=float((np.abs(b)>1).mean())
    return out


def separation(a,b,sa,sb):
    clean=float(np.linalg.norm(np.asarray(a)-b));shift=float(np.linalg.norm(np.asarray(sa)-sb))
    return {'clean_l2':clean,'shift_l2':shift,'shift_over_clean':shift/clean if clean>1e-12 else None,
            'clean_distinguishable_numeric':clean>1e-12,'shift_distinguishable_numeric':shift>1e-12,
            'scope':'descriptive geometry, not physical state-label accuracy or ACPC SR'}


def interaction(cc,sc,cs,ss):return rms(np.asarray(ss)-sc-cs+cc)
