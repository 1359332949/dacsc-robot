"""Execution-aligned action metrics for CoRP controller commands.

Rotation channels are treated as rotation vectors (axis-angle) in controller
units; this must be checked against the LIBERO controller mapping before any
claim about physical rotational error.
"""
from __future__ import annotations
import numpy as np


def execution_clip(action):
    """Clip every executed controller dimension to the simulator range."""
    a=np.asarray(action,dtype=np.float64)
    if a.shape[-1]!=7: raise ValueError(f'expected last dim 7, got {a.shape}')
    return np.clip(a,-1.0,1.0)


def _rotation_matrix(rotvec):
    r=np.asarray(rotvec,dtype=np.float64); theta=float(np.linalg.norm(r))
    if theta<1e-12: return np.eye(3)
    k=r/theta; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)


def rotation_geodesic(a,b):
    """SO(3) geodesic angle between two rotation-vector commands."""
    R=_rotation_matrix(a).T@_rotation_matrix(b)
    c=np.clip((np.trace(R)-1.0)/2.0,-1.0,1.0)
    return float(np.arccos(c))


DEFAULT_OSC_POSE_OUTPUT_MIN = np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], dtype=np.float64)
DEFAULT_OSC_POSE_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float64)


def controller_physical(action, output_min=None, output_max=None):
    """Map executed [-1,1] controller input to physical OSC_POSE units.

    XYZ becomes metres; rotation becomes a scaled axis-angle rotation vector in
    radians. The gripper dimension is not part of OSC_POSE and is unchanged.
    """
    a=execution_clip(action)
    if output_min is None: output_min=DEFAULT_OSC_POSE_OUTPUT_MIN
    if output_max is None: output_max=DEFAULT_OSC_POSE_OUTPUT_MAX
    out=np.array(a,copy=True)
    out[...,:6]=0.5*(a[...,:6]+1.0)*(np.asarray(output_max)-np.asarray(output_min))+np.asarray(output_min)
    return out


def action_error(pred,ref,executed=True,horizon=12,physical=True):
    """Per-step component metrics; returns means over the execution window."""
    p=np.asarray(pred,dtype=np.float64); r=np.asarray(ref,dtype=np.float64)
    if p.shape!=r.shape: raise ValueError(f'shape mismatch {p.shape} vs {r.shape}')
    p=p[:horizon]; r=r[:horizon]
    if executed: p=execution_clip(p); r=execution_clip(r)
    norm_xyz=np.linalg.norm(p[:,:3]-r[:,:3],axis=-1)
    norm_rot=np.array([rotation_geodesic(p[i,3:6],r[i,3:6]) for i in range(len(p))])
    gripper=np.abs(p[:,6]-r[:,6])
    out={'xyz_norm_mean':float(norm_xyz.mean()),'rotation_norm_geodesic_mean':float(norm_rot.mean()),
         'gripper_abs_mean':float(gripper.mean()),'executed':bool(executed),'horizon':int(len(p))}
    if physical:
        pp=controller_physical(p); rr=controller_physical(r)
        xyz_m=np.linalg.norm(pp[:,:3]-rr[:,:3],axis=-1)
        rot_rad=np.array([rotation_geodesic(pp[i,3:6],rr[i,3:6]) for i in range(len(pp))])
        out.update({'xyz_m_mean':float(xyz_m.mean()),'rotation_rad_geodesic_mean':float(rot_rad.mean())})
    return out


def state_balanced_mean(records,key):
    """Average chunks within state first, then average states equally."""
    by={}
    for r in records: by.setdefault(r['state_id'],[]).append(float(r[key]))
    vals=[float(np.mean(v)) for v in by.values()]
    return float(np.mean(vals)) if vals else None
