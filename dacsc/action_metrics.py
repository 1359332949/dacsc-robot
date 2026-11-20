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


def action_error(pred,ref,executed=True,horizon=12):
    """Per-step component metrics; returns means over the execution window."""
    p=np.asarray(pred,dtype=np.float64); r=np.asarray(ref,dtype=np.float64)
    if p.shape!=r.shape: raise ValueError(f'shape mismatch {p.shape} vs {r.shape}')
    p=p[:horizon]; r=r[:horizon]
    if executed: p=execution_clip(p); r=execution_clip(r)
    xyz=np.linalg.norm(p[:,:3]-r[:,:3],axis=-1)
    rot=np.array([rotation_geodesic(p[i,3:6],r[i,3:6]) for i in range(len(p))])
    grip=np.abs(p[:,6]-r[:,6])
    return {'xyz_mean':float(xyz.mean()),'rotation_geodesic_mean':float(rot.mean()),
            'gripper_abs_mean':float(grip.mean()),'executed':bool(executed),'horizon':int(len(p))}


def state_balanced_mean(records,key):
    """Average chunks within state first, then average states equally."""
    by={}
    for r in records: by.setdefault(r['state_id'],[]).append(float(r[key]))
    vals=[float(np.mean(v)) for v in by.values()]
    return float(np.mean(vals)) if vals else None
