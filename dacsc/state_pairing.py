"""Fail-closed MuJoCo state pairing checks for LIBERO-Plus V1.1.

Design rules:
- a required field missing on either side is a failure, never an implicit pass;
- one side empty and the other non-empty is a failure;
- shapes and finiteness are checked before numerical tolerances;
- body orientation and EEF pose are compared explicitly;
- camera comparison distinguishes "expected same" from "expected changed".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

REQUIRED_ARRAYS = ("qpos", "qvel", "time", "body_xpos", "body_xquat", "eef_pos")
OPTIONAL_EMPTY_OK = ("act", "mocap_pos", "mocap_quat")
DEFAULT_TOL = {"qpos": 1e-6, "qvel": 1e-6, "act": 1e-6, "time": 1e-9,
               "body_xpos": 1e-5, "eef_pos": 1e-5, "mocap": 1e-6,
               "body_xquat_rad": 1e-5}


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).copy()


def _find_eef(names: list[str]) -> str | None:
    for cand in ("gripper0_eef", "robot0_eef"):
        if cand in names:
            return cand
    return None


def snapshot(env) -> dict[str, Any]:
    sim = env.sim
    names = [str(n) for n in sim.model.body_names]
    cam_names = [str(n) for n in sim.model.camera_names]
    eef_name = _find_eef(names)
    eef_pos = None
    if eef_name is not None:
        eef_pos = _arr(sim.data.body_xpos[names.index(eef_name)]).tolist()
    out = {
        "body_names": names,
        "camera_names": cam_names,
        "eef_name": eef_name,
        "qpos": _arr(sim.data.qpos).tolist(),
        "qvel": _arr(sim.data.qvel).tolist(),
        "act": _arr(sim.data.act).tolist() if getattr(sim.data, "act", None) is not None else [],
        "time": float(sim.data.time),
        "body_xpos": _arr(sim.data.body_xpos).tolist(),
        "body_xquat": _arr(sim.data.body_xquat).tolist(),
        "eef_pos": eef_pos,
        "cam_pos": _arr(sim.model.cam_pos).tolist(),
        "cam_quat": _arr(sim.model.cam_quat).tolist(),
        "cam_fovy": _arr(sim.model.cam_fovy).tolist(),
        "cam_xpos": _arr(sim.data.cam_xpos).tolist(),
        "cam_xmat": _arr(sim.data.cam_xmat).tolist(),
        "model_signature": model_signature(env),
    }
    mocap_pos = getattr(sim.data, "mocap_pos", None)
    mocap_quat = getattr(sim.data, "mocap_quat", None)
    out["mocap_pos"] = _arr(mocap_pos).tolist() if mocap_pos is not None else []
    out["mocap_quat"] = _arr(mocap_quat).tolist() if mocap_quat is not None else []
    return out


def model_signature(env) -> dict[str, Any]:
    m = env.sim.model
    return {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "nbody": int(m.nbody), "njnt": int(m.njnt), "ngeom": int(m.ngeom),
        "body_names": [str(x) for x in m.body_names],
        "joint_names": [str(x) for x in m.joint_names],
        "joint_types": [int(x) for x in m.jnt_type],
        "jnt_qposadr": [int(x) for x in m.jnt_qposadr],
        "jnt_dofadr": [int(x) for x in m.jnt_dofadr],
        "actuator_names": [str(x) for x in m.actuator_names],
        "geom_names": [str(x) for x in m.geom_names],
        "geom_bodyid": [int(x) for x in m.geom_bodyid],
        "geom_type": [int(x) for x in m.geom_type],
        "geom_size": _arr(m.geom_size).round(12).tolist(),
        "geom_pos": _arr(m.geom_pos).round(12).tolist(),
        "geom_quat": _arr(m.geom_quat).round(12).tolist(),
        "geom_contype": [int(x) for x in m.geom_contype],
        "geom_conaffinity": [int(x) for x in m.geom_conaffinity],
        "geom_condim": [int(x) for x in m.geom_condim],
        "geom_friction": _arr(m.geom_friction).round(12).tolist(),
        "geom_solref": _arr(m.geom_solref).round(12).tolist(),
        "geom_solimp": _arr(m.geom_solimp).round(12).tolist(),
        "geom_margin": _arr(m.geom_margin).round(12).tolist(),
        "geom_gap": _arr(m.geom_gap).round(12).tolist(),
        "body_mass": _arr(m.body_mass).round(12).tolist(),
        "body_inertia": _arr(m.body_inertia).round(12).tolist(),
        "dof_damping": _arr(m.dof_damping).round(12).tolist(),
        "dof_frictionloss": _arr(m.dof_frictionloss).round(12).tolist(),
        "dof_armature": _arr(m.dof_armature).round(12).tolist(),
        "actuator_gainprm": _arr(m.actuator_gainprm).round(12).tolist(),
        "actuator_biasprm": _arr(m.actuator_biasprm).round(12).tolist(),
        "actuator_gear": _arr(m.actuator_gear).round(12).tolist(),
        "actuator_ctrlrange": _arr(m.actuator_ctrlrange).round(12).tolist(),
    }


def model_compatible(a: dict[str, Any], b: dict[str, Any], strict_bodies: bool = True,
                     strict_geometry: bool = False, strict_physics: bool = False) -> dict[str, Any]:
    sa, sb = a["model_signature"], b["model_signature"]
    checks = {}
    for key in ("nq", "nv", "nu", "njnt", "jnt_qposadr", "jnt_dofadr",
                "joint_names", "joint_types", "actuator_names"):
        checks[key] = sa.get(key) == sb.get(key)
    if strict_bodies:
        checks["nbody"] = sa.get("nbody") == sb.get("nbody")
        checks["body_names"] = sa.get("body_names") == sb.get("body_names")
    if strict_geometry:
        checks["ngeom"] = sa.get("ngeom") == sb.get("ngeom")
        for key in ("geom_names", "geom_bodyid", "geom_type", "geom_size", "geom_pos", "geom_quat"):
            checks[key] = sa.get(key) == sb.get(key)
    if strict_physics:
        for key in ("geom_contype", "geom_conaffinity", "geom_condim", "geom_friction",
                    "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
                    "body_mass", "body_inertia", "dof_damping", "dof_frictionloss",
                    "dof_armature", "actuator_gainprm", "actuator_biasprm",
                    "actuator_gear", "actuator_ctrlrange"):
            checks[key] = sa.get(key) == sb.get(key)
    return {"checks": checks, "pass": all(checks.values())}


def restore(env, snap: dict[str, Any]) -> None:
    """Restore numerical state; model compatibility must be checked beforehand."""
    sim = env.sim
    def put(dst, src):
        dst = np.asarray(dst); src = np.asarray(src)
        if dst.shape != src.shape:
            raise ValueError(f"state shape mismatch: {dst.shape} vs {src.shape}")
        dst[...] = src
    put(sim.data.qpos, snap["qpos"])
    put(sim.data.qvel, snap["qvel"])
    if getattr(sim.data, "act", None) is not None and len(snap.get("act", [])):
        put(sim.data.act, snap["act"])
    if getattr(sim.data, "mocap_pos", None) is not None and len(snap.get("mocap_pos", [])):
        put(sim.data.mocap_pos, snap["mocap_pos"])
    if getattr(sim.data, "mocap_quat", None) is not None and len(snap.get("mocap_quat", [])):
        put(sim.data.mocap_quat, snap["mocap_quat"])
    sim.data.time = float(snap["time"])
    sim.forward()


def _quat_angle_rad(q1, q2) -> float:
    q1 = np.asarray(q1, dtype=np.float64); q2 = np.asarray(q2, dtype=np.float64)
    n1 = np.linalg.norm(q1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(q2, axis=-1, keepdims=True)
    if np.any(n1 < 1e-12) or np.any(n2 < 1e-12):
        return float("inf")
    q1 = q1 / n1; q2 = q2 / n2
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return float((2.0 * np.arccos(dot)).max())


def _array_check(name, a, b, tol, optional_empty=False):
    x = np.asarray(a) if a is not None else None
    y = np.asarray(b) if b is not None else None
    if x is None or y is None:
        return {"present": False, "shape": False, "finite": False, "error": float("inf"), "pass": False}
    if x.size == 0 or y.size == 0:
        if optional_empty and x.size == 0 and y.size == 0:
            return {"present": True, "shape": True, "finite": True, "error": 0.0, "pass": True}
        return {"present": True, "shape": False, "finite": False, "error": float("inf"), "pass": False}
    shape = x.shape == y.shape
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    err = float(np.max(np.abs(x - y))) if shape and finite else float("inf")
    return {"present": True, "shape": shape, "finite": finite, "error": err,
            "pass": bool(shape and finite and err <= tol[name])}


def compare(a: dict[str, Any], b: dict[str, Any], tol: dict[str, float] | None = None) -> dict[str, Any]:
    tol = {**DEFAULT_TOL, **(tol or {})}
    checks = {}
    for key in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat"):
        checks[key] = _array_check(key if key != "mocap_pos" and key != "mocap_quat" else "mocap",
                                   a.get(key), b.get(key), tol,
                                   optional_empty=key in OPTIONAL_EMPTY_OK)
    checks["time"] = _array_check("time", [a.get("time")], [b.get("time")], tol)
    checks["body_xpos"] = _array_check("body_xpos", a.get("body_xpos"), b.get("body_xpos"), tol)
    checks["eef_pos"] = _array_check("eef_pos", a.get("eef_pos"), b.get("eef_pos"), tol)
    # quaternion equivalence (q == -q)
    qa = a.get("body_xquat"); qb = b.get("body_xquat")
    if qa is None or qb is None or np.asarray(qa).size == 0 or np.asarray(qb).size == 0:
        checks["body_xquat"] = {"present": False, "pass": False, "error": float("inf")}
    else:
        qa = np.asarray(qa); qb = np.asarray(qb)
        if qa.shape != qb.shape:
            checks["body_xquat"] = {"present": True, "shape": False, "pass": False, "error": float("inf")}
        else:
            ang = _quat_angle_rad(qa, qb)
            checks["body_xquat"] = {"present": True, "shape": True, "finite": bool(np.isfinite(ang)),
                                    "error_rad": ang, "pass": bool(np.isfinite(ang) and ang <= tol["body_xquat_rad"])}
    names_ok = a.get("body_names") == b.get("body_names")
    cam_names_ok = a.get("camera_names") == b.get("camera_names")
    eef_ok = bool(a.get("eef_name")) and a.get("eef_name") == b.get("eef_name")
    checks["body_names"] = {"pass": bool(names_ok)}
    checks["camera_names"] = {"pass": bool(cam_names_ok)}
    checks["eef_name"] = {"pass": eef_ok}
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks.values())}


def camera_compare(a: dict[str, Any], b: dict[str, Any], expect_equal: bool,
                   tol: float = 1e-9, target_cameras: list[str] | None = None) -> dict[str, Any]:
    """Compare camera arrays; for expect_equal=False require target cameras to change.

    target_cameras defaults to all cameras (legacy behaviour).  A Camera variant
    may legitimately change only one policy view; non-target cameras are allowed
    to stay identical.
    """
    names = list(a.get("camera_names", []))
    names_ok = names == b.get("camera_names") and bool(names)
    checks = {"names": bool(names_ok)}
    if target_cameras is None:
        target_cameras = names
    missing = [c for c in target_cameras if c not in names]
    checks["targets_present"] = not missing
    diffs: dict[str, dict[str, float]] = {c: {} for c in names}
    shape_ok = True; finite = True
    for key in ("cam_pos", "cam_quat", "cam_fovy", "cam_xpos", "cam_xmat"):
        x = np.asarray(a.get(key, []), dtype=np.float64)
        y = np.asarray(b.get(key, []), dtype=np.float64)
        if x.size == 0 or y.size == 0 or x.shape != y.shape:
            shape_ok = False
            for c in names: diffs[c][key] = float("inf")
            continue
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            finite = False
            for c in names: diffs[c][key] = float("inf")
            continue
        for i, c in enumerate(names):
            diffs[c][key] = float(np.max(np.abs(x[i] - y[i])))
    checks["shape"] = shape_ok
    checks["finite"] = finite
    if expect_equal:
        checks["expected"] = bool(shape_ok and finite and all(
            v <= tol for c in names for v in diffs[c].values()))
    else:
        changed = {c: any(diffs[c][k] > tol for k in diffs[c]) for c in names}
        checks["expected"] = bool(shape_ok and finite and not missing and all(
            changed[c] for c in target_cameras))
        non_target = [c for c in names if c not in target_cameras]
        checks["non_target_unchanged"] = all(not changed[c] for c in non_target)
    return {"checks": checks, "diffs": diffs, "target_cameras": list(target_cameras),
            "pass": all(checks.values())}


def digest(snap: dict[str, Any]) -> str:
    payload = {
        "body_names": snap.get("body_names"),
        "camera_names": snap.get("camera_names"),
        "eef_name": snap.get("eef_name"),
        "qpos": np.round(np.asarray(snap.get("qpos", [])), 10).tolist(),
        "qvel": np.round(np.asarray(snap.get("qvel", [])), 10).tolist(),
        "act": np.round(np.asarray(snap.get("act", [])), 10).tolist(),
        "mocap_pos": np.round(np.asarray(snap.get("mocap_pos", [])), 10).tolist(),
        "mocap_quat": np.round(np.asarray(snap.get("mocap_quat", [])), 10).tolist(),
        "time": round(float(snap.get("time", 0.0)), 10),
        "body_xpos": np.round(np.asarray(snap.get("body_xpos", [])), 8).tolist(),
        "body_xquat": np.round(np.asarray(snap.get("body_xquat", [])), 8).tolist(),
        "eef_pos": np.round(np.asarray(snap.get("eef_pos") or []), 8).tolist(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
