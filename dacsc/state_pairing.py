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
OPTIONAL_EMPTY_OK = ("act", "mocap_pos", "mocap_quat", "ctrl", "qacc_warmstart",
                    "qfrc_applied", "xfrc_applied", "eq_active")
EXTRA_STATE_ARRAYS = ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
                      "eq_active", "userdata")
DEFAULT_TOL = {"qpos": 1e-6, "qvel": 1e-6, "act": 1e-6, "time": 1e-9,
               "body_xpos": 1e-4, "eef_pos": 1e-4, "mocap": 1e-6,
               "body_xquat_rad": 1e-4, "ctrl": 1e-6, "qacc_warmstart": 1e-6,
               "qfrc_applied": 1e-6, "xfrc_applied": 1e-6, "eq_active": 1e-6}


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
    site_names = [str(n) for n in sim.model.site_names]
    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", None)
    eef_site_id = int(getattr(robots[0], "eef_site_id")) if robots and hasattr(robots[0], "eef_site_id") else None
    eef_site_name = site_names[eef_site_id] if eef_site_id is not None and eef_site_id < len(site_names) else None
    out = {
        "body_names": names,
        "site_names": site_names,
        "eef_site_id": eef_site_id,
        "eef_site_name": eef_site_name,
        "site_bodyid": [int(x) for x in sim.model.site_bodyid],
        "site_pos": _arr(sim.model.site_pos).tolist(),
        "site_quat": _arr(sim.model.site_quat).tolist(),
        "site_size": _arr(sim.model.site_size).tolist(),
        "site_type": [int(x) for x in sim.model.site_type],
        "site_xpos": _arr(sim.data.site_xpos).tolist(),
        "site_xmat": _arr(sim.data.site_xmat).tolist(),
        "controller": controller_snapshot(env),
        "mj_state": sim.get_state(),
        "camera_names": cam_names,
        "eef_name": eef_name,
        "qpos": _arr(sim.data.qpos).tolist(),
        "qvel": _arr(sim.data.qvel).tolist(),
        "act": _arr(sim.data.act).tolist() if getattr(sim.data, "act", None) is not None else [],
        "time": float(sim.data.time),
        "body_xpos": _arr(sim.data.body_xpos).tolist(),
        "body_xquat": _arr(sim.data.body_xquat).tolist(),
        "eef_pos": eef_pos,
        "policy_eef_pos": (_arr(sim.data.site_xpos[eef_site_id]).tolist() if eef_site_id is not None else None),
        "policy_eef_quat": (_arr(sim.data.body_xquat[names.index(eef_name)]).tolist() if eef_name is not None else None),
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
    for key in EXTRA_STATE_ARRAYS:
        value = getattr(sim.data, key, None)
        out[key] = _arr(value).tolist() if value is not None else []
    return out


def _robot_controller(env):
    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", None)
    if not robots:
        return None
    return getattr(robots[0], "controller", None)


def controller_snapshot(env) -> dict[str, Any]:
    """Capture OSC controller internals that MuJoCo state does not contain."""
    ctrl = _robot_controller(env)
    if ctrl is None:
        return {}
    out: dict[str, Any] = {"controller_type": type(ctrl).__name__}
    for key in ("goal_pos", "goal_ori", "ori_ref", "relative_ori"):
        value = getattr(ctrl, key, None)
        if value is not None:
            out[key] = _arr(value).tolist()
    for key in ("interpolator_pos", "interpolator_ori"):
        interp = getattr(ctrl, key, None)
        if interp is None:
            continue
        out[key] = {
            "start": _arr(interp.start).tolist(),
            "goal": _arr(interp.goal).tolist(),
            "step": int(getattr(interp, "step", 0)),
            "total_steps": int(getattr(interp, "total_steps", 0)),
        }
    return out


def reset_episode_flags(env, timestep: int = 0) -> None:
    """Clear robosuite's episode-level done/timestep flags after restore."""
    inner = getattr(env, "env", env)
    if hasattr(inner, "done"):
        inner.done = False
    if hasattr(inner, "timestep"):
        inner.timestep = int(timestep)


def restore_controller(env, state: dict[str, Any]) -> None:
    ctrl = _robot_controller(env)
    if ctrl is None or not state:
        return
    for key in ("goal_pos", "goal_ori", "ori_ref", "relative_ori"):
        if key in state:
            setattr(ctrl, key, np.asarray(state[key], dtype=np.float64).copy())
    for key in ("interpolator_pos", "interpolator_ori"):
        if key not in state:
            continue
        interp = getattr(ctrl, key, None)
        if interp is None:
            continue
        interp.start = np.asarray(state[key]["start"], dtype=np.float64).copy()
        interp.goal = np.asarray(state[key]["goal"], dtype=np.float64).copy()
        interp.step = int(state[key]["step"])
        if "total_steps" in state[key]:
            interp.total_steps = int(state[key]["total_steps"])


def model_signature(env) -> dict[str, Any]:
    m = env.sim.model
    return {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "nbody": int(m.nbody), "njnt": int(m.njnt), "ngeom": int(m.ngeom),
        "body_names": [str(x) for x in m.body_names],
        "site_names": [str(x) for x in m.site_names],
        "site_bodyid": [int(x) for x in m.site_bodyid],
        "site_pos": _arr(m.site_pos).round(12).tolist(),
        "site_quat": _arr(m.site_quat).round(12).tolist(),
        "site_size": _arr(m.site_size).round(12).tolist(),
        "site_type": [int(x) for x in m.site_type],
        "body_pos": _arr(m.body_pos).round(12).tolist(),
        "body_quat": _arr(m.body_quat).round(12).tolist(),
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
        "opt_timestep": float(m.opt.timestep),
        "opt_gravity": _arr(m.opt.gravity).round(12).tolist(),
        "opt_integrator": int(m.opt.integrator),
        "opt_solver": int(m.opt.solver),
        "opt_iterations": int(m.opt.iterations),
        "opt_tolerance": float(m.opt.tolerance),
        "opt_cone": int(m.opt.cone),
        "opt_jacobian": int(getattr(m.opt, "jacobian", -1)),
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
        checks["body_pos"] = sa.get("body_pos") == sb.get("body_pos")
        checks["body_quat"] = sa.get("body_quat") == sb.get("body_quat")
        for key in ("site_names", "site_bodyid", "site_pos", "site_quat", "site_size", "site_type"):
            checks[key] = sa.get(key) == sb.get(key)
    if strict_geometry:
        checks["ngeom"] = sa.get("ngeom") == sb.get("ngeom")
        for key in ("geom_names", "geom_bodyid", "geom_type", "geom_size", "geom_pos", "geom_quat"):
            checks[key] = sa.get(key) == sb.get(key)
    if strict_physics:
        for key in ("geom_contype", "geom_conaffinity", "geom_condim", "geom_friction",
                    "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
                    "body_mass", "body_inertia", "dof_damping", "dof_frictionloss",
                    "dof_armature", "actuator_gainprm", "actuator_biasprm",
                    "actuator_gear", "actuator_ctrlrange", "opt_timestep", "opt_gravity",
                    "opt_integrator", "opt_solver", "opt_iterations", "opt_tolerance",
                    "opt_cone", "opt_jacobian"):
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
    if snap.get("mj_state") is not None:
        sim.set_state(snap["mj_state"])
    else:
        put(sim.data.qpos, snap["qpos"])
        put(sim.data.qvel, snap["qvel"])
    for key in EXTRA_STATE_ARRAYS + ("act",):
        dst = getattr(sim.data, key, None)
        src = snap.get(key, [])
        if dst is not None and len(src):
            put(dst, src)
    restore_controller(env, snap.get("controller", {}))
    reset_episode_flags(env, 0)
    if getattr(sim.data, "mocap_pos", None) is not None and len(snap.get("mocap_pos", [])):
        put(sim.data.mocap_pos, snap["mocap_pos"])
    if getattr(sim.data, "mocap_quat", None) is not None and len(snap.get("mocap_quat", [])):
        put(sim.data.mocap_quat, snap["mocap_quat"])
    sim.data.time = float(snap["time"])
    sim.forward()
    # mj_forward recomputes warm-start acceleration; restore it afterwards so
    # the next integration step starts from the same solver state.
    if getattr(sim.data, "qacc_warmstart", None) is not None and len(snap.get("qacc_warmstart", [])):
        put(sim.data.qacc_warmstart, snap["qacc_warmstart"])


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


def _controller_compare(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-6) -> dict[str, Any]:
    ca = a.get("controller", {}) or {}
    cb = b.get("controller", {}) or {}
    if not ca and not cb:
        return {"present": False, "pass": False, "error": "controller state missing on both sides"}
    if not ca or not cb:
        return {"present": True, "pass": False, "error": "controller state missing on one side"}
    checks = {}
    ok = True
    type_a = ca.get("controller_type"); type_b = cb.get("controller_type")
    checks["controller_type"] = {"pass": bool(type_a and type_a == type_b), "a": type_a, "b": type_b}
    ok = ok and checks["controller_type"]["pass"]
    for key in ("goal_pos", "goal_ori"):
        x = np.asarray(ca.get(key, []), dtype=np.float64)
        y = np.asarray(cb.get(key, []), dtype=np.float64)
        if x.size == 0 or y.size == 0 or x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            checks[key] = {"present": True, "pass": False, "error": float("inf")}
            ok = False
    for key in ("ori_ref", "relative_ori"):
        x = np.asarray(ca.get(key, []), dtype=np.float64)
        y = np.asarray(cb.get(key, []), dtype=np.float64)
        if x.size == 0 and y.size == 0:
            checks[key] = {"pass": True, "error": 0.0, "not_applicable": True}
            continue
        if x.size == 0 or y.size == 0 or x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            checks[key] = {"present": True, "pass": False, "error": float("inf")}
            ok = False
        else:
            err = float(np.max(np.abs(x - y)))
            checks[key] = {"present": True, "pass": bool(err <= tol), "error": err}
            ok = ok and err <= tol
    for key in ("interpolator_pos", "interpolator_ori"):
        x = ca.get(key); y = cb.get(key)
        if x is None and y is None:
            checks[key] = {"pass": True, "error": 0.0, "not_applicable": True}
            continue
        if x is None or y is None:
            checks[key] = {"present": True, "pass": False, "error": float("inf")}
            ok = False
            continue
        ok_key = True; max_err = 0.0
        for field in ("start", "goal"):
            xa = np.asarray(x.get(field, []), dtype=np.float64)
            ya = np.asarray(y.get(field, []), dtype=np.float64)
            if xa.size == 0 or ya.size == 0 or xa.shape != ya.shape or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
                ok_key = False; max_err = float("inf"); continue
            max_err = max(max_err, float(np.max(np.abs(xa - ya))))
        step_ok = int(x.get("step", -1)) == int(y.get("step", -2))
        total_ok = int(x.get("total_steps", -1)) == int(y.get("total_steps", -2))
        checks[key] = {"pass": bool(ok_key and step_ok and total_ok and max_err <= tol),
                       "error": max_err, "step_ok": step_ok, "total_steps_ok": total_ok}
        ok = ok and checks[key]["pass"]
    return {"present": True, "checks": checks, "pass": bool(ok)}


def compare(a: dict[str, Any], b: dict[str, Any], tol: dict[str, float] | None = None) -> dict[str, Any]:
    tol = {**DEFAULT_TOL, **(tol or {})}
    checks = {}
    for key in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat"):
        checks[key] = _array_check(key if key != "mocap_pos" and key != "mocap_quat" else "mocap",
                                   a.get(key), b.get(key), tol,
                                   optional_empty=key in OPTIONAL_EMPTY_OK)
    for key in EXTRA_STATE_ARRAYS:
        checks[key] = _array_check(key, a.get(key), b.get(key), tol, optional_empty=True)
    if a.get("time") is None or b.get("time") is None:
        checks["time"] = {"present": False, "shape": False, "finite": False,
                          "error": float("inf"), "pass": False}
    else:
        checks["time"] = _array_check("time", [a["time"]], [b["time"]], tol)
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
    if a.get("site_names") is not None or b.get("site_names") is not None:
        if a.get("site_names") is None or b.get("site_names") is None:
            checks["site_names"] = {"present": False, "pass": False}
        else:
            checks["site_names"] = {"present": True, "pass": a.get("site_names") == b.get("site_names")}
        for key in ("site_xpos", "site_xmat", "policy_eef_pos", "policy_eef_quat"):
            xa = a.get(key); xb = b.get(key)
            if xa is None or xb is None:
                checks[key] = {"present": False, "pass": False}
            else:
                checks[key] = _array_check("body_xpos", xa, xb, {"body_xpos": 1e-3})
    names_ok = a.get("body_names") == b.get("body_names")
    cam_names_ok = a.get("camera_names") == b.get("camera_names")
    eef_ok = bool(a.get("eef_name")) and a.get("eef_name") == b.get("eef_name")
    checks["body_names"] = {"pass": bool(names_ok)}
    checks["camera_names"] = {"pass": bool(cam_names_ok)}
    checks["eef_name"] = {"pass": eef_ok}
    checks["controller"] = _controller_compare(a, b)
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks.values())}


def camera_compare(a: dict[str, Any], b: dict[str, Any], expect_equal: bool,
                   tol: float = 1e-4, target_cameras: list[str] | None = None) -> dict[str, Any]:
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


def digest_model(snap: dict[str, Any]) -> str:
    blob = json.dumps(snap.get("model_signature", {}), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def digest_controller(snap: dict[str, Any]) -> str:
    blob = json.dumps(snap.get("controller", {}), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


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
        **{k: np.round(np.asarray(snap.get(k, [])), 10).tolist() for k in EXTRA_STATE_ARRAYS},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
