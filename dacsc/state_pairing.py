"""State snapshot/restore and numerical pairing checks for LIBERO-Plus V1.1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).copy()


def snapshot(env) -> dict[str, Any]:
    """Capture full MuJoCo state, body poses, EEF and camera parameters."""
    sim = env.sim
    names = [str(n) for n in sim.model.body_names]
    cam_names = [str(n) for n in sim.model.camera_names]
    eef_idx = names.index("gripper0_eef") if "gripper0_eef" in names else None
    out = {
        "body_names": names,
        "camera_names": cam_names,
        "qpos": _arr(sim.data.qpos).tolist(),
        "qvel": _arr(sim.data.qvel).tolist(),
        "act": _arr(sim.data.act).tolist() if getattr(sim.data, "act", None) is not None else [],
        "time": float(sim.data.time),
        "body_xpos": _arr(sim.data.body_xpos).tolist(),
        "body_xquat": _arr(sim.data.body_xquat).tolist(),
        "cam_pos": _arr(sim.model.cam_pos).tolist(),
        "cam_quat": _arr(sim.model.cam_quat).tolist(),
        "cam_fovy": _arr(sim.model.cam_fovy).tolist(),
        "eef_pos": _arr(sim.data.body_xpos[eef_idx]).tolist() if eef_idx is not None else None,
    }
    mocap_pos = getattr(sim.data, "mocap_pos", None)
    mocap_quat = getattr(sim.data, "mocap_quat", None)
    out["mocap_pos"] = _arr(mocap_pos).tolist() if mocap_pos is not None else []
    out["mocap_quat"] = _arr(mocap_quat).tolist() if mocap_quat is not None else []
    return out


def restore(env, snap: dict[str, Any]) -> None:
    """Restore the numerical state; caller must verify it afterwards."""
    sim = env.sim
    sim.data.qpos[:] = np.asarray(snap["qpos"], dtype=sim.data.qpos.dtype)
    sim.data.qvel[:] = np.asarray(snap["qvel"], dtype=sim.data.qvel.dtype)
    if getattr(sim.data, "act", None) is not None and snap.get("act"):
        sim.data.act[:] = np.asarray(snap["act"], dtype=sim.data.act.dtype)
    if getattr(sim.data, "mocap_pos", None) is not None and snap.get("mocap_pos"):
        sim.data.mocap_pos[:] = np.asarray(snap["mocap_pos"], dtype=sim.data.mocap_pos.dtype)
    if getattr(sim.data, "mocap_quat", None) is not None and snap.get("mocap_quat"):
        sim.data.mocap_quat[:] = np.asarray(snap["mocap_quat"], dtype=sim.data.mocap_quat.dtype)
    sim.data.time = float(snap["time"])
    sim.forward()


def compare(a: dict[str, Any], b: dict[str, Any], tol: dict[str, float] | None = None) -> dict[str, Any]:
    """Return max-abs errors and pass/fail; only comparable fields are checked."""
    tol = tol or {"qpos": 1e-6, "qvel": 1e-6, "act": 1e-6, "time": 1e-9,
                  "body_xpos": 1e-5, "mocap": 1e-6}
    errors: dict[str, float] = {}
    for key in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat"):
        x, y = np.asarray(a.get(key, [])), np.asarray(b.get(key, []))
        if x.size == 0 or y.size == 0:
            continue
        if x.shape != y.shape:
            errors[key] = float("inf")
        else:
            errors[key] = float(np.max(np.abs(x - y)))
    x, y = np.asarray(a["body_xpos"]), np.asarray(b["body_xpos"])
    errors["body_xpos"] = float("inf") if x.shape != y.shape else float(np.max(np.abs(x - y)))
    errors["time"] = abs(float(a["time"]) - float(b["time"]))
    pass_ = {
        "qpos": errors.get("qpos", 0.0) <= tol["qpos"],
        "qvel": errors.get("qvel", 0.0) <= tol["qvel"],
        "act": errors.get("act", 0.0) <= tol["act"],
        "body_xpos": errors["body_xpos"] <= tol["body_xpos"],
        "time": errors["time"] <= tol["time"],
        "mocap": max(errors.get("mocap_pos", 0.0), errors.get("mocap_quat", 0.0)) <= tol["mocap"],
        "body_names": a["body_names"] == b["body_names"],
        "camera_names": a["camera_names"] == b["camera_names"],
    }
    return {"errors": errors, "pass": pass_, "all_pass": all(pass_.values())}


def camera_equal(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-9) -> bool:
    for key in ("cam_pos", "cam_quat", "cam_fovy"):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or float(np.max(np.abs(x - y))) > tol:
            return False
    return True


def digest(snap: dict[str, Any]) -> str:
    payload = {
        "body_names": snap["body_names"],
        "camera_names": snap["camera_names"],
        "qpos": np.round(np.asarray(snap["qpos"]), 10).tolist(),
        "qvel": np.round(np.asarray(snap["qvel"]), 10).tolist(),
        "time": round(float(snap["time"]), 10),
        "body_xpos": np.round(np.asarray(snap["body_xpos"]), 8).tolist(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
