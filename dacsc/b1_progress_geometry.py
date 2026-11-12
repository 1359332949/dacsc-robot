"""Pure geometry for On/In task-progress metrics (no success labels)."""

from __future__ import annotations
import numpy as np


def quat_to_R(q):
    """q = (x,y,z,w) as robosuite body_xquat; return 3x3 rotation."""
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def half_extents_world(size, R):
    """World-axis half extents of an oriented box."""
    h = np.asarray(size, dtype=float) / 2.0
    return np.abs(R) @ h


def box_inside_approx(center_local, size, margin=0.0):
    """Smooth inside-box score in [0,1] (1 = centre fully inside, 0 = outside by margin)."""
    size = np.maximum(np.asarray(size, dtype=float), 1e-6)
    half = size / 2.0
    d = np.abs(np.asarray(center_local, dtype=float)) - half
    return float(np.prod(np.clip(1.0 - np.maximum(d, 0.0) / (half + 1e-9), 0.0, 1.0)))


def p_on(obj_pos, obj_size, obj_R, support_pos, support_size, support_R,
         contact_tol=0.03, margin=0.02):
    """Object bottom vs support top + xy footprint containment."""
    obj_half = half_extents_world(obj_size, obj_R)
    sup_half = half_extents_world(support_size, support_R)
    obj_bottom = obj_pos[2] - obj_half[2]
    sup_top = support_pos[2] + sup_half[2]
    contact = float(np.clip(1.0 - abs(obj_bottom - sup_top) / contact_tol, 0.0, 1.0))
    local = obj_R.T @ (obj_pos - support_pos)
    inside = box_inside_approx(local[:2], support_size[:2] + 2 * margin, margin)
    return contact * inside


def p_in(obj_pos, obj_size, obj_R, cont_pos, cont_size, cont_R,
         margin=0.02, rim_tol=0.05):
    """Object centre inside the container's oriented interior box (below rim)."""
    local = cont_R.T @ (obj_pos - cont_pos)
    inside = box_inside_approx(local, cont_size, margin)
    cont_half = half_extents_world(cont_size, cont_R)
    cont_top = cont_pos[2] + cont_half[2]
    below_rim = float(np.clip((cont_top + rim_tol - obj_pos[2]) / rim_tol, 0.0, 1.0))
    return inside * below_rim
