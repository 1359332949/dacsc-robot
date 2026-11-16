"""Camera-parameter mutation for a single LIBERO tabletop environment."""

from __future__ import annotations


def parse_view_name(name: str) -> tuple[int, int, int, int, int]:
    import re
    m = re.search(r"_view_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_initstate_", name)
    if not m:
        raise ValueError(f"not a _view_ variant name: {name}")
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def apply_camera(env, config: tuple[int, int, int, int, int]):
    from libero.libero.envs.problems.libero_tabletop_manipulation import (
        rotate_around_y, rotate_around_z, scale_distance_from_pivot)
    h, v, scale, rot, vv = config
    pos = [0.6586131746834771, 0.0, 1.6103500240372423]
    quat = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
    if int(v) != 0:
        r = rotate_around_y(original_quat=quat, original_pos=pos, degrees=int(v))
        pos, quat = r["new_pos"], r["new_quat"]
    r = rotate_around_z(original_quat=quat, original_pos=pos, degrees=int(h))
    pos, quat = r["new_pos"], r["new_quat"]
    if float(scale) / 100.0 != 1.0:
        r = scale_distance_from_pivot(original_quat=quat, original_pos=pos,
                                      scale_factor=float(scale) / 100.0)
        pos, quat = r["new_pos"], r["new_quat"]
    if int(rot) != 0:
        r = rotate_around_z(original_quat=quat, degrees=int(rot)); quat = r["new_quat"]
    if int(vv) != 0:
        r = rotate_around_y(original_quat=quat, degrees=int(vv)); quat = r["new_quat"]
    names = list(env.sim.model.camera_names); i = names.index("agentview")
    env.sim.model.cam_pos[i] = [round(x, 4) for x in pos]
    env.sim.model.cam_quat[i] = [round(x, 4) for x in quat]
    env.sim.forward()
    return env.sim.model.cam_pos[i].tolist(), env.sim.model.cam_quat[i].tolist()
