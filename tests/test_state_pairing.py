import numpy as np
from dacsc.state_pairing import camera_equal, compare, digest


def make_snap(delta=0.0):
    return {
        "body_names": ["world", "table", "robot0_eef"],
        "camera_names": ["agentview", "eye_in_hand"],
        "qpos": [0.0, 1.0, 2.0],
        "qvel": [0.0, 0.0, 0.0],
        "act": [],
        "time": 1.0,
        "body_xpos": [[0, 0, 0], [1, 0, 0], [0.5, 0.2 + delta, 0.3]],
        "body_xquat": [[1, 0, 0, 0]] * 3,
        "cam_pos": [[1, 0, 1], [0, 1, 1]],
        "cam_quat": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "cam_fovy": [45.0, 45.0],
        "eef_pos": [0.5, 0.2 + delta, 0.3],
        "mocap_pos": [], "mocap_quat": [],
    }


def test_compare_exact():
    a = make_snap(); b = make_snap()
    out = compare(a, b)
    assert out["all_pass"] is True


def test_compare_detects_body_offset():
    out = compare(make_snap(), make_snap(1e-3))
    assert out["pass"]["body_xpos"] is False
    assert out["all_pass"] is False


def test_digest_is_deterministic():
    a = make_snap()
    assert digest(a) == digest(a)
    b = make_snap(); b["qpos"][0] = 0.1
    assert digest(a) != digest(b)


def test_camera_equal():
    a = make_snap(); b = make_snap()
    assert camera_equal(a, b) is True
    b["cam_pos"][0][0] += 0.1
    assert camera_equal(a, b) is False
