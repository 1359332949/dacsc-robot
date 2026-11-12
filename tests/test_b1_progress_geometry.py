import numpy as np
import pytest
from dacsc.b1_progress_geometry import p_in, p_on, quat_to_R


def test_on_perfect_contact():
    R = np.eye(3)
    p = p_on(np.array([0, 0, 0.04]), np.array([0.06, 0.06, 0.06]), R,
             np.array([0, 0, 0.0]), np.array([0.2, 0.2, 0.02]), R)
    assert p > 0.95


def test_on_above_and_outside():
    R = np.eye(3)
    high = p_on(np.array([0, 0, 0.5]), np.array([0.06, 0.06, 0.06]), R,
                np.array([0, 0, 0.0]), np.array([0.2, 0.2, 0.02]), R)
    outside = p_on(np.array([0.5, 0, 0.04]), np.array([0.06, 0.06, 0.06]), R,
                   np.array([0, 0, 0.0]), np.array([0.2, 0.2, 0.02]), R)
    assert high < 0.1 and outside < 0.1


def test_in_inside_and_outside():
    R = np.eye(3)
    inside = p_in(np.array([0, 0, 0.1]), np.array([0.05, 0.05, 0.05]), R,
                  np.array([0, 0, 0.05]), np.array([0.2, 0.2, 0.2]), R)
    outside = p_in(np.array([0.5, 0, 0.1]), np.array([0.05, 0.05, 0.05]), R,
                   np.array([0, 0, 0.05]), np.array([0.2, 0.2, 0.2]), R)
    above = p_in(np.array([0, 0, 0.6]), np.array([0.05, 0.05, 0.05]), R,
                 np.array([0, 0, 0.05]), np.array([0.2, 0.2, 0.2]), R)
    assert inside > 0.9 and outside < 0.05 and above < 0.05


def test_in_rotated_container():
    # container rotated 45 deg about z; a point inside in local coords must stay inside
    th = np.pi / 4
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    cont_pos = np.array([0.0, 0.0, 0.05])
    local = np.array([0.03, -0.02, 0.03])
    world = cont_pos + R @ local
    p = p_in(world, np.array([0.05, 0.05, 0.05]), R, cont_pos,
             np.array([0.2, 0.2, 0.2]), R)
    # evaluate with the rotated container frame
    p2 = p_in(world, np.array([0.05, 0.05, 0.05]), R, cont_pos,
              np.array([0.2, 0.2, 0.2]), R)
    assert p > 0.5 and p2 > 0.5


def test_quat_rotation():
    q = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    R = quat_to_R(q)
    assert np.allclose(R @ np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), atol=1e-6)
