import unittest

import numpy as np

from dacsc.state_pairing import camera_compare, compare, digest, model_compatible


def make_snap(delta=0.0, act=None, eef_name="gripper0_eef"):
    return {
        "body_names": ["world", "table", "gripper0_eef"],
        "camera_names": ["agentview", "eye_in_hand"],
        "eef_name": eef_name,
        "qpos": [0.0, 1.0, 2.0], "qvel": [0.0, 0.0, 0.0],
        "act": [] if act is None else act, "time": 1.0,
        "body_xpos": [[0, 0, 0], [1, 0, 0], [0.5, 0.2 + delta, 0.3]],
        "body_xquat": [[1, 0, 0, 0]] * 3,
        "eef_pos": [0.5, 0.2 + delta, 0.3],
        "cam_pos": [[1, 0, 1], [0, 1, 1]], "cam_quat": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "cam_fovy": [45.0, 45.0],
        "cam_xpos": [[1, 0, 1], [0, 1, 1]], "cam_xmat": [[1, 0, 0]] * 2,
        "mocap_pos": [], "mocap_quat": [],
        "model_signature": {"nq": 3},
    }


class TestStatePairing(unittest.TestCase):
    def test_exact_match_passes(self):
        self.assertTrue(compare(make_snap(), make_snap())["all_pass"])

    def test_body_offset_fails(self):
        out = compare(make_snap(), make_snap(1e-3))
        self.assertFalse(out["all_pass"])
        self.assertFalse(out["checks"]["body_xpos"]["pass"])

    def test_missing_qpos_fails(self):
        b = make_snap(); b.pop("qpos")
        self.assertFalse(compare(make_snap(), b)["all_pass"])

    def test_one_side_empty_act_fails(self):
        a = make_snap(act=[0.1, 0.2]); b = make_snap(act=[])
        self.assertFalse(compare(a, b)["all_pass"])

    def test_both_empty_act_allowed(self):
        self.assertTrue(compare(make_snap(act=[]), make_snap(act=[]))["all_pass"])

    def test_body_quaternion_mismatch_fails(self):
        b = make_snap(); b["body_xquat"] = [[0.9, 0.1, 0, 0]] * 3
        self.assertFalse(compare(make_snap(), b)["all_pass"])

    def test_eef_missing_fails(self):
        b = make_snap(); b["eef_pos"] = None; b["eef_name"] = None
        self.assertFalse(compare(make_snap(), b)["all_pass"])

    def test_digest_deterministic(self):
        self.assertEqual(digest(make_snap()), digest(make_snap()))
        self.assertNotEqual(digest(make_snap()), digest(make_snap(0.1)))

    def test_camera_equal_and_expected_change(self):
        a = make_snap(); b = make_snap()
        self.assertTrue(camera_compare(a, b, expect_equal=True)["pass"])
        c = make_snap(); c["cam_pos"] = [[2, 0, 1], [0, 1, 1]]
        self.assertFalse(camera_compare(a, c, expect_equal=True)["pass"])
        self.assertTrue(camera_compare(a, c, expect_equal=False,
                                       target_cameras=["agentview"])["pass"])

    def test_camera_nan_fails(self):
        b = make_snap(); b["cam_pos"] = [[float("nan"), 0, 1], [0, 1, 1]]
        self.assertFalse(camera_compare(make_snap(), b, expect_equal=True)["pass"])
        self.assertFalse(camera_compare(make_snap(), b, expect_equal=False)["pass"])

    def test_camera_shape_mismatch_fails(self):
        b = make_snap(); b["cam_pos"] = [[1, 0, 1]]
        self.assertFalse(camera_compare(make_snap(), b, expect_equal=False)["pass"])

    def test_quaternion_sign_equivalence_passes(self):
        b = make_snap(); b["body_xquat"] = [[-1, 0, 0, 0]] * 3
        self.assertTrue(compare(make_snap(), b)["all_pass"])

    def test_camera_target_subset_only(self):
        # Only agentview changes: valid Camera shift when agentview is the target.
        b = make_snap(); b["cam_pos"] = [[2, 0, 1], [0, 1, 1]]
        self.assertTrue(camera_compare(make_snap(), b, expect_equal=False,
                                       target_cameras=["agentview"])["pass"])
        # If only eye_in_hand changes but agentview is target, fail.
        c = make_snap(); c["cam_pos"] = [[1, 0, 1], [0, 2, 1]]
        self.assertFalse(camera_compare(make_snap(), c, expect_equal=False,
                                        target_cameras=["agentview"])["pass"])

    def test_strict_geometry_and_physics(self):
        sig = {"nq": 3, "nv": 3, "nu": 2, "njnt": 3, "nbody": 3,
               "body_names": ["w", "table", "eef"],
               "joint_names": ["j1", "j2", "j3"], "joint_types": [0, 0, 0],
               "jnt_qposadr": [0, 1, 2], "jnt_dofadr": [0, 1, 2],
               "actuator_names": ["a1", "a2"],
               "ngeom": 2, "geom_names": ["g1", "g2"], "geom_bodyid": [1, 2],
               "geom_type": [0, 0], "geom_size": [[1, 1, 1], [1, 1, 1]],
               "geom_pos": [[0, 0, 0], [0, 0, 0]], "geom_quat": [[1, 0, 0, 0], [1, 0, 0, 0]],
               "geom_contype": [1, 1], "geom_conaffinity": [1, 1], "geom_condim": [3, 3],
               "geom_friction": [[1, 1, 1], [1, 1, 1]], "geom_solref": [[1, 1], [1, 1]],
               "geom_solimp": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
               "geom_margin": [0, 0], "geom_gap": [0, 0],
               "body_mass": [1, 1, 1], "body_inertia": [[1, 1, 1]] * 3,
               "dof_damping": [1, 1, 1], "dof_frictionloss": [0, 0, 0],
               "dof_armature": [0, 0, 0],
               "actuator_gainprm": [[1, 0, 0], [1, 0, 0]],
               "actuator_biasprm": [[0, -1, 0], [0, -1, 0]],
               "actuator_gear": [[1, 0, 0, 0, 0, 0]] * 2,
               "actuator_ctrlrange": [[-1, 1], [-1, 1]]}
        a = make_snap(); a["model_signature"] = sig
        b = make_snap(); b["model_signature"] = {k: (v[:] if isinstance(v, list) else v) for k, v in sig.items()}
        self.assertTrue(model_compatible(a, b, strict_geometry=True, strict_physics=True)["pass"])
        b["model_signature"]["geom_size"] = [[2, 1, 1], [1, 1, 1]]
        self.assertFalse(model_compatible(a, b, strict_geometry=True)["pass"])
        b["model_signature"]["geom_size"] = sig["geom_size"]
        b["model_signature"]["body_mass"] = [2, 1, 1]
        self.assertFalse(model_compatible(a, b, strict_geometry=True, strict_physics=True)["pass"])

    def test_model_compatibility(self):
        a = make_snap(); b = make_snap()
        a["model_signature"] = {"nq": 3, "nv": 3, "nu": 2, "njnt": 3,
                                "body_names": ["w", "table", "eef"],
                                "joint_names": ["j1", "j2", "j3"],
                                "joint_types": [0, 0, 0],
                                "jnt_qposadr": [0, 1, 2], "jnt_dofadr": [0, 1, 2],
                                "actuator_names": ["a1", "a2"]}
        b["model_signature"] = dict(a["model_signature"])
        self.assertTrue(model_compatible(a, b)["pass"])
        b["model_signature"]["nq"] = 4
        self.assertFalse(model_compatible(a, b)["pass"])


if __name__ == "__main__":
    unittest.main()
