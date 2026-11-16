"""Re-apply LIBERO-Plus Sensor Noise to an observation dict.

Upstream ControlEnv applies noise only inside step()/reset(); its
regenerate_obs_from_state() path (used by set_init_state) returns unnoised
observations.  This helper lets a restored state be noised without stepping
the physics.
"""

from __future__ import annotations
import numpy as np


def apply_agentview_noise(obs: dict, noise: int, seed: int | None = None) -> tuple[dict, dict]:
    from PIL import Image
    from libero.libero.envs.env_wrapper import (
        motion_blur, gaussian_blur, zoom_blur, fog, glass_blur)
    if not isinstance(obs, dict) or "agentview_image" not in obs:
        raise TypeError("obs must be a dict containing agentview_image")
    if not (0 <= int(noise) <= 50):
        raise ValueError(f"noise must be in 0..50, got {noise}")
    if int(noise) == 0:
        return dict(obs), {"noise": 0, "seed": seed, "mechanism": "none",
                           "affected_view": "none"}
    saved_state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(int(seed))
        out = dict(obs)
        img = np.asarray(obs["agentview_image"])
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        pil = Image.fromarray(img)
        if noise <= 10:
            blurred = motion_blur(pil, severity=int(noise))
        elif noise <= 20:
            blurred = gaussian_blur(pil, severity=int(noise) - 10).astype(np.uint8)
        elif noise <= 30:
            blurred = zoom_blur(pil, severity=int(noise) - 20).astype(np.uint8)
        elif noise <= 40:
            blurred = fog(pil, severity=int(noise) - 30).astype(np.uint8)
        else:
            blurred = glass_blur(pil, severity=int(noise) - 40).astype(np.uint8)
        out["agentview_image"] = np.asarray(blurred, dtype=np.uint8)
        meta = {"noise": int(noise), "seed": seed,
                "mechanism": ("motion_blur" if noise <= 10 else
                              "gaussian_blur" if noise <= 20 else
                              "zoom_blur" if noise <= 30 else
                              "fog" if noise <= 40 else "glass_blur"),
                "affected_view": "agentview_image only"}
        return out, meta
    finally:
        np.random.set_state(saved_state)
