"""Shared policy-observation rendering path for E2-A/E2-B.

A physical state is restored into the target env and rendered through the same
observation path used by the policy, instead of low-level sim.render().
"""

from __future__ import annotations
import numpy as np
from .state_pairing import restore
from .libero_noise import apply_agentview_noise


def render_policy_observation(env, snapshot, sensor_noise: int = 0, noise_seed: int = 0):
    """Restore a MuJoCo snapshot and return the env's policy observations."""
    restore(env, snapshot)
    # Mirror ControlEnv.regenerate_obs_from_state after restoration.
    env.env.sim.forward()
    try:
        env.check_success()
    except Exception:
        pass
    for name in ("_post_process", "_update_observables"):
        fn = getattr(env, name, None)
        if fn is None:
            fn = getattr(env.env, name, None)
        if fn is not None:
            try:
                fn(force=True) if name == "_update_observables" else fn()
            except TypeError:
                fn()
    obs = env.env._get_observations()
    if sensor_noise:
        obs, _ = apply_agentview_noise(obs, int(sensor_noise), seed=int(noise_seed))
    return obs
