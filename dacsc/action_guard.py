"""Output-space SO(3) action trust region."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation


@torch.no_grad()
def limit_rotation_delta(
    candidate_normalized: torch.Tensor,
    reference_normalized: torch.Tensor,
    normalizer,
    max_rotation_delta_rad: float,
) -> torch.Tensor:
    """Bound each action's geodesic rotation change relative to a reference.

    Non-rotation action dimensions are unchanged. A zero budget exactly
    preserves the reference rotation. Inputs and outputs use policy-normalized
    action coordinates; clipping is performed in controller coordinates.
    """
    if not np.isfinite(max_rotation_delta_rad) or max_rotation_delta_rad < 0:
        raise ValueError("max_rotation_delta_rad must be finite and non-negative")
    if candidate_normalized.shape != reference_normalized.shape:
        raise ValueError("candidate/reference action shapes must match")
    if candidate_normalized.shape[-1] < 6:
        raise ValueError("action must contain xyz and 3 rotation dimensions")

    candidate_ctrl = normalizer.unnormalize_action(candidate_normalized).float()
    reference_ctrl = normalizer.unnormalize_action(reference_normalized).float()
    shape = candidate_ctrl.shape
    candidate_np = candidate_ctrl[..., 3:6].detach().cpu().numpy().reshape(-1, 3)
    reference_np = reference_ctrl[..., 3:6].detach().cpu().numpy().reshape(-1, 3)

    reference_rot = Rotation.from_rotvec(reference_np)
    candidate_rot = Rotation.from_rotvec(candidate_np)
    relative = reference_rot.inv() * candidate_rot
    relative_rotvec = relative.as_rotvec()
    relative_angle = np.linalg.norm(relative_rotvec, axis=1)

    scale = np.ones_like(relative_angle)
    over = relative_angle > max_rotation_delta_rad
    if max_rotation_delta_rad == 0.0:
        scale[over] = 0.0
    else:
        scale[over] = max_rotation_delta_rad / np.maximum(relative_angle[over], 1e-12)
    bounded_rot = reference_rot * Rotation.from_rotvec(relative_rotvec * scale[:, None])

    out_ctrl = candidate_ctrl.clone()
    bounded_tensor = torch.as_tensor(
        bounded_rot.as_rotvec().reshape(shape[:-1] + (3,)),
        device=out_ctrl.device,
        dtype=out_ctrl.dtype,
    )
    out_ctrl[..., 3:6] = bounded_tensor
    return normalizer.normalize_action(out_ctrl)
