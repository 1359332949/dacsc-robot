"""Apply the ISC-AP closed-form interface calibration at the cross-attention write."""

from __future__ import annotations

import torch


class InterfaceCalibration:
    def __init__(self, attention, calib: dict, *, num_steps: int = 10, last_step_only: bool = True):
        self.attention = attention
        self.original = attention.attend
        self.mean = torch.tensor(calib["mean"], dtype=torch.float32)
        self.basis = torch.tensor(calib["basis"], dtype=torch.float32)          # [K, D]
        self.transform = torch.tensor(calib["transform"], dtype=torch.float32)  # [K, K]
        self.target_mean = torch.tensor(calib["target_mean"], dtype=torch.float32)
        self.num_steps = max(1, int(num_steps))
        self.last_step_only = last_step_only
        self.call = 0
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        self.call += 1
        if self.last_step_only and (self.call % self.num_steps) != 0:
            return output
        device = output.device
        mean = self.mean.to(device)
        basis = self.basis.to(device)
        transform = self.transform.to(device)
        target_mean = self.target_mean.to(device)
        u = (output - mean) @ basis.T               # [B, T, K]
        u_corrected = (u - target_mean) @ transform.T
        delta = (u_corrected - u) @ basis
        return output + delta

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
