# SmolVLA SOH v2 libero_10: frozen-weight transfer with Clean calibration

Spatial SOH weights frozen; libero_10 init0 Clean calibration allowed. Init1 evaluation, seed0. Not zero-shot or independent confirmation.

| Group | Frozen | SOH | Rescue / harm | Net | Exact p | Task-cluster 95% CI pp |
|---|---:|---:|---:|---:|---:|---|
| Clean | 4/10 | 4/10 | 0/0 | +0 | 1 | [0.0, 0.0] |
| Sensor | 0/20 | 1/20 | 1/0 | +1 | 1 | [0.0, 15.0] |
| motion_blur | 0/4 | 0/4 | 0/0 | +0 | 1 | [0.0, 0.0] |
| gaussian_blur | 0/4 | 0/4 | 0/0 | +0 | 1 | [0.0, 0.0] |
| zoom_blur | 0/4 | 0/4 | 0/0 | +0 | 1 | [0.0, 0.0] |
| glass_blur | 0/4 | 1/4 | 1/0 | +1 | 1 | [0.0, 75.0] |
| fog | 0/4 | 0/4 | 0/0 | +0 | 1 | [0.0, 0.0] |

Gate details and hashes are in summary.json; all full trajectories are retained in pairs/.
Single-seed exploratory transfer; no Lighting/Background evaluation in this run; no universal geometry guarantee.
