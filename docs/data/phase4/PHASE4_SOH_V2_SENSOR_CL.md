# SOH Sensor-only paired closed loop

20 Sensor conditions, 40 episodes.

| arm | success | rate |
|---|---:|---:|
| Frozen | 3/20 | 15.0% |
| SOH | 9/20 | 45.0% |

rescued/harmed = 8/2, net +6, exact p=0.109375

| subgroup | n | Frozen | SOH | net |
|---|---:|---:|---:|---:|
| fog | 4 | 2/4 | 3/4 | +1 |
| gaussian_blur | 4 | 1/4 | 1/4 | +0 |
| glass_blur | 4 | 0/4 | 1/4 | +1 |
| motion_blur | 4 | 0/4 | 3/4 | +3 |
| zoom_blur | 4 | 0/4 | 1/4 | +1 |

gate accepted replans 3498/3561; mean episode accept rate 98.56%
