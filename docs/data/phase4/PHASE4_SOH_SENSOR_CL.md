# SOH Sensor-only paired closed loop

20 Sensor conditions, 40 episodes.

| arm | success | rate |
|---|---:|---:|
| Frozen | 3/20 | 15.0% |
| SOH | 6/20 | 30.0% |

rescued/harmed = 5/2, net +3, exact p=0.453125

| subgroup | n | Frozen | SOH | net |
|---|---:|---:|---:|---:|
| fog | 4 | 2/4 | 2/4 | +0 |
| gaussian_blur | 4 | 1/4 | 1/4 | +0 |
| glass_blur | 4 | 0/4 | 0/4 | +0 |
| motion_blur | 4 | 0/4 | 2/4 | +2 |
| zoom_blur | 4 | 0/4 | 1/4 | +1 |

gate accepted replans 3785/3816; mean episode accept rate 99.22%
