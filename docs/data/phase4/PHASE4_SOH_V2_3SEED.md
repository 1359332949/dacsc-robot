# SOH v2 six-mode paired closed loop: seeds 0, 1, 2

pairs: 90

| group | n | Frozen | SOH | rescued/harmed | net | p | task-cluster 95% CI pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sensor | 60 | 12/60 | 26/60 | 19/5 | +14 | 0.00661075 | [-3.3, +46.7] |
| Background | 30 | 13/30 | 12/30 | 3/4 | -1 | 1 | [-30.0, +16.7] |

## Sensor subgroups

| subgroup | n | Frozen | SOH | net | p |
|---|---:|---:|---:|---:|---:|
| fog | 12 | 7/12 | 9/12 | +2 | 0.625 |
| gaussian_blur | 12 | 2/12 | 4/12 | +2 | 0.6875 |
| glass_blur | 12 | 0/12 | 2/12 | +2 | 0.5 |
| motion_blur | 12 | 2/12 | 5/12 | +3 | 0.4531 |
| zoom_blur | 12 | 1/12 | 6/12 | +5 | 0.0625 |

## Background subgroups

| subgroup | n | Frozen | SOH | net | p |
|---|---:|---:|---:|---:|---:|
| scene_texture | 15 | 9/15 | 6/15 | -3 | 0.375 |
| table_texture | 15 | 4/15 | 6/15 | +2 | 0.5 |
