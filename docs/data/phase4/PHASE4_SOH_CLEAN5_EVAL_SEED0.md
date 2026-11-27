# Clean5 closed-loop seed 0

pairs: 30

| group | n | Frozen | SOH | rescued/harmed | net | p | task-cluster 95% CI pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sensor | 20 | 4/20 | 6/20 | 3/1 | +2 | 0.625 | [-10.0, +30.0] |
| Background | 10 | 4/10 | 4/10 | 1/1 | +0 | 1 | [-30.0, +30.0] |

## Sensor subgroups

| subgroup | n | Frozen | SOH | net | p |
|---|---:|---:|---:|---:|---:|
| fog | 4 | 2/4 | 4/4 | +2 | 0.5 |
| gaussian_blur | 4 | 1/4 | 0/4 | -1 | 1 |
| glass_blur | 4 | 0/4 | 0/4 | +0 | 1 |
| motion_blur | 4 | 0/4 | 1/4 | +1 | 1 |
| zoom_blur | 4 | 1/4 | 1/4 | +0 | 1 |

## Background subgroups

| subgroup | n | Frozen | SOH | net | p |
|---|---:|---:|---:|---:|---:|
| scene_texture | 5 | 3/5 | 2/5 | -1 | 1 |
| table_texture | 5 | 1/5 | 2/5 | +1 | 1 |
