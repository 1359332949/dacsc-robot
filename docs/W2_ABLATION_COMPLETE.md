# W2 complete 8-arm ablation

| Group | v1_hard075 | v1_hard075_rms | w2_proj_q95 | js_shrink_q95 | w2_proj_rms | w2_proj_rms_noclamp | js_shrink_rms | js_shrink_rms_noclamp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean near_xyz | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| clean near_rot | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sensor_noise near_xyz | 11.06 | 13.67 | 7.57 | 8.53 | 11.22 | 11.26 | 13.74 | 13.05 |
| sensor_noise near_rot | 10.02 | 9.18 | 6.46 | 7.97 | 8.47 | 8.19 | 9.30 | 9.45 |
| background near_xyz | 16.19 | 17.74 | 15.98 | 16.79 | 18.09 | 20.00 | 17.98 | 19.65 |
| background near_rot | 14.51 | 14.93 | 14.04 | 15.21 | 15.99 | 17.77 | 16.06 | 17.51 |
| lighting near_xyz | 25.22 | 26.53 | 23.05 | 22.40 | 24.97 | 24.92 | 23.04 | 23.00 |
| lighting near_rot | 18.66 | 20.28 | 17.10 | 19.15 | 20.21 | 20.59 | 20.82 | 21.20 |

## Deltas vs v1_hard075

| Group/metric | v1_hard075_rms | w2_proj_q95 | js_shrink_q95 | w2_proj_rms | w2_proj_rms_noclamp | js_shrink_rms | js_shrink_rms_noclamp |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean near_xyz | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| clean near_rot | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| sensor_noise near_xyz | +2.61 | -3.49 | -2.53 | +0.16 | +0.20 | +2.68 | +2.00 |
| sensor_noise near_rot | -0.84 | -3.56 | -2.05 | -1.55 | -1.83 | -0.72 | -0.57 |
| background near_xyz | +1.55 | -0.22 | +0.59 | +1.90 | +3.80 | +1.79 | +3.45 |
| background near_rot | +0.42 | -0.47 | +0.70 | +1.48 | +3.26 | +1.55 | +3.00 |
| lighting near_xyz | +1.31 | -2.17 | -2.82 | -0.25 | -0.30 | -2.18 | -2.22 |
| lighting near_rot | +1.61 | -1.56 | +0.49 | +1.55 | +1.93 | +2.16 | +2.54 |
