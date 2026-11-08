# Latest validation evidence

This directory keeps only reproducibility receipts used by the current
CoRP-RoboCasa and FFN-analysis pipeline.

| File | Scope |
| --- | --- |
| `channel_upstream_static_audit.json` | Fixed upstream source identities for channel-analysis references |
| `embodied_ffn_upstream_static_audit.json` | Embodied FFN intervention source mapping |
| `current_validation.json` | Latest local tests, static checks and validation boundaries; source-tree hash distinguishes it from later edits |
| `server_preflight_20261107.json` | Target-server environment, real-video, CUDA/BF16, MuJoCo and manifest-reset preflight |
| `video_backend_equivalence_20261108.json` | Real-video exact-pixel comparison of official `opencv` timestamp reads and official `pyav` cache reads |
| `rollout_chain_diagnostics_20261108.json` | State/camera/action/success-chain and training-initial-state policy rollout diagnostics |
| `ffn_clean_jitter_pairs_20261108.json` | Read-only 12-layer clean/color_jitter FFN statistics for four paired OpenDrawer initial states |
| `ffn_action_intervention_20261108.json` | Held-out action-chunk and closed-loop diagnostics for layer0/layer11 FFN interventions |
| `pretrain/` | RoboCasa dataset coverage and split audit |

These receipts are point-in-time local evidence. They do not establish CUDA,
BF16, real-video, MuJoCo, closed-loop success, or method effectiveness.
