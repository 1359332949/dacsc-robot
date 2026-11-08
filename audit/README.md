# Latest validation evidence

This directory keeps only reproducibility receipts used by the current
CoRP-RoboCasa and FFN-analysis pipeline.

| File | Scope |
| --- | --- |
| `channel_upstream_static_audit.json` | Fixed upstream source identities for channel-analysis references |
| `domaindrop_vit_upstream_audit.json` | DomainDrop/DgCD revisions, hashes, preserved formulas and CoRP adaptation boundaries |
| `embodied_ffn_upstream_static_audit.json` | Embodied FFN intervention source mapping |
| `current_validation.json` | Latest local tests, static checks and validation boundaries; source-tree hash distinguishes it from later edits |
| `server_preflight_20261107.json` | Target-server environment, real-video, CUDA/BF16, MuJoCo and manifest-reset preflight |
| `video_backend_equivalence_20261108.json` | Real-video exact-pixel comparison of official `opencv` timestamp reads and official `pyav` cache reads |
| `rollout_chain_diagnostics_20261108.json` | State/camera/action/success-chain and training-initial-state policy rollout diagnostics |
| `ffn_clean_jitter_pairs_20261108.json` | Read-only 12-layer clean/color_jitter FFN statistics for four paired OpenDrawer initial states |
| `ffn_action_intervention_20261108.json` | Held-out action-chunk and closed-loop diagnostics for layer0/layer11 FFN interventions |
| `full_layer11_replacement_20261108.json` | Privileged all-channel layer11 FFN replacement closed-loop control |
| `full_block_replacement_20261108.json` | Privileged complete layer11 block replacement positive control |
| `layer11_branch_decomposition_20261108.json` | Privileged layer11 attention/FFN decomposition action-chunk table and attention-path closed-loop validation |
| `layer11_branch_stability_20261108.json` | Fixed-pool clean/jitter qualification and held-out attention-path closed-loop stability check |
| `layer11_composition_diagnostic_20261108.json` | Privileged attention+FFN, residual-only and full-block composition diagnostics on the low-action-error failures |
| `layer11_residual_drift_20261108.json` | Read-only layer0–11 clean/color_jitter residual drift by view, token group and encoder stage |
| `pretrain/` | RoboCasa dataset coverage and split audit |

These receipts are point-in-time evidence under the stated local or target-server
scope. Privileged clean-reference diagnostics do not establish a deployable
method or method effectiveness.
