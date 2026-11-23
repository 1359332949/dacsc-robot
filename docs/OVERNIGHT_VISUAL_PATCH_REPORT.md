# Overnight visual patch/channel report (in progress)

Status: 2026-11-24 06:37 Asia/Shanghai. Server SSH became unreachable
(connection closed after Clash TUN fake-IP 198.18.1.130); experiments on the
server may still be running, but results after the last successful poll could
not be retrieved.

## Protocol

Frozen CoRP, V2 100-case Dev only. No Adapter, no target-test, no Blind.
All runs use dev_verified environment/sensor/policy seeds, official
sensor_transform once, H=12 E=12 FP32 OSMesa, initial agentview SHA check.
Channel sensitivity uses only source_fit Clean frames from
rgb_oracle_pairs_fresh.npz (20 images, one source task; limitation).

## Phase 0 (done, commit 481dd73)

- consensus pair_dist equal weights scale=ones(7); unit test.
- K=3 same-task Clean ratio<0.80 gate:
  Sensor 6/13 recall, 2/17 shift-success FPR; Background 7/13, 2/17;
  Lighting 1/6, 0/24; Clean FPR 0/10.

## Phase 1 Local Patch Energy (done; raw JSON on server)

Clean gate: eps 0.1 = 10/10, eps 0.2 = 10/10.

| Factor | Frozen | LPE eps .1 | LPE eps .2 |
|---|---:|---:|---:|
| Sensor 30 | 17 | 19 | 18 |
| Background 30 | 17 | 18 | 18 |
| Lighting 30 | 24 | 25 | 25 |

Paired vs Frozen (rescued/harmed/net):
- Sensor .1: 3/1/+2; .2: 2/1/+1
- Background .1: 3/2/+1; .2: 2/1/+1
- Lighting .1: 1/0/+1; .2: 1/0/+1

LPE is a weak positive, not a breakthrough. K=3 displacement/command
recovery was not recorded in this run; limitation.

## Phase 2 T3R spatial patch bias

Implementation: boundary-prototype background weights on agentview only;
manual cross-attention with additive logit bias; parity to original resampler
at beta=0 max abs 2.26e-6. Clean gate beta 1/2/4 all 10/10. Shift run launched
(3 betas x 90 cases); results not retrieved.

## Phase 3 channel suppression

Source-only calibration saved: docs/data/v2_channel_sensitivity_source.json,
20 source_fit Clean frames, synthetic brightness/contrast/blur perturbations;
top R channels [130,164,134,327,...]. No target Shift images/labels used.
Channel Clean runs failed early with CUDA device-side assert (attribute to
CPU index tensor/indexing); code fixed (index to device, clamp) but rerun not
confirmed due server SSH loss. No channel result claim.

## Honest status

- Phase 0 complete.
- Phase 1 complete (weak positive).
- Phase 2 Clean gate complete; Shift pending.
- Phase 3 code/calibration complete; closed-loop result pending.
- No target-test, no Adapter, no z0=0 candidate.
