# Overnight visual patch/channel report (morning)

Date: 2026-11-24 08:00 Asia/Shanghai. Frozen CoRP, V2 100-case Dev only.
No Adapter/LoRA/MLP/risk classifier, no CoRP weight change, no Blind/target-test.
Runs use dev_verified environment/sensor/policy seeds; official sensor_transform
once; H=12 E=12 FP32 OSMesa; initial agentview SHA check.

## Headline

Spatial patch suppression (T3R-style, agentview only) is the first zero-training
intervention with positive paired net gain on **all three** hard factors while
keeping Clean 10/10. LPE is weakly positive. Channel suppression code and
source-only calibration are complete, but its closed-loop result is NOT
confirmed because of a CUDA device-side assert and a later SSH/network outage.

## Frozen reference (dev_verified)

Clean 10/10; Sensor 17/30; Background 17/30; Lighting 24/30; total 58/90.

## Phase 1: Local Patch Energy (complete)

Clean gate: eps 0.1 = 10/10, eps 0.2 = 10/10.

| Factor | Frozen | LPE .1 | LPE .2 |
|---|---:|---:|---:|
| Sensor 30 | 17 | 19 | 18 |
| Background 30 | 17 | 18 | 18 |
| Lighting 30 | 24 | 25 | 25 |

Paired vs Frozen (rescued/harmed/net):
- Sensor .1: 3/1/+2; .2: 2/1/+1
- Background .1: 3/2/+1; .2: 2/1/+1
- Lighting .1: 1/0/+1; .2: 1/0/+1

Weak positive only. K=3 displacement/command recovery was not logged in these
runs (limitation).

## Phase 2: T3R spatial patch suppression (complete closed loop)

Boundary-prototype background weights on agentview; eye-in-hand unchanged;
manual cross-attention with additive negative logit bias. Parity to original
resampler at beta=0: max abs 2.26e-6.

Clean gate: beta 1.0 = 10/10, beta 2.0 = 10/10, beta 4.0 = 10/10.

| Factor | Frozen | beta 1 | beta 2 | beta 4 |
|---|---:|---:|---:|---:|
| Sensor 30 | 17 | 17 | 18 | 19 |
| Background 30 | 17 | 15 | 19 | 18 |
| Lighting 30 | 24 | 24 | 25 | 26 |
| Total /90 | 58 | 56 | 62 | 63 |

Paired vs Frozen (rescued/harmed/net):
- beta1: Sensor 2/2/0; Background 1/3/-2; Lighting 1/1/0; total -2
- beta2: Sensor 3/2/+1; Background 2/0/+2; Lighting 1/0/+1; total +4
- beta4: Sensor 4/2/+2; Background 2/1/+1; Lighting 3/1/+2; total +5

beta4 is the best zero-training intervention so far. Single seed, 10 tasks,
development reuse; not a confirmatory claim.

## Phase 3: source-only channel sensitivity (calibration complete, closed loop unconfirmed)

Calibration used only 20 source_fit Clean agentview frames from
rgb_oracle_pairs_fresh.npz and synthetic brightness/contrast/blur perturbations.
No Dev Shift images or success labels were read. Output:
docs/data/v2_channel_sensitivity_source.json. Top R channels include
130, 164, 134, 327, 382, 254, 274, 242, 316, 138.

Channel bug found and fixed twice: index tensor moved to CUDA, then channel
indexing corrected to the last dimension (`patches[:,:,idx]`, not token dim).
Clean gates now: p10/g0.3 10/10, p10/g0.5 9/10, p20/g0.3 10/10, p20/g0.5
10/10. Closed-loop Shift runs for p10/g0.3, p20/g0.3, p20/g0.5 were launched
but had only 3-13/90 episodes when this report was written; **no channel shift
result is claimed yet.**

## Server/network blocker

The server host resolved through Clash TUN fake-IP 198.18.1.130. SSH handshake
succeeded but shell commands/scp repeatedly hung or were closed by remote.
Spatial/LPE results above were retrieved before/around the outage; later channel
runs and raw JSON retrieval were not possible. The server may still be running;
the 30-minute scheduled task will keep trying.

## Honest conclusion

1. LPE: Clean-safe, weak positive (+1 to +2).
2. Spatial beta4: Clean-safe, +5 total, positive on Sensor/Background/Lighting;
   best current candidate.
3. Channel: calibration complete; closed-loop unconfirmed; do not cite.
4. No Blind, no target-test, no Adapter, no z0=0 candidate.
5. These are single-seed development results, not statistical significance.

## Addendum: multi-prototype + ReLU spatial (2026-11-24 11:47)

Both beta4 and beta6 passed Clean 10/10. Shift was run with 4 shards each,
merged and deduplicated: 90 unique cases per beta, 30/factor, no duplicates,
all scripts asserted initial agentview SHA.

| Multi/ReLU | Sensor | Background | Lighting | Total | Net |
|---|---:|---:|---:|---:|---:|
| Frozen | 17 | 17 | 24 | 58 | — |
| beta4 | 16 | 20 | 22 | 58 | 0 |
| beta6 | 18 | 14 | 21 | 53 | -5 |

Paired beta4: Sensor -1, Background +3, Lighting -2.
Paired beta6: Sensor +1, Background -3, Lighting -3.

Conclusion: multi-prototype max-cosine + ReLU does NOT improve over the
single-prototype sigmoid beta4 (+5). It trades Background gain for Sensor and
Lighting losses. The original single-prototype beta4 remains the best result;
this negative ablation is retained and must be disclosed.

## Addendum 2: 10-task source calibration, channel mean shrink, Space×Channel

10-task source calibration: 50 source_fit Clean frames (10 tasks × 5 states)
rendered from protocols/static_visual_v2/p0/dev_manifest.json; perturbations
brightness/contrast/gaussian/motion/texture; s_dom relative to clean; channel
means saved to docs/data/v2_channel_sensitivity_source_v2.json. No Dev Shift
images or labels used.

Clean gates with new calibration:
- channel mean shrink p10/g0.3: 9/10 (just at gate)
- Space×Channel (beta4 + p10/g0.3): 10/10

90-case shift (12 shards, merge: 90 unique, 0 dup):

| Method | Sensor | Background | Lighting | Total | Net |
|---|---:|---:|---:|---:|---:|
| Frozen | 17 | 17 | 24 | 58 | — |
| single spatial beta4 | 19 | 18 | 26 | 63 | +5 |
| 10-task channel mean p10/g0.3 | 16 | 22 | 20 | 58 | 0 |
| Space×Channel beta4+p10/g0.3 | 18 | 17 | 24 | 59 | +1 |

Paired net: channel Sensor -1, Background +5, Lighting -4.
Space×Channel Sensor +1, Background 0, Lighting 0.

Interpretation: the 10-task channel mean shrink strongly helps Background
(+5) but harms Lighting (-4) and does not fix Sensor; combined with spatial
beta4 cancels the gains (net +1). Single-prototype spatial beta4 remains the
best overall zero-training method (+5). No target-test, no Adapter.
