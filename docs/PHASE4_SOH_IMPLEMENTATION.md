# Phase 4 SOH-VLA implementation status

Status: **code fixed after the second audit; Step-1 now uses only the real
7-DOF action and two Flow times, but the pre-registered "first 256 dims > 85%
energy" gate still FAILS (78.7%). Do not start Step-2 or Step-3 as if the
assumption were confirmed.**

## Files

| File | Role |
|---|---|
| `dacsc/smolvla_soh.py` | deep gate + orthogonal projection + streaming fingerprint + basis-mixed low-rank hypernetwork |
| `scripts/phase4_soh_subspace.py` | Step 1: source init0 action-sensitivity estimate |
| `scripts/phase4_soh_train.py` | Step 2: frozen-teacher offline meta-training (feature + identity; Flow loss not faked) |
| `scripts/phase4_soh_closedloop.py` | Step 3 wrapper with independent SOH preflight/worker/report |
| `tests/test_smolvla_soh.py` | module, projection, fingerprint, EMA, orthonormality tests |
| `tests/test_phase4_soh_train.py` | generic perturbation CPU/CUDA generator tests |

Old audited files were not modified.

## Bugs fixed after the first audit

1. `SOHModule.forward` now receives one per-view fingerprint, not the two-view
   list.
2. SigLIP layer output is unpacked with a tuple/dataclass helper; Layer-3 and
   Layer-12 captures are kept **per view**, not overwritten by the second camera.
3. Training perturbation tensors use the generator on the same device; CUDA
   generators no longer touch CPU tensors.
4. Clean identity uses the Clean image's own fingerprint, not the perturbed
   fingerprint.
5. Step-1 differentiates the native
   `embed_prefix -> vlm_with_expert(fill_kv_cache) -> denoise_step` chain with
   an explicit noise tensor. `select_action`'s `@torch.no_grad` is bypassed; the
   fallback is no longer used in this run.
6. Step-1 accumulates **both camera** Layer-3 gradients, not only the last view.
7. Step-1 now samples multiple frames per source init0 trajectory.
8. Closed-loop preflight is independent: reject bit-exact Frozen, accept finite,
   encoder calls 2/4, parameter versions unchanged. It does not assert equality
   with the old Alpha=.75 V1 branch.
9. Worker launcher forwards `--soh`; prepare/report do not load CUDA.
10. SOH report/latency/preflight use SOH-specific names and record checkpoint,
    subspace, and source hashes.

## Step 1 result (source init0, actual native 7-DOF Flow gradient)

```text
tasks: 10
frames per task: 4
Flow times per frame: 0.25 and 0.75
valid camera gradients: 108
mode: native_flow_velocity (all)
real action slice: v_t[:, :10, :7]
k = 256

energy prefix:
  16  = 0.3155
  32  = 0.4063
  64  = 0.5105
  128 = 0.6388
  256 = 0.7866
  384 = 0.8757
  512 = 0.9350
```

Pre-registered gate:

```text
first 256 dims > 85% energy  -> FAIL (78.7%)
```

The 25 zero-padded max_action_dim slots were removed from the gradient, so this
is the cleanest estimate so far. It is still not sparse enough for the original
hard k=256 claim. Changing k after seeing the result would be post-hoc; it is
recorded, not silently applied.

## Step 2 training smoke

A one-task, one-frame, one-step smoke completed on the server:

```text
collected_task 0 frames 1
epoch 0 loss 13.3767 n 1
TRAIN_DONE /tmp/soh_smoke.pt (9.2 MB)
```

This only proves the training graph, spectral checkpoint save and freeze path are
runnable; it is **not** a trained model and was not evaluated.

## Step 2 training status

Implemented: frozen SmolVLA, multiple source Clean frames, four generic
perturbations, per-view feature/identity losses. The single-step Flow velocity
loss `L_act` is still not enabled; the script records that fact rather than
faking it.

## Step 3/4 status

Not run. The closed-loop wrapper exists, but the Step-1 gate failed and Step-2
has not produced a checkpoint.

## Reproduction commands

```bash
python -m pytest -q tests/test_smolvla_soh.py tests/test_phase4_soh_train.py

python scripts/phase4_soh_subspace.py \
  --out runs/phase4_soh/subspace_init0.npz \
  --tasks 10 --frames-per-task 4 --k 256
```

## Acceptance status

| Gate | Status |
|---|---|
| zero-scale exact identity | PASS |
| projection orthogonality | PASS |
| parameter budget <1.5M | PASS |
| dual-view capture | PASS |
| CUDA-generator perturbations | PASS |
| Step-1 actual 7-DOF Flow gradient | PASS (differentiable, 108 camera gradients) |
| Step-1 256-dim >85% energy | **FAIL (78.7%)** |
| spectral W_dom implementation | implemented; k no longer hard-coded |
| Step-2 training smoke | PASS (1 task/1 frame/1 step; not a trained model) |
| Step-2 full training | not run |
| Step-3 paired closed loop | not run |
| Step-4 ablations | not run |

## First-round SOH t=0 diagnostic (1 epoch, not 20)

Because the audit gate is still unresolved, only a **one-epoch** first-round
checkpoint was trained (102 shuffled frames, loss 23.97; 1.216M parameters).
This is not the requested 20-epoch run.

Checkpoint receipt records:

```text
act_weight = 0.5
flow_loss = 7-DOF single-step native Flow velocity MSE
damping_beta = 1.0
```

The 50-case edbac60 t=0 diagnostic (`runs/phase4_soh_diag`) gave:

| group | n | accepted | SOH xyz red% | SOH rot red% | DeepGate xyz red% | DeepGate rot red% |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 10 | 0 | N/A | N/A | N/A | N/A |
| Lighting | 10 | 0 | +0.00 | +0.00 | +0.00 | +0.00 |
| Background | 10 | 9 | +0.21 | +3.96 | +1.98 | -0.73 |
| Sensor | 20 | 20 | -0.86 | -1.38 | +7.05 | -1.54 |

The target was +20% to +40% XYZ reduction on unseen Background/Sensor. The
first round does not approach it: Background is roughly flat, and Sensor is
worse than Frozen. Clean/Lighting are untouched because the gate rejected them.

Decision: **do not start the 100-episode closed loop.** The next structural
question is not a larger k; it is whether a per-patch low-rank channel operator
is expressive enough. A spatial local-mixing / token-gating variant needs to be
tested before more training compute is spent.

## Honest status

| Gate | Status |
|---|---|
| Step-1 actual 7-DOF Flow SVD | PASS (differentiable) |
| Step-1 256-dim >85% energy | FAIL (78.7%) |
| spectral W_dom | implemented |
| 10 unit tests | PASS |
| 1-epoch train smoke | PASS |
| 50-case t=0 SOH diagnostic | **negative / below target** |
| 100-episode closed loop | NOT RUN |

## 20-epoch converged SOH t=0 diagnostic

Training used the LoRA-style asymmetric zero init (`b_basis=0`, `scale=1`),
normalized feature/action losses, `act_weight=1.0`, `id_weight=0.5`, and cosine
LR 1e-3 -> 1e-4. 20 epochs x 102 shuffled frames completed.

Training curve (mean per epoch):

```text
epoch 0 : loss 0.2765, l_feat 0.2351, l_act 0.0356, scale 1.0038
epoch 10: loss 0.1801, l_feat 0.1537, l_act 0.0154, scale 1.0859
epoch 19: loss 0.1574, l_feat 0.1413, l_act 0.0143, scale 1.1097
b_basis norm final: 6.6416
```

50-case t=0 diagnostic (pooled and per-case means are both shown):

| group | n | accepted | SOH xyz pooled% | SOH xyz mean% | SOH rot pooled% | SOH rot mean% | DeepGate xyz pooled% | DeepGate xyz mean% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 10 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Lighting | 10 | 0 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| Background | 10 | 9 | +1.11 | +2.22 | +14.73 | +7.55 | +5.17 | +1.98 |
| Sensor | 20 | 20 | **+26.82** | **+19.51** | +11.67 | +3.50 | +12.05 | +7.05 |

This is the first positive SOH result:

* Clean/Lighting remain zero-drift because the gate rejects them;
* Sensor xyz pooled reduction is +26.8% (above the +20% target);
* Background xyz is still nearly flat (+1.1%), while rotation improves +14.7%;
* DeepGate pooled comparisons are from the same rows and the old edbac60
  per-case numbers are not mixed with the new pooled definition.

Decision: SOH is now a **conditional, sensor-driven candidate**, not a universal
Background fix. The next test should be a limited paired closed-loop on the
Sensor-heavy arm, not a full 100-episode campaign until the Background gap is
understood.

## Sensor-only paired closed loop (20 conditions, 40 episodes)

Ran on the 20 edbac60 bilateral Sensor severity-3 conditions with the 20-epoch
checkpoint. Frozen vs SOH, same initial physical/render state, same RNG prefix,
220 steps, paired pixel cache.

| arm | success | rate |
|---|---:|---:|
| Frozen | 3/20 | 15.0% |
| SOH | 6/20 | 30.0% |

```text
rescued/harmed = 5/2
net = +3
exact two-sided p = 0.453125
```

Subgroups:

| subgroup | n | Frozen | SOH | net |
|---|---:|---:|---:|---:|
| fog | 4 | 2/4 | 2/4 | +0 |
| gaussian_blur | 4 | 1/4 | 1/4 | +0 |
| glass_blur | 4 | 0/4 | 0/4 | +0 |
| motion_blur | 4 | 0/4 | 2/4 | +2 |
| zoom_blur | 4 | 0/4 | 1/4 | +1 |

Gate accepted 3785/3816 replans (99.2% episode-mean). This is a positive but
non-significant exploratory result; it does not meet the pre-set 9-12/20
closed-loop bar. It also does not fix glass_blur.

Decision: keep SOH as an exploratory Sensor candidate; do not launch the full
100-condition campaign yet.
