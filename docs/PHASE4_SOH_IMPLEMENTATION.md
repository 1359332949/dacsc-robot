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
| Step-2 training | not run |
| Step-3 paired closed loop | not run |
| Step-4 ablations | not run |
