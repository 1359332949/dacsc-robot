# Phase 4 SOH-VLA implementation status

Status: **code implemented and tensor-tested; Step-1 subspace estimate ran but
fell back to a feature proxy.  Do not start Step-2 training or Step-3 closed-loop
promotion from this fallback alone.**

## Files

| File | Role |
|---|---|
| `dacsc/smolvla_soh.py` | deep gate + orthogonal projection + streaming fingerprint + basis-mixed low-rank hypernetwork |
| `scripts/phase4_soh_subspace.py` | Step 1: source init0 Layer-3 sensitivity estimate |
| `scripts/phase4_soh_train.py` | Step 2: frozen-teacher offline meta-training (feature + identity; Flow loss not faked) |
| `scripts/phase4_soh_closedloop.py` | Step 3 wrapper around the audited Phase-3 closed-loop harness |
| `tests/test_smolvla_soh.py` | pure-tensor contract tests |

Old audited files (`dacsc/smolvla_v1_hook.py`, `dacsc/smolvla_deepgate.py`,
`scripts/phase3_smolvla_deepgate_closedloop.py`) were not modified.

## Module contract

* `x3: [1, 1024, 768]` BF16 from SigLIP Layer 3.
* `u_geo: [768, k]` orthonormal source basis; `P_dom = I - U_geo U_geo^T`.
* `z = E_psi(raw_fp)`, `raw_fp: [1, 3840]`.
* `A_t = sum_m w_m A^(m)`, `B_t = sum_m w_m B^(m)`.
* `Delta X = GELU(X P_dom A_t) * gamma * B_t`; `X_fixed = X + s * Delta X P_dom`.
* `s` is zero-initialized; at `s=0` the module is bit-identical to the frozen
  network (tested).
* Two cameras are serial: one fingerprint and one repair per camera per replan.

## Step 1 result (source init0, 10 tasks)

```text
k = 256
energy prefix: 16=0.5253, 32=0.6247, 64=0.7175,
               128=0.8099, 256=0.8998, 512=0.9737
gradient mode: feature_projection_fallback (10/10)
```

The 256-dim prefix holds ~90% of the estimated Gram energy, which is
consistent with a low-dimensional axis. **However**, the pinned checkout wraps
the differentiable `select_action` path in `@torch.no_grad`; the attempted
`predict_action_chunk` gradient still did not produce a differentiable action
output. Therefore this Step-1 number is a **Layer-12 feature projection proxy,
not an action Jacobian**. It cannot be used as the paper's "action-sensitivity"
claim yet.

## Step 2 training status

Implemented:

* frozen SmolVLA;
* source init0 Clean frames only;
* four generic perturbations (FFT low-pass, Gaussian blur, colour/brightness,
  low-frequency noise blend);
* `L_feat = MSE + cosine` at Layer 12;
* clean identity penalty.

Not enabled:

* single-step Flow velocity loss `L_act`.  It must be added through an explicit
  native Flow hook and audited; the script records this rather than faking it.

## Step 3/4 status

Not run.  The closed-loop wrapper exists, but launching it before the action
Jacobian and the training protocol are valid would violate the stated
acceptance gate.

## Reproduction commands

```bash
# tensor contract tests
python -m pytest -q tests/test_smolvla_soh.py

# Step 1
python scripts/phase4_soh_subspace.py \
  --out runs/phase4_soh/subspace_init0.npz --tasks 10 --k 256

# Step 2 (after Step 1 is fixed)
python scripts/phase4_soh_train.py \
  --subspace runs/phase4_soh/subspace_init0.npz \
  --out runs/phase4_soh/soh_ckpt.pt
```

## Honest acceptance status

| Gate | Status |
|---|---|
| zero-scale exact identity | PASS |
| projection orthogonality | PASS |
| parameter budget < 1.5M | PASS |
| Step-1 action-sensitivity SVD | **BLOCKED / feature proxy only** |
| Step-2 training | not run |
| Step-3 paired closed loop | not run |
| Step-4 ablations | not run |
