"""Step 1: estimate the source action-sensitivity basis at SigLIP Layer 3.

This script does not modify the policy and does not run any adapter.  For each
source-domain Clean observation, it differentiates a scalar random projection of
the frozen action output with respect to the native Layer-3 tokens, then
accumulates the channel-space Gram matrix

    G <- G + g^T g,   g = d<u, a> / dX_3  in R^{1024 x 768}.

The leading eigenvectors of G are the directions in the 768-d channel space to
which the action head is most sensitive.  ``U_geo`` is their orthonormal basis;
``P_dom = I - U_geo U_geo^T`` is the candidate repair space.

The result is an estimate, not a causal guarantee.  When the pinned LeRobot
select_action path is wrapped in inference_mode (as in this checkout), the
script records and uses a Layer-12 feature-projection fallback, not an action
Jacobian; never report that fallback as action sensitivity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase3_smolvla_common import load_actor, seed_torch, setup_case
from dacsc.smolvla_v1_hook import LAYERS


def _capture_layer(actor, layer: int, requires_grad: bool):
    vision = actor._model.model.vlm_with_expert.vlm.model.vision_model
    captured = {}

    def hook(module, args, output):
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output
        if requires_grad:
            x = x.detach().requires_grad_(True)
        captured['x'] = x
        if isinstance(output, tuple):
            return (x, *output[1:])
        return x

    handle = vision.encoder.layers[layer - 1].register_forward_hook(hook)
    return handle, captured


def _random_action_projection(action: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if not action.requires_grad:
        raise RuntimeError('action has no autograd graph')
    u = torch.randn(action.shape, generator=generator, device=action.device, dtype=action.dtype)
    return (action * u).sum()


def action_layer3_gradient(actor, batch, generator: torch.Generator, layer: int = 3, fallback_layer: int = 12):
    """Return g = d<u, a> / dX_layer, shape [1, 1024, 768].

    Primary path differentiates the frozen native action output.  If the
    installed policy wraps selection in an inference context, fall back to a
    random projection of the unadapted Layer-12 feature; the caller records
    which path was used and must not label the fallback as action sensitivity.
    """
    model = actor._model
    handle, captured = _capture_layer(actor, layer, requires_grad=True)
    try:
        model.reset()
        with torch.enable_grad():
            # select_action is decorated with @torch.no_grad; predict_action_chunk
            # is not, so use it with an explicit Flow noise and differentiate the
            # resulting chunk.  This is still the native frozen action path.
            shape = (batch['observation.state'].shape[0], model.config.chunk_size, model.config.max_action_dim)
            noise = torch.randn(shape, generator=generator, device=batch['observation.state'].device, dtype=torch.float32)
            action = model.predict_action_chunk(batch, noise=noise)
            if not action.requires_grad:
                raise RuntimeError('native predict_action_chunk has no autograd graph')
            scalar = _random_action_projection(action, generator)
            grad = torch.autograd.grad(scalar, captured['x'], retain_graph=False)[0]
            return grad.detach(), 'action_projection'
    except RuntimeError:
        # Fall through to the feature-only diagnostic.
        pass
    finally:
        handle.remove()

    # The frozen action path in this checkpoint is under inference_mode, so it
    # cannot provide gradients.  Use a direct, unadapted dual-camera vision
    # forward and a Hutchinson projection of the Layer-12 feature as a
    # diagnostic fallback.  This is explicitly *not* an action Jacobian.
    vision = model.model.vlm_with_expert.vlm.model.vision_model
    handle, captured = _capture_layer(actor, layer, requires_grad=True)
    handle12, captured12 = _capture_layer(actor, fallback_layer, requires_grad=False)
    try:
        images, _ = model.prepare_images(batch)
        with torch.enable_grad():
            for im in images:
                vision(pixel_values=im.to(dtype=vision.dtype), patch_attention_mask=None)
            feat = captured12['x'].float()
            u = torch.randn(feat.shape, generator=generator, device=feat.device, dtype=torch.float32)
            scalar = (feat * u).sum()
            grad = torch.autograd.grad(scalar, captured['x'], retain_graph=False)[0]
            return grad.detach(), 'feature_projection_fallback'
    finally:
        handle.remove()
        handle12.remove()


def main(args):
    seed_torch(args.seed)
    actor = load_actor()
    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    grams = []
    provenance = []
    modes = []
    for task in range(args.tasks):
        suite, spec, obs = setup_case(task, 0)
        try:
            batch = actor._build_observation(obs)
            grad, mode = action_layer3_gradient(actor, batch, gen)
            g = grad[0].float()
            assert g.shape == (1024, 768), g.shape
            if not torch.isfinite(g).all():
                raise ValueError(f'nonfinite Layer-3 gradient for task {task}')
            grams.append(g.T @ g)
            modes.append(mode)
            provenance.append({'task': task, 'task_name': spec.task, 'init_index': 0,
                               'gradient_mode': mode, 'grad_shape': list(g.shape)})
            print(json.dumps({'task': task, 'mode': mode, 'grad_norm': float(g.norm())}), flush=True)
        finally:
            suite.close()

    gram = torch.stack(grams).mean(0)
    eigval, eigvec = torch.linalg.eigh(gram)          # ascending
    order = torch.argsort(eigval, descending=True)
    eigval, eigvec = eigval[order], eigvec[:, order]
    eigval = eigval.clamp_min(0)
    total = eigval.sum().clamp_min(1e-12)
    ratios = (eigval / total).cpu().numpy()
    k = min(args.k, eigvec.shape[1])
    u_geo = eigvec[:, :k].contiguous()
    energy = {str(q): float(ratios[:q].sum()) for q in (16, 32, 64, 128, 256, 384, 512, 768) if q <= len(ratios)}
    result = {
        'schema': 'phase4_soh_subspace_init0_v1',
        'tasks': args.tasks,
        'init_index': 0,
        'seed': args.seed,
        'gradient_modes': modes,
        'k': k,
        'eigenvalues': eigval.cpu().numpy(),
        'u_geo': u_geo.cpu().numpy(),
        'energy_prefix': energy,
        'provenance': provenance,
        'warning': ('U_geo is a source-domain action-sensitivity estimate; it is not a physical guarantee and not a test-set fit.'
                    if all(m == 'action_projection' for m in modes) else
                    'FALLBACK: action Jacobian was not differentiable in this pinned checkpoint path; U_geo is a Layer-12 feature-projection proxy, not action sensitivity.'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    np.savez_compressed(out, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    (out.with_suffix('.json')).write_text(json.dumps({k: v for k, v in result.items() if k != 'eigenvalues'
                                                      and k != 'u_geo' and k != 'provenance'}, indent=2) + '\n')
    print('SUBSPACE_DONE', json.dumps({'out': str(out), 'k': k, 'energy': energy,
                                       'modes': modes}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('runs/phase4_soh/subspace_init0.npz'))
    p.add_argument('--tasks', type=int, default=10)
    p.add_argument('--k', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
