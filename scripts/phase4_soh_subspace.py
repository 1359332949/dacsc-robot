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




def _capture_views(actor, layer: int = 3):
    vision = actor._model.model.vlm_with_expert.vlm.model.vision_model
    caps = []

    def hook(module, args, output):
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output
        x = x.detach().requires_grad_(True)
        caps.append(x)
        if isinstance(output, tuple):
            return (x, *output[1:])
        return x

    handle = vision.encoder.layers[layer - 1].register_forward_hook(hook)
    return handle, caps


def flow_velocity_gradient(actor, batch, generator: torch.Generator):
    """Differentiate one native Flow velocity with respect to both Layer-3 views.

    This bypasses the policy's @torch.no_grad select_action wrapper: it calls
    the native embed_prefix -> vlm_with_expert(fill_kv_cache) -> denoise_step
    chain directly, with an explicit noise tensor.  Returns [g_view0, g_view1].
    """
    from lerobot.common.policies.smolvla.modeling_smolvla import make_att_2d_masks

    policy = actor._model
    flow = policy.model
    vision = flow.vlm_with_expert.vlm.model.vision_model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    handle, caps = _capture_views(actor, 3)
    try:
        with torch.enable_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state)
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = flow.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=flow.config.use_cache,
                fill_kv_cache=True,
            )
            shape = (state.shape[0], flow.config.chunk_size, flow.config.max_action_dim)
            noise = torch.randn(shape, generator=generator, device=state.device, dtype=torch.float32)
            t = torch.full((state.shape[0],), 1.0, dtype=torch.float32, device=state.device)
            v_t = flow.denoise_step(prefix_pad_masks, past_key_values, noise, t)
            u = torch.randn(v_t.shape, generator=generator, device=v_t.device, dtype=v_t.dtype)
            scalar = (v_t * u).sum()
            grads = torch.autograd.grad(scalar, caps, allow_unused=False)
            if len(grads) != 2:
                raise RuntimeError(f'expected two camera gradients, got {len(grads)}')
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            return [g.detach() for g in grads], 'native_flow_velocity'
    finally:
        handle.remove()


def action_layer3_gradient(actor, batch, generator: torch.Generator, layer: int = 3, fallback_layer: int = 12):
    """Return g = d<u, a> / dX_layer, shape [1, 1024, 768].

    Primary path differentiates the frozen native action output.  If the
    installed policy wraps selection in an inference context, fall back to a
    random projection of the unadapted Layer-12 feature; the caller records
    which path was used and must not label the fallback as action sensitivity.
    """
    model = actor._model
    try:
        grads, mode = flow_velocity_gradient(actor, batch, generator)
        # Symmetric dual-view accumulation: average their channel Gramians.
        return grads, mode
    except Exception as exc:
        print(json.dumps({'fallback_reason': repr(exc)}), flush=True)
    # Fall through to the feature-only diagnostic.

    # The frozen action path in this checkpoint is under inference_mode, so it
    # cannot provide gradients.  Use a direct, unadapted dual-camera vision
    # forward and a Hutchinson projection of the Layer-12 feature as a
    # diagnostic fallback.  This is explicitly *not* an action Jacobian.
    vision = model.model.vlm_with_expert.vlm.model.vision_model
    handle, caps = _capture_views(actor, layer)
    try:
        images, _ = model.prepare_images(batch)
        with torch.enable_grad():
            outs = [vision(pixel_values=im.to(dtype=vision.dtype), patch_attention_mask=None) for im in images]
            # Per-view Layer-12 capture; never overwrite view 0 with view 1.
            feats = []
            for out in outs:
                if isinstance(out, tuple):
                    x = out[0]
                elif hasattr(out, 'last_hidden_state'):
                    x = out.last_hidden_state
                else:
                    x = out
                feats.append(x.float())
            grads = []
            for i, feat in enumerate(feats):
                u = torch.randn(feat.shape, generator=generator, device=feat.device, dtype=torch.float32)
                scalar = (feat * u).sum()
                grads.append(torch.autograd.grad(scalar, caps[i], retain_graph=True)[0].detach())
            return grads, 'feature_projection_fallback'
    finally:
        handle.remove()



def collect_batches(actor, task: int, frames_per_task: int):
    """Collect several Clean batches from one source init0 trajectory."""
    suite, spec, obs = setup_case(task, 0)
    batches = []
    try:
        actor.reset(0)
        seed_torch(0)
        stride = max(1, 220 // max(1, frames_per_task))
        for step in range(220):
            if step % stride == 0 and len(batches) < frames_per_task:
                b = actor._build_observation(obs)
                batches.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                                for k, v in b.items()})
            action = actor.act(obs)
            result = suite.step(action)
            obs = result.obs
            if result.done or result.success:
                break
    finally:
        suite.close()
    return batches



def main(args):
    seed_torch(args.seed)
    actor = load_actor()
    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    grams = []
    provenance = []
    modes = []
    for task in range(args.tasks):
        suite0, spec, _ = setup_case(task, 0)
        suite0.close()
        batches = collect_batches(actor, task, args.frames_per_task)
        if not batches:
            raise RuntimeError(f'no frames for task {task}')
        for frame_idx, batch in enumerate(batches):
            grads, mode = action_layer3_gradient(actor, batch, gen)
            gs = [g[0].float() if g.ndim == 3 else g.float() for g in grads]
            for g in gs:
                assert g.shape == (1024, 768), g.shape
                if not torch.isfinite(g).all():
                    raise ValueError(f'nonfinite Layer-3 gradient for task {task}')
            grams.extend(g.T @ g for g in gs)
            modes.extend([mode] * len(gs))
            provenance.append({'task': task, 'task_name': spec.task, 'init_index': 0,
                               'frame': frame_idx, 'gradient_mode': mode,
                               'grad_shape': [list(g.shape) for g in gs]})
            print(json.dumps({'task': task, 'frame': frame_idx, 'mode': mode,
                              'grad_norm': [float(g.norm()) for g in gs]}), flush=True)

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
        'frames_per_task': args.frames_per_task,
        'init_index': 0,
        'seed': args.seed,
        'gradient_modes': modes,
        'k': k,
        'eigenvalues': eigval.cpu().numpy(),
        'u_geo': u_geo.cpu().numpy(),
        'energy_prefix': energy,
        'provenance': provenance,
        'warning': ('U_geo is a source-domain action-sensitivity estimate; it is not a physical guarantee and not a test-set fit.'
                    if all(m == 'native_flow_velocity' for m in modes) else
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
    p.add_argument('--frames-per-task', type=int, default=8)
    p.add_argument('--k', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
