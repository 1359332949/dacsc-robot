"""Step 2: offline meta-training of the SOH residual (feature/identity first).

Training data are *only* source-domain Clean init0 frames plus generic
mathematical perturbations implemented locally in this file.  No LIBERO-Plus
code, texture, fog, glass or test background is imported.

The freeze is strict: ``actor._model`` parameters are never updated; only
``SOHModule`` parameters receive gradients.  The current script supports the
feature-alignment and Clean-identity losses.  A single-step Flow-velocity loss
requires an explicit native Flow hook and is intentionally not silently faked;
add it only after the feature path and the paired closed-loop protocol are
audited.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phase3_smolvla_common import load_actor, seed_torch, setup_case, reference_for
from dacsc.smolvla_soh import SOHModule, parameter_count
from dacsc.smolvla_v1_hook import SmolVLAV1Hook
from lerobot.common.policies.smolvla.modeling_smolvla import make_att_2d_masks


def _layer_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state
    return output


def _soh_hooks(vision, soh, fps, capture):
    """Register per-view Layer-3 SOH and Layer-12 capture hooks."""
    view_idx = 0

    def l3(module, args, output):
        nonlocal view_idx
        x = _layer_tensor(output)
        fixed = soh(x, fps[view_idx])
        view_idx += 1
        if isinstance(output, tuple):
            return (fixed, *output[1:])
        if hasattr(output, 'last_hidden_state'):
            output.last_hidden_state = fixed
            return output
        return fixed

    def l12(module, args, output):
        x = _layer_tensor(output)
        capture.setdefault('l12', []).append(x)
        return None

    h3 = vision.encoder.layers[2].register_forward_hook(l3)
    h12 = vision.encoder.layers[11].register_forward_hook(l12)
    return h3, h12


def _capture_vision(vision, images, *, soh=None, fps=None):
    capture = {}
    handles = []
    if soh is not None:
        h3, h12 = _soh_hooks(vision, soh, fps, capture)
        handles.extend([h3, h12])
    else:
        def l3(module, args, output):
            capture.setdefault('l3', []).append(_layer_tensor(output).detach())
            return None

        def l12(module, args, output):
            capture.setdefault('l12', []).append(_layer_tensor(output).detach())
            return None

        handles.append(vision.encoder.layers[2].register_forward_hook(l3))
        handles.append(vision.encoder.layers[11].register_forward_hook(l12))
    try:
        with torch.no_grad() if soh is None else torch.enable_grad():
            outs = [vision(pixel_values=im.to(dtype=vision.dtype), patch_attention_mask=None) for im in images]
        if len(capture.get('l12', [])) != 2:
            raise RuntimeError(f'expected two camera captures, got {len(capture.get("l12", []))}')
        return outs, capture
    finally:
        for h in handles:
            h.remove()


def _gaussian_kernel(size: int, sigma: float, device, dtype):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / k.sum()
    return k[:, None] * k[None, :]


def generic_perturb(images: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply one of four generic math perturbations; no benchmark asset used."""
    x = images.float()
    mode = int(torch.randint(0, 4, (1,), generator=gen, device=x.device).item())
    if mode == 0:
        # Random low-pass Fourier attenuation.
        b, c, h, w = x.shape
        f = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing='ij')
        radius = float(torch.empty(1, device=x.device).uniform_(0.25, 0.85, generator=gen).item())
        mask = (yy.square() + xx.square() <= radius * radius).float()[None, None]
        x = torch.fft.ifft2(torch.fft.ifftshift(f * mask, dim=(-2, -1)), dim=(-2, -1)).real
    elif mode == 1:
        # Random Gaussian blur.
        size = int(torch.randint(3, 15, (1,), generator=gen, device=x.device).item()) | 1
        sigma = float(torch.empty(1, device=x.device).uniform_(0.5, 4.0, generator=gen).item())
        k = _gaussian_kernel(size, sigma, x.device, x.dtype)[None, None]
        x = F.conv2d(x, k.expand(x.shape[1], 1, size, size), padding=size // 2, groups=x.shape[1])
    elif mode == 2:
        # Random brightness/contrast/colour matrix.
        b = x.mean(dim=(-2, -1), keepdim=True)
        gain = float(torch.empty(1, device=x.device).uniform_(0.55, 1.45, generator=gen).item())
        bias = float(torch.empty(1, device=x.device).uniform_(-0.15, 0.15, generator=gen).item())
        x = (x - b) * gain + b + bias
        mat = torch.eye(3, device=x.device, dtype=x.dtype)
        mat = mat + 0.08 * torch.randn(3, 3, generator=gen, device=x.device, dtype=x.dtype)
        x = torch.einsum('ij,bjhw->bihw', mat, x)
    else:
        # Random low-frequency bilinear noise blend.
        b, c, h, w = x.shape
        low = torch.rand(1, c, 8, 8, generator=gen, device=x.device, dtype=x.dtype)
        low = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
        alpha = float(torch.empty(1, device=x.device).uniform_(0.05, 0.35, generator=gen).item())
        x = (1.0 - alpha) * x + alpha * low
    return x.clamp(0.0, 1.0)




def _flow_velocity(actor, batch, *, soh=None, fps=None, noise=None, flow_time: float = 0.5):
    """Native single-step Flow velocity; optionally repair Layer 3 via SOH."""
    policy = actor._model
    flow = policy.model
    vision = flow.vlm_with_expert.vlm.model.vision_model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    handle = None
    if soh is not None:
        handle, h12 = _soh_hooks(vision, soh, fps, {})
    try:
        context = torch.enable_grad() if soh is not None else torch.no_grad()
        with context:
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
            if noise is None:
                shape = (state.shape[0], flow.config.chunk_size, flow.config.max_action_dim)
                noise = torch.randn(shape, device=state.device, dtype=torch.float32)
            t = torch.full((state.shape[0],), float(flow_time), dtype=torch.float32, device=state.device)
            return flow.denoise_step(prefix_pad_masks, past_key_values, noise, t)
    finally:
        if handle is not None:
            handle.remove()
            h12.remove()



def collect_frames(actor, task: int, frames_per_task: int):
    """Run one source init0 episode and sample Clean dual-camera batches."""
    suite, spec, obs = setup_case(task, 0)
    frames = []
    try:
        actor.reset(0)
        seed_torch(0)
        for step in range(220):
            if step % max(1, 220 // frames_per_task) == 0:
                batch = actor._build_observation(obs)
                frames.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                               for k, v in batch.items()})
            action = actor.act(obs)
            result = suite.step(action)
            obs = result.obs
            if result.done or result.success:
                break
    finally:
        suite.close()
    return frames[:frames_per_task]



def main(args):
    seed_torch(args.seed)
    actor = load_actor()
    model = actor._model
    vision = model.model.vlm_with_expert.vlm.model.vision_model
    model.eval().requires_grad_(False)
    z = np.load(args.subspace)
    u_geo = torch.as_tensor(z['u_geo'], device='cuda', dtype=torch.float32)
    if 'v_basis' in z.files and 'eigenvalues' in z.files:
        v_basis = torch.as_tensor(z['v_basis'], device='cuda', dtype=torch.float32)
        eigenvalues = torch.as_tensor(z['eigenvalues'], device='cuda', dtype=torch.float32)
        soh = SOHModule(v_basis=v_basis, eigenvalues=eigenvalues,
                        rank=args.rank, num_bases=args.num_bases).cuda()
    else:
        soh = SOHModule(u_geo, rank=args.rank, num_bases=args.num_bases).cuda()
    opt = torch.optim.AdamW(soh.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    print(json.dumps({'params': parameter_count(soh), 'u_geo': list(u_geo.shape),
                      'lr': args.lr, 'act_weight': args.act_weight, 'id_weight': args.id_weight}), flush=True)

    # Collect several Clean frames from every source init0 trajectory.
    frames = []
    refs_by_task = {}
    for task in range(args.tasks):
        suite, spec, _ = setup_case(task, 0)
        suite.close()
        task_frames = collect_frames(actor, task, args.frames_per_task)
        if not task_frames:
            raise RuntimeError(f'no frames collected for task {task}')
        frames.extend((task, b, spec.task) for b in task_frames)
        refs_by_task[task] = reference_for(args.calib, task)
        print(json.dumps({'collected_task': task, 'frames': len(task_frames)}), flush=True)

    for epoch in range(args.epochs):
        losses = []
        parts = []
        order = list(range(len(frames)))
        random.shuffle(order)
        steps_this_epoch = len(frames) if args.steps <= 0 else args.steps
        for step in range(steps_this_epoch):
            task, batch, _ = frames[order[step % len(order)]]
            gen = torch.Generator(device='cuda').manual_seed(args.seed + 1000 * epoch + step)
            clean_images, _ = model.prepare_images(batch)
            # Teacher: clean Layer-12 features, per camera.
            _, clean = _capture_vision(vision, clean_images)
            teacher = torch.cat([x.float() for x in clean['l12']], dim=0)
            # Student pass 1 on a generic perturbation.  Perturb raw [0,1]
            # camera tensors, then use the native image preparation path.
            pert_batch = dict(batch)
            pert_batch['observation.images.image'] = generic_perturb(batch['observation.images.image'], gen)
            pert_batch['observation.images.image2'] = generic_perturb(batch['observation.images.image2'], gen)
            pert_images, _ = model.prepare_images(pert_batch)
            _, raw = _capture_vision(vision, pert_images)
            refs = refs_by_task[task]
            fps = []
            for v in range(2):
                fps.append(SOHModule.build_raw_fingerprint(
                    raw['l3'][v], raw['l12'][v],
                    (refs[3][0][v], refs[3][1][v]),
                    (refs[12][0][v], refs[12][1][v]),
                ))
            _, fixed = _capture_vision(vision, pert_images, soh=soh, fps=fps)
            pred = torch.cat([x.float() for x in fixed['l12']], dim=0)
            l_feat = F.mse_loss(pred, teacher) / (teacher.var() + 1e-6) + (1.0 - F.cosine_similarity(pred.flatten(1), teacher.flatten(1), dim=1)).mean()
            # Single-step 7-DOF native Flow velocity consistency.
            # Teacher and student share the same noise and random time.
            noise_shape = (batch['observation.state'].shape[0], model.model.config.chunk_size,
                           model.model.config.max_action_dim)
            noise = torch.randn(noise_shape, generator=gen, device='cuda', dtype=torch.float32)
            flow_time = float(torch.empty(1, device='cuda').uniform_(0.1, 0.9, generator=gen).item())
            v_clean = _flow_velocity(actor, batch, noise=noise, flow_time=flow_time)
            v_fixed = _flow_velocity(actor, pert_batch, soh=soh, fps=fps, noise=noise, flow_time=flow_time)
            v_fixed_7 = v_fixed[:, :10, :7].float()
            v_clean_7 = v_clean[:, :10, :7].float().detach()
            l_act = F.mse_loss(v_fixed_7, v_clean_7) / (v_clean_7.var() + 1e-6)
            # Clean identity: 20% of updates use the clean image and *its own*
            # fingerprint; residual must vanish on Clean.
            if (step % 5) == 0:
                clean_fps = []
                for v in range(2):
                    clean_fps.append(SOHModule.build_raw_fingerprint(
                        clean['l3'][v], clean['l12'][v],
                        (refs[3][0][v], refs[3][1][v]),
                        (refs[12][0][v], refs[12][1][v]),
                    ))
                _, clean_fixed = _capture_vision(vision, clean_images, soh=soh, fps=clean_fps)
                l_id = sum(F.mse_loss(a.float(), b.float()) for a, b in zip(clean_fixed['l12'], clean['l12']))
            else:
                l_id = pred.new_zeros(())
            loss = l_feat + args.act_weight * l_act + args.id_weight * l_id
            parts.append((float(l_feat.detach()), float(l_act.detach()), float(l_id.detach())))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(soh.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        if parts:
            arr = np.asarray(parts)
            lr = float(opt.param_groups[0]['lr'])
            print(json.dumps({'epoch': epoch, 'loss': float(np.mean(losses)), 'n': len(losses),
                              'l_feat': float(arr[:, 0].mean()), 'l_act': float(arr[:, 1].mean()),
                              'l_id': float(arr[:, 2].mean()), 'b_basis_norm': float(soh.b_basis.norm()),
                              'scale': float(soh.scale), 'lr': lr}), flush=True)
        scheduler.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    payload = {'state_dict': soh.state_dict(), 'rank': args.rank,
               'num_bases': args.num_bases, 'damping_beta': getattr(soh, 'damping_beta', 1.0),
               'schema': 'phase4_soh_ckpt_v1'}
    if soh.v_basis.numel() > 0:
        payload['v_basis'] = soh.v_basis.cpu()
        if 'eigenvalues' in z.files:
            payload['eigenvalues'] = torch.as_tensor(z['eigenvalues'], dtype=torch.float32)
    else:
        payload['u_geo'] = soh.u_geo.cpu()
    torch.save(payload, out)
    (out.with_suffix('.json')).write_text(json.dumps({'schema': 'phase4_soh_train_receipt_v1',
        'params': parameter_count(soh), 'epochs': args.epochs, 'steps_per_epoch': args.steps,
        'tasks': args.tasks, 'seed': args.seed, 'losses_last_epoch': losses,
        'flow_loss': '7-DOF single-step native Flow velocity MSE',
        'act_weight': args.act_weight, 'id_weight': args.id_weight,
        'damping_beta': getattr(soh, 'damping_beta', 1.0),
        'perturbations': 'generic FFT low-pass, Gaussian blur, colour/brightness, low-frequency noise blend'}, indent=2) + '\n')
    print('TRAIN_DONE', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--subspace', type=Path, default=Path('runs/phase4_soh/subspace_init0.npz'))
    p.add_argument('--calib', type=Path, default=Path('runs/phase3_smolvla/spatial_init0_calib.npz'))
    p.add_argument('--out', type=Path, default=Path('runs/phase4_soh/soh_ckpt.pt'))
    p.add_argument('--tasks', type=int, default=10)
    p.add_argument('--frames-per-task', type=int, default=24)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--num-bases', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--id-weight', type=float, default=0.5)
    p.add_argument('--act-weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
