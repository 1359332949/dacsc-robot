"""External TTA baselines: Pixel-Align, Deep Block-12, true ActMAD-style all-layer SGD."""
from __future__ import annotations
import torch
from dacsc.single_block_probe import align, score

class Calib:
    def __init__(self, path):
        z = __import__('numpy').load(path, allow_pickle=False)
        self.mu = torch.from_numpy(z['mu_l']).float().cuda()
        self.ss = torch.from_numpy(z['ss_l']).float().cuda()
        self.tau_b3 = torch.from_numpy(z['tau_b3']).float().cuda()
        self.tau_b12 = torch.from_numpy(z['tau_b12']).float().cuda()
        self.pixel_mean = torch.from_numpy(z['pixel_mean']).float().cuda()
        self.pixel_std = torch.from_numpy(z['pixel_std']).float().cuda()
    def ref(self, block_idx):
        # 0-indexed: blocks[2] -> block_idx=2; blocks[11] -> block_idx=11
        assert 0 <= block_idx < 12
        return (self.mu[block_idx], self.ss[block_idx],
                self.tau_b3 if block_idx == 2 else self.tau_b12)

def pixel_align_raw(raw, cal):
    out = {'state': raw['state'], 'instruction': raw['instruction'], 'images': {}}
    for vi, v in enumerate(('agentview', 'eye_in_hand')):
        x = raw['images'][v].float().cuda() / 255.0
        m = x.mean((1, 2), keepdim=True)
        s = x.std((1, 2), unbiased=False, keepdim=True) + 1e-6
        mu = cal.pixel_mean[vi].view(1, 1, 1, 3)
        ss = cal.pixel_std[vi].view(1, 1, 1, 3)
        y = ((x - m) / s * ss + mu).clamp(0, 1)
        out['images'][v] = (y * 255).round().clamp(0, 255).to(torch.uint8)
    return out

class ActMAD:
    """All-12-layer feature-statistic matching, recent<=4 frames, one SGD/replan."""
    def __init__(self, policy, cal, lr=2.5e-4, max_frames=4):
        self.policy = policy; self.cal = cal; self.lr = lr; self.max_frames = max_frames
        self.backbone = policy.representation_extractor.visual_encoder.backbone
        for p in self.backbone.parameters(): p.requires_grad_(True)
        self.initial = {k: v.detach().clone() for k, v in self.backbone.state_dict().items()}
        self.reset()
        self.caps = {i: None for i in range(12)}
        self.hooks = []
        for i, b in enumerate(self.backbone.blocks):
            def mk(i):
                def h(m, a, x): self.caps[i] = x[:, 1:]; return None
                return h
            self.hooks.append(b.register_forward_hook(mk(i)))
    def reset(self):
        self.backbone.load_state_dict(self.initial)
        for p in self.backbone.parameters(): p.requires_grad_(True)
        self.opt = torch.optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=.9, weight_decay=5e-4)
        self.frames = []
        self.last_loss = None; self.last_param_l2 = 0.0
    def update(self, batch):
        # keep recent preprocessed frames; all frames are forwarded with grad
        frame = {'images': {v: batch['images'][v].detach() for v in batch['images']},
                 'state': batch['state'].detach() if torch.is_tensor(batch['state']) else batch['state']}
        self.frames.append(frame)
        if len(self.frames) > self.max_frames: self.frames.pop(0)
        imgs = {v: torch.cat([f['images'][v] for f in self.frames], 0) for v in self.frames[0]['images']}
        states = torch.cat([f['state'] for f in self.frames], 0) if torch.is_tensor(self.frames[0]['state']) else self.frames[0]['state']
        with torch.enable_grad():
            self.opt.zero_grad(); self.backbone.zero_grad(set_to_none=True)
            self.policy.representation_extractor.visual_encoder(torch.cat([imgs[v] for v in self.policy.representation_extractor.view_names]))
            n = len(self.frames); losses = []
            for i in range(12):
                p = self.caps[i].reshape(n, 2, 256, 384)
                m = p.mean((0, 2)); q = p.square().mean((0, 2)); sd = (q - m.square()).clamp_min(1e-6).sqrt()
                mu, ss, _ = self.cal.ref(i)
                losses.append((((m - mu).square() + (sd - ss).square()) / (ss.square() + 1e-6)).mean())
            loss = torch.stack(losses).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
            before = {k: v.detach().clone() for k, v in self.backbone.state_dict().items()}
            self.opt.step(); self.opt.zero_grad(set_to_none=True)
            self.last_param_l2 = float(sum((self.backbone.state_dict()[k] - before[k]).square().sum() for k in before).sqrt())
            self.last_loss = float(loss.detach()); return self.last_loss
    def close(self):
        for h in self.hooks: h.remove()
        self.backbone.load_state_dict(self.initial)
        for p in self.backbone.parameters(): p.requires_grad_(False)
