"""External TTA baselines: Pixel-Align, Deep Block-12, true ActMAD-style all-layer SGD."""
from __future__ import annotations
import copy,torch
from dacsc.single_block_probe import align,score

class Calib:
    def __init__(self,path):
        z=__import__('numpy').load(path,allow_pickle=False); self.mu=torch.from_numpy(z['mu_l']).float().cuda(); self.ss=torch.from_numpy(z['ss_l']).float().cuda(); self.tau_b3=torch.from_numpy(z['tau_b3']).float().cuda(); self.tau_b12=torch.from_numpy(z['tau_b12']).float().cuda(); self.pixel_mean=torch.from_numpy(z['pixel_mean']).float().cuda(); self.pixel_std=torch.from_numpy(z['pixel_std']).float().cuda()
    def ref(self,layer): return (self.mu[layer],self.ss[layer],self.tau_b3 if layer==3 else self.tau_b12)

def pixel_align_raw(raw,cal):
    out={'state':raw['state'],'instruction':raw['instruction'],'images':{}}
    for vi,v in enumerate(('agentview','eye_in_hand')):
        x=raw['images'][v].float().cuda()/255.; m=x.mean((1,2),keepdim=True); s=x.std((1,2),unbiased=False,keepdim=True)+1e-6; mu=cal.pixel_mean[vi].view(1,1,1,3); ss=cal.pixel_std[vi].view(1,1,1,3); y=((x-m)/s*ss+mu).clamp(0,1); out['images'][v]=(y*255).round().clamp(0,255).to(torch.uint8)
    return out

class ActMAD:
    def __init__(self,policy,cal,lr=2.5e-4):
        self.policy=policy; self.cal=cal; self.lr=lr; self.backbone=policy.representation_extractor.visual_encoder.backbone; [p.requires_grad_(True) for p in self.backbone.parameters()]; self.initial={k:v.detach().clone() for k,v in self.backbone.state_dict().items()}; self.opt=torch.optim.SGD(self.backbone.parameters(),lr=lr,momentum=.9,weight_decay=5e-4); self.caps={i:None for i in range(12)}; self.buf={i:[] for i in range(12)}; self.hooks=[]
        for i,b in enumerate(self.backbone.blocks):
            def mk(i):
                def h(m,a,x): self.caps[i]=x[:,1:]; return None
                return h
            self.hooks.append(b.register_forward_hook(mk(i)))
    def reset(self):
        self.backbone.load_state_dict(self.initial); self.opt=torch.optim.SGD(self.backbone.parameters(),lr=self.lr,momentum=.9,weight_decay=5e-4); self.buf={i:[] for i in range(12)}
    def update(self,batch):
        with torch.enable_grad():
            self.opt.zero_grad(); self.backbone.zero_grad(set_to_none=True); self.policy.representation_extractor.visual_encoder(torch.cat([batch['images'][v] for v in self.policy.representation_extractor.view_names]))
            losses=[]
            for i in range(12):
                p=self.caps[i]; m=p.mean(1); q=p.square().mean(1); sd=(q-m.square()).clamp_min(1e-6).sqrt()
                if len(self.buf[i])>=8: self.buf[i].pop(0)
                if self.buf[i]:
                    bm=torch.stack([x[0] for x in self.buf[i]]).mean(0); bs=torch.stack([x[1] for x in self.buf[i]]).mean(0)
                else: bm,bs=m.detach(),sd.detach()
                mu,ss,_=self.cal.ref(i); losses.append((((m-mu).square()+(sd-ss).square())/(ss.square()+1e-6)).mean() + (((bm-mu).square()+(bs-ss).square())/(ss.square()+1e-6)).mean())
                self.buf[i].append((m.detach(),sd.detach()))
            loss=torch.stack(losses).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.backbone.parameters(),1.0); self.opt.step(); self.opt.zero_grad(set_to_none=True); return float(loss.detach())
    def close(self):
        for h in self.hooks: h.remove()
