import sys,json,torch
from pathlib import Path
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy
from target_only_action_validation import sample_z
from dacsc.differentiable_policy import sample_from_z0
from corp.preprocessing import prepare_observation
p=load_policy(ROOT/'repos/CoRP/checkpoints/corp_libero.pt',device='cuda:0'); p.eval()
# random normalized observation tensors; compare sampler paths only
B=1; obs={'images':{'agentview':torch.randint(0,255,(B,224,224,3),device='cuda:0',dtype=torch.uint8),'eye_in_hand':torch.randint(0,255,(B,224,224,3),device='cuda:0',dtype=torch.uint8)},'state':torch.zeros(B,8,device='cuda:0'),'instruction':['pick up the object']}
batch=prepare_observation(obs,config=p.config,normalizer=p.normalizer,device='cuda:0'); rep=p.representation_extractor(**batch); g=torch.Generator(device='cuda:0'); g.manual_seed(7); z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g)
a=sample_z(p,rep,z0); b=sample_from_z0(p,rep,z0); print(json.dumps({'max_abs_norm':float((a-b).abs().max()),'allclose':bool(torch.allclose(a,b,atol=1e-6))}))
