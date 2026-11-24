import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy
from corp.preprocessing import prepare_observation
p=load_policy(ROOT/'repos/CoRP/checkpoints/corp_libero.pt',device='cuda:0'); p.eval(); z=np.load(ROOT/'runs/patch_diag/rgb_oracle_pairs_fresh.npz'); idx=0
imgs={'agentview':torch.tensor(z['clean_agentview'][idx],device='cuda:0',dtype=torch.uint8).unsqueeze(0),'eye_in_hand':torch.tensor(z['clean_eye'][idx],device='cuda:0',dtype=torch.uint8).unsqueeze(0)}; state=torch.tensor(z['state8'][idx],device='cuda:0',dtype=torch.float32)[None]
def tokens(imgs):
 b=prepare_observation({'images':imgs,'state':state,'instruction':['pick up the object']},config=p.config,normalizer=p.normalizer,device='cuda:0'); ext=p.representation_extractor; vis=ext.visual_encoder(torch.cat([b['images'][v] for v in ext.view_names],dim=0)); pat={v:vis[j:j+1] for j,v in enumerate(ext.view_names)}; lang,pool=ext.language_encoder(b['instruction']); vt=ext.cross_attention_resampler(pat,pool); return vt[0].detach().cpu().numpy()
base=tokens(imgs); zero=torch.zeros_like(imgs['agentview'])
a=tokens({'agentview':zero,'eye_in_hand':imgs['eye_in_hand']}); e=tokens({'agentview':imgs['agentview'],'eye_in_hand':torch.zeros_like(imgs['eye_in_hand'])})
def d(x,y,sl): return float(np.mean(np.linalg.norm(x[sl]-y[sl],axis=-1)))
print(json.dumps({'agent_change_first48':d(a,base,slice(0,48)),'agent_change_second48':d(a,base,slice(48,96)),'eye_change_first48':d(e,base,slice(0,48)),'eye_change_second48':d(e,base,slice(48,96))},indent=2))
