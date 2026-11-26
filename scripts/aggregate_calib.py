import json,numpy as np,torch
from pathlib import Path
# Corrected threshold equation. Never overwrite a reference used in published runs.
R=Path('/hy-tmp/patch-diag/runs/phase2_calib')
fs=sorted(R.glob('task_*.npz'),key=lambda p:int(p.stem.split('_')[1]))
tasks=[np.load(f,allow_pickle=False) for f in fs]; names=[str(x['task']) for x in tasks]; print('tasks',len(tasks))
mu_l=torch.stack([torch.from_numpy(x['mu_l']).float() for x in tasks]).mean(0)
ss_l=torch.stack([torch.from_numpy(x['ss_l']).float() for x in tasks]).mean(0)
pix_n=sum(int(x['pixel_n']) for x in tasks); pix_sum=sum(torch.from_numpy(x['pixel_sum']).float() for x in tasks); pix_sq=sum(torch.from_numpy(x['pixel_sq']).float() for x in tasks)
mu_pix=pix_sum/pix_n; ss_pix=(pix_sq/pix_n-mu_pix.square()).clamp_min(0).sqrt(); assert float(mu_pix.min())>=0.05 and float(mu_pix.max())<=0.95,(float(mu_pix.min()),float(mu_pix.max())); assert float(ss_pix.min())>=0.05 and float(ss_pix.max())<=0.50,(float(ss_pix.min()),float(ss_pix.max()))
def loto_tau(key):
 taus=[]
 for i in range(len(tasks)):
  others=[j for j in range(len(tasks)) if j!=i]; M=[]; Q=[]
  for j in others: M.append(torch.from_numpy(tasks[j][key+'_m']).float()); Q.append(torch.from_numpy(tasks[j][key+'_q']).float())
  M=torch.cat(M,0); Q=torch.cat(Q,0); mu=M.mean(0); ss=(Q.mean(0)-mu.square()).clamp_min(1e-3).sqrt()
  m=torch.from_numpy(tasks[i][key+'_m']).float(); q=torch.from_numpy(tasks[i][key+'_q']).float(); s=((m-mu).square()+((q-m.square()).clamp_min(0).sqrt()-ss).square())/(ss.square()+1e-6); taus.append(s)
 return torch.cat(taus).quantile(.95,dim=0)
tau_b3=loto_tau('b3'); tau_b12=loto_tau('b12')
out=R/'spatial_shared_calib_corrected.npz'
if out.exists(): raise FileExistsError('Preserve completed calibration artifacts: '+str(out))
np.savez_compressed(out,tasks=np.array(names,dtype=object),pixel_mean=mu_pix.numpy(),pixel_std=ss_pix.numpy(),mu_l=mu_l.numpy(),ss_l=ss_l.numpy(),tau_b3=tau_b3.numpy(),tau_b12=tau_b12.numpy())
print(json.dumps({'n_tasks':len(tasks),'pixel_n':pix_n,'tau_b3_shape':list(tau_b3.shape),'tau_b12_shape':list(tau_b12.shape)},indent=2))
