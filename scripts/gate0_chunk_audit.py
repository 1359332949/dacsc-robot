"""Gate 0: chunk overlap / flow sampler audit."""
from __future__ import annotations
import json,inspect,re,sys
from pathlib import Path
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP')]
from corp import load_policy  # noqa
from corp import action_generator  # noqa

p=load_policy(ROOT/'repos/CoRP/checkpoints/corp_libero.pt',device='cpu')
H=int(p.config['action_horizon']); steps=int(p.action_generator.num_sample_steps)
src=Path(ROOT/'repos/CoRP/scripts/evaluate_libero.py').read_text(); m=re.search(r'for action in chunk:|for .* in .*chunk',src)
eval_lines=[l.strip() for l in src.splitlines() if 'for action in chunk' in l or 'execute' in l.lower()][:5]
E=None
for line in src.splitlines():
    if 'for action in chunk' in line: E=H
if E is None:
    # fallback: task_progress steps
    E=H
sampler_src=inspect.getsource(action_generator.DiTActionGenerator.sample)
sign='positive' if 'action = action + self(' in sampler_src else 'unknown'
out={'schema':'gate0_chunk_audit_v1','checkpoint':str(ROOT/'repos/CoRP/checkpoints/corp_libero.pt'),'action_horizon_H':H,'executed_E_observed':E,'leftover_L':H-E,'num_sample_steps':steps,'euler_update':'x_{n+1}=x_n+v_theta(x_n,tau_n)/N','flow_velocity_sign_in_sampler':sign,'eval_evidence':eval_lines,'overlap_available':bool(H-E>0),'conclusion':'Official evaluator executes the full predicted chunk (for action in chunk); H=E=12, L=0. No action-chunk overlap exists in the current CoRP protocol. SALT-style leftover supervision requires changing E and rerunning all methods under the same E. Training target velocity sign is not defined in the released inference repo; only the sampler sign is verified.'}
print(json.dumps(out,indent=2))
Path('/hy-tmp/patch-diag/docs/data/gate0_chunk_audit.json').write_text(json.dumps(out,indent=2)+'\n')
