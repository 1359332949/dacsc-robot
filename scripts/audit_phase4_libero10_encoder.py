"""Independent real encoder/LLM/Flow call count check; no rollout or fitting."""
import numpy as np
import torch
import phase4_libero10_transfer as run
from dacsc.smolvla_soh import SmolVLASOHAdapter
from phase3_smolvla_common import load_actor, reference_for, seed_torch, sha


def main():
    torch.set_num_threads(1)
    manifest,soh,gates=run.configure();actor=load_actor();model=actor._model
    core=model.model;expert=core.vlm_with_expert
    layer0=expert.vlm.model.vision_model.encoder.layers[0]
    funcs=dict(run.shift.corruption_functions(),glass_blur=run.paired.glass_blur_exact)
    versions=[p._version for p in model.parameters()];sv=[p._version for p in soh.parameters()]
    rows=[]
    for group in ('Clean','Sensor'):
        case=next(c for c in manifest['cases'] if c['group']==group)
        suite,_,raw=run.setup(case)
        try:
            obs=run.shift.perturb(raw,case,0,funcs);refs=reference_for(run.CALIB,case['task_id']);tau=gates[str(case['task_id'])]['threshold']
            frozen=None
            for mode in ('frozen','forced_reject','forced_accept'):
                counts=dict(encoder=0,prefill=0,flow=0)
                def enc(_module,_args,_output):counts['encoder']+=1
                handle=layer0.register_forward_hook(enc)
                original_forward=expert.forward;original_denoise=core.denoise_step
                def forward(*args,**kwargs):
                    if kwargs.get('fill_kv_cache'):counts['prefill']+=1
                    return original_forward(*args,**kwargs)
                def denoise(*args,**kwargs):counts['flow']+=1;return original_denoise(*args,**kwargs)
                expert.forward=forward;core.denoise_step=denoise
                adapter=None
                try:
                    actor.reset(0);seed_torch(0)
                    if mode!='frozen':
                        adapter=SmolVLASOHAdapter(model,refs,tau,soh,ema_beta=0.,decision_override=mode=='forced_accept')
                        adapter.begin()
                    action=actor.act(obs)
                    if adapter:adapter.end()
                    assert np.isfinite(action).all()
                    expected=dict(encoder=4 if mode=='forced_accept' else 2,prefill=1,flow=10)
                    assert counts==expected,(mode,counts)
                    if mode=='frozen':frozen=action.copy()
                    if mode=='forced_reject':assert np.array_equal(frozen,action)
                    rows.append(dict(group=group,case_id=case['id'],mode=mode,actual_calls=dict(counts),
                        action_l2_vs_frozen=float(np.linalg.norm(action-frozen))))
                finally:
                    if adapter:adapter.close()
                    expert.forward=original_forward;core.denoise_step=original_denoise;handle.remove()
        finally:suite.close()
    assert versions==[p._version for p in model.parameters()] and sv==[p._version for p in soh.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in list(model.parameters())+list(soh.parameters()))
    run.write(run.OUT/'encoder_call_audit.json',dict(status='PASS',rows=rows,weights_frozen_and_versions_unchanged=True,
        measurement='actual hooks on encoder layer0, VLM fill_kv_cache, and native denoise_step',
        soh_sha256=sha(run.SOH),calib_sha256=sha(run.CALIB)))
    print('ACTUAL_ENCODER_AUDIT_PASS',rows,flush=True)


if __name__=='__main__':main()
