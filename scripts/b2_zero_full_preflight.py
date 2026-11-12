"""Full 24-episode zero-update pairing audit at replans 8/12 (two-env replay)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation, is_success
from corp import load_policy
from corp.preprocessing import prepare_observation
from dacsc.sg_cpa import SGCpa
from sg_cpa_phase4_final import CKPT, A_FILE
DEV="cuda:0"
CELLS={("Camera","libero_spatial"):[705,620,671],("Background","libero_spatial"):[11,19,18],
       ("Camera","libero_object"):[653,701,901],("Background","libero_object"):[3,8,18]}
class C: pass
c=C(); c.libero_src=Path("/hy-tmp/libero_plus"); c.libero_config_path=Path("/hy-tmp/libero_plus_config"); c.render_gpu=0; c.device=DEV
configure_libero(c)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
policy=load_policy(CKPT, device=DEV)
sg=SGCpa(policy, torch.tensor(json.loads(A_FILE.read_text())["action_subspace"],dtype=torch.float64), seed=1234)
suites={s:benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial","libero_object")}
def fork(): return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])
def mkenv(bddl, states, seed):
    e=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
    np.random.seed(seed); torch.manual_seed(seed); e.seed(seed); e.reset(); o=e.set_init_state(states[0])
    for _ in range(SETTLING_STEPS): o,_,d,_=e.step(DUMMY_ACTION)
    return e,o
def obj_name(suite, inst):
    if suite=="libero_spatial": return "akita_black_bowl_1_main"
    for cand in ("alphabet_soup_1_main","salad_dressing_1_main","butter_1_main"):
        if cand.split("_")[0] in inst.lower(): return cand
    return "alphabet_soup_1_main"
def prep_of(policy, obs, task):
    return prepare_observation(policy_observation(obs,task.language,policy),config=policy.config,normalizer=policy.normalizer,device=torch.device(DEV))
def sample_chunk(policy, prep, seed):
    rep=policy.representation_extractor(**prep)
    with torch.no_grad(): sg.adapter.beta.zero_()
    with fork():
        torch.manual_seed(seed)
        with torch.no_grad(): return policy.action_generator.sample(rep).detach()
rows=[]; branch_stats=[]
for (domain,suite),tids in CELLS.items():
    for tid in tids:
        task=suites[suite].get_task(tid)
        bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
        states=torch.load(Path(get_libero_path("init_states"))/task.problem_folder/task.init_states_file,weights_only=False)
        for ep in range(2):
            seed=30+tid*1000+ep
            A,oA=mkenv(bddl,states,seed); B,oB=mkenv(bddl,states,seed)
            names=[str(x) for x in A.sim.model.body_names]; oi=names.index(obj_name(suite,task.language)); gi=names.index("gripper0_eef")
            obj0=np.array(A.sim.data.body_xpos[oi])
            step=0; branch=0; done=False
            while not done and branch<=12 and step<240:
                if branch in (8,12):
                    preA=prep_of(policy,oA,task); preB=prep_of(policy,oB,task)
                    obs_diff=max(float((preA["images"][v]-preB["images"][v]).abs().max()) for v in preA["images"])
                    actA=sample_chunk(policy,preA,seed+7919*branch); actB=sample_chunk(policy,preB,seed+7919*branch)
                    chunk_diff=float((actA-actB).abs().max())
                    eef=np.array(A.sim.data.body_xpos[gi]); op=np.array(A.sim.data.body_xpos[oi])
                    contact=float(np.linalg.norm(eef-op)); lift=float(op[2]-obj0[2])
                    eligible = contact<=0.10
                    if not eligible:
                        branch_stats.append({"domain":domain,"suite":suite,"task_id":tid,"episode":ep,"branch":branch,"eligible":False,"contact":contact,"lift":lift})
                        # advance without branching
                        ctrl=policy.normalizer.unnormalize_action(actA)[0].float().cpu().numpy()
                        for a in ctrl[:6]:
                            oA,_,done,_=A.step(np.clip(a,-1.0,1.0)); step+=1
                            if done: break
                        oB=oB  # B replays same actions below
                        prepB=prep_of(policy,oB,task); actB2=sample_chunk(policy,prepB,seed+7919*branch)
                        ctrlB=policy.normalizer.unnormalize_action(actB2)[0].float().cpu().numpy()
                        for a in ctrlB[:6]:
                            oB,_,doneB,_=B.step(np.clip(a,-1.0,1.0))
                            if doneB: break
                        branch+=1; continue
                    # zero-update 4x6 both branches
                    trA=[]; trB=[]; eefA=[]; eefB=[]; stepsA=stepsB=0; succA=succB=False
                    for k in range(4):
                        for env,obs,tr,ee,isA in ((A,oA,trA,eefA,True),(B,oB,trB,eefB,False)):
                            prep=prep_of(policy,obs,task)
                            ch=sample_chunk(policy,prep,seed+7919*(branch+k))
                            ctrl=policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                            for a in ctrl[:6]:
                                obs,reward,d,info=env.step(np.clip(a,-1.0,1.0))
                                tr.append([float(x) for x in env.sim.data.body_xpos[oi]])
                                ee.append([float(x) for x in env.sim.data.body_xpos[gi]])
                                if isA: stepsA+=1; succA=succA or is_success(env,info,reward,d)
                                else: stepsB+=1; succB=succB or is_success(env,info,reward,d)
                                if d: break
                            if isA: oA=obs
                            else: oB=obs
                    L=min(len(trA),len(trB))
                    td=max(float(np.abs(np.array(trA[:L])-np.array(trB[:L])).max()),0.0) if L else None
                    ed=max(float(np.abs(np.array(eefA[:L])-np.array(eefB[:L])).max()),0.0) if L else None
                    rows.append({"domain":domain,"suite":suite,"task_id":tid,"episode":ep,"branch":branch,
                                 "contact":contact,"lift":lift,"obs_diff":obs_diff,"chunk_diff":chunk_diff,
                                 "nA":len(trA),"nB":len(trB),"max_obj_diff":td,"max_eef_diff":ed,
                                 "stepsA":stepsA,"stepsB":stepsB,"succA":bool(succA),"succB":bool(succB)})
                    branch_stats.append({"domain":domain,"suite":suite,"task_id":tid,"episode":ep,"branch":branch,"eligible":True,"contact":contact,"lift":lift})
                    # after branch, continue from current envs (note: this mutates states; audit branch only once per episode)
                    break  # stop this episode after first eligible branch audit
                # advance one replan frozen
                prep=prep_of(policy,oA,task); ch=sample_chunk(policy,prep,seed+7919*branch)
                ctrl=policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                for a in ctrl[:6]:
                    oA,_,done,_=A.step(np.clip(a,-1.0,1.0)); step+=1
                    if done: break
                prepB=prep_of(policy,oB,task); chB=sample_chunk(policy,prepB,seed+7919*branch)
                ctrlB=policy.normalizer.unnormalize_action(chB)[0].float().cpu().numpy()
                for a in ctrlB[:6]:
                    oB,_,doneB,_=B.step(np.clip(a,-1.0,1.0))
                    if doneB: break
                branch+=1
            A.close(); B.close()
            print("preflight",domain,suite,tid,ep,"branches",branch,flush=True)
json.dump({"rows":rows,"branches":branch_stats},open("/hy-tmp/validation/b2_full_preflight.json","w"),indent=2)
print("eligible branches",len(rows),"skipped",sum(1 for b in branch_stats if not b["eligible"]))
print("max obs diff",max(r["obs_diff"] for r in rows),"max chunk diff",max(r["chunk_diff"] for r in rows),
      "max traj diff",max(r["max_obj_diff"] for r in rows if r["max_obj_diff"] is not None),
      "max eef diff",max(r["max_eef_diff"] for r in rows if r["max_eef_diff"] is not None),
      "steps equal",all(r["stepsA"]==r["stepsB"] for r in rows))
