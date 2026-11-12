"""Per-episode, per-branch zero-update audit with separate denominators (branch 8 and 12)."""
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
def mkenv(bddl,states,seed,ep=0):
    e=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
    np.random.seed(seed); torch.manual_seed(seed); e.seed(seed); e.reset(); o=e.set_init_state(states[ep % len(states)])
    for _ in range(SETTLING_STEPS): o,_,d,_=e.step(DUMMY_ACTION)
    return e,o
def prep_of(obs,task):
    return prepare_observation(policy_observation(obs,task.language,policy),config=policy.config,normalizer=policy.normalizer,device=torch.device(DEV))
def chunk(prep,seed):
    rep=policy.representation_extractor(**prep)
    with torch.no_grad(): sg.adapter.beta.zero_()
    with fork():
        torch.manual_seed(seed)
        with torch.no_grad(): return policy.action_generator.sample(rep).detach()
def objname(suite,inst):
    if suite=="libero_spatial": return "akita_black_bowl_1_main"
    for x in ("alphabet_soup_1_main","salad_dressing_1_main","butter_1_main"):
        if x.split("_")[0] in inst.lower(): return x
    return "alphabet_soup_1_main"
out=[]
for (domain,suite),tids in CELLS.items():
    for tid in tids:
        task=suites[suite].get_task(tid)
        bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
        states=torch.load(Path(get_libero_path("init_states"))/task.problem_folder/task.init_states_file,weights_only=False)
        for ep in range(2):
            seed=30+tid*1000+ep
            for branch in (8,12):
                A,oA=mkenv(bddl,states,seed,ep); B,oB=mkenv(bddl,states,seed,ep)
                names=[str(x) for x in A.sim.model.body_names]; oi=names.index(objname(suite,task.language)); gi=names.index("gripper0_eef")
                obj0=np.array(A.sim.data.body_xpos[oi])
                step=0; succ=False; done=False; t=0; ok=True
                while t<branch and not done:
                    for env,obs,isA in ((A,oA,True),(B,oB,False)):
                        ch=chunk(prep_of(obs,task),seed+7919*t)
                        ctrl=policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                        for a in ctrl[:6]:
                            obs,r,d,info=env.step(np.clip(a,-1.0,1.0)); step+=1
                            succ=succ or is_success(env,info,r,d); done=done or d
                            if d: break
                        if isA: oA=obs
                        else: oB=obs
                    t+=1
                eef=np.array(A.sim.data.body_xpos[gi]); op=np.array(A.sim.data.body_xpos[oi])
                contact=float(np.linalg.norm(eef-op)); lift=float(op[2]-obj0[2])
                eligible=(contact<=0.10) and (not succ) and (not done)
                rec={"domain":domain,"suite":suite,"task_id":tid,"episode":ep,"branch":branch,
                     "eligible":eligible,"contact":contact,"lift":lift,"success_before":bool(succ),"done_before":bool(done)}
                if eligible:
                    trA=[];trB=[];eA=[];eB=[];stepsA=stepsB=0;succA=succB=False
                    for k in range(4):
                        for env,obs,tr,ee,isA in ((A,oA,trA,eA,True),(B,oB,trB,eB,False)):
                            ch=chunk(prep_of(obs,task),seed+7919*(branch+k))
                            ctrl=policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                            for a in ctrl[:6]:
                                obs,r,d,info=env.step(np.clip(a,-1.0,1.0))
                                tr.append([float(x) for x in env.sim.data.body_xpos[oi]])
                                ee.append([float(x) for x in env.sim.data.body_xpos[gi]])
                                if isA: stepsA+=1; succA=succA or is_success(env,info,r,d)
                                else: stepsB+=1; succB=succB or is_success(env,info,r,d)
                                if d: break
                            if isA: oA=obs
                            else: oB=obs
                    L=min(len(trA),len(trB))
                    rec.update({"obs_diff":None,"chunk_diff":None,
                                "max_obj_diff":max(float(np.abs(np.array(trA[i])-np.array(trB[i])).max()) for i in range(L)) if L else None,
                                "max_eef_diff":max(float(np.abs(np.array(eA[i])-np.array(eB[i])).max()) for i in range(L)) if L else None,
                                "stepsA":stepsA,"stepsB":stepsB,"succA":bool(succA),"succB":bool(succB)})
                out.append(rec)
                A.close(); B.close()
                print("audit",domain,suite,tid,ep,branch,eligible,flush=True)
json.dump(out,open("/hy-tmp/validation/b2_zero_branch_audit.json","w"),indent=2)
from collections import Counter
print(Counter((r["branch"],r["eligible"]) for r in out))
