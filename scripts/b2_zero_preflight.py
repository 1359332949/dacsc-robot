"""Two-env 4x6 zero-update preflight (production execution protocol)."""
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
class C: pass
c=C(); c.libero_src=Path("/hy-tmp/libero_plus"); c.libero_config_path=Path("/hy-tmp/libero_plus_config"); c.render_gpu=0; c.device=DEV
configure_libero(c)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
policy=load_policy(CKPT, device=DEV)
a_basis=torch.tensor(json.loads(A_FILE.read_text())["action_subspace"],dtype=torch.float64)
sg=SGCpa(policy,a_basis,seed=1234)
suites={s:benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial","libero_object")}
def fork(): return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])
def mkenv(bddl, states, seed):
    e=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
    np.random.seed(seed); torch.manual_seed(seed); e.seed(seed); e.reset(); o=e.set_init_state(states[0])
    for _ in range(SETTLING_STEPS): o,_,d,_=e.step(DUMMY_ACTION)
    return e,o
def run():
    rows=[]
    for suite,tid in (("libero_spatial",705),("libero_object",3)):
        task=suites[suite].get_task(tid)
        bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
        states=torch.load(Path(get_libero_path("init_states"))/task.problem_folder/task.init_states_file,weights_only=False)
        seed=30+tid*1000
        A,oA=mkenv(bddl,states,seed); B,oB=mkenv(bddl,states,seed)
        names=[str(x) for x in A.sim.model.body_names]
        obj="akita_black_bowl_1_main" if suite=="libero_spatial" else "alphabet_soup_1_main"
        tgt="plate_1_main" if suite=="libero_spatial" else "basket_1_main"
        oi=names.index(obj); ti=names.index(tgt)
        d0=float(np.linalg.norm(np.array(A.sim.data.body_xpos[oi])-np.array(A.sim.data.body_xpos[ti])))
        # prefix 8 replans, frozen, matched seeds
        def replan(env, obs, k):
            prep=prepare_observation(policy_observation(obs,task.language,policy),config=policy.config,normalizer=policy.normalizer,device=torch.device(DEV))
            rep=policy.representation_extractor(**prep)
            with torch.no_grad(): sg.adapter.beta.zero_()
            with fork():
                torch.manual_seed(seed+7919*k)
                with torch.no_grad(): ch=policy.action_generator.sample(rep)
            ctrl=policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
            for a in ctrl[:6]:
                obs,_,d,_=env.step(np.clip(a,-1.0,1.0))
                if d: break
            return obs
        for k in range(8):
            oA=replan(A,oA,k); oB=replan(B,oB,k)
        trajA=[];trajB=[]
        # 4 segments x 6 steps, both beta=0
        for k in range(4):
            for env,obs,traj,which in ((A,oA,trajA,"A"),(B,oB,trajB,"B")):
                prep=prepare_observation(policy_observation(obs,task.language,policy),config=policy.config,normalizer=policy.normalizer,device=torch.device(DEV))
                rep=policy.representation_extractor(**prep)
                with torch.no_grad(): sg.adapter.beta.zero_()
                with fork():
                    torch.manual_seed(seed+7919*(8+k))
                    with torch.no_grad(): ch=policy.action_generator.sample(rep)
                ctrl=policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                for a in ctrl[:6]:
                    obs,_,d,info=env.step(np.clip(a,-1.0,1.0))
                    traj.append([float(x) for x in env.sim.data.body_xpos[oi]])
                    if d: break
                if which=="A": oA=obs
                else: oB=obs
        pa=np.array(trajA);pb=np.array(trajB);L=min(len(pa),len(pb))
        maxdiff=float(np.abs(pa[:L]-pb[:L]).max()) if L else None
        rows.append({"suite":suite,"task_id":tid,"max_traj_diff":maxdiff,"nA":len(trajA),"nB":len(trajB)})
        print(rows[-1])
        A.close();B.close()
    Path("/hy-tmp/validation/b2_zero_preflight.json").write_text(json.dumps(rows,indent=2))
run()
