"""Bounded Phase3B development pilot; native Vanilla IO plus audited Plus assets.

No result-dependent selection. Never reads a Blind/Held-out manifest.
"""
from __future__ import annotations
import argparse, ast, ctypes, hashlib, itertools, json, random, re, time
from pathlib import Path
import numpy as np
import torch
from phase3_smolvla_common import (FixedInitSuite, load_actor, reference_for,
    snapshot, seed_torch, rng_hash, save_json, sha, array_hash, predict_at)
from dacsc.smolvla_v1_hook import SmolVLAV1Hook
from dacsc.static_visual_v2 import isolated_seed, seed_for

PLUS=Path('/hy-tmp/LIBERO-plus')
FAMILIES=('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')
KEYS=('agentview_image','robot0_eye_in_hand_image')


def official_definitions():
    path=PLUS/'libero/libero/envs/problems/libero_tabletop_manipulation.py'
    tree=ast.parse(path.read_text()); definitions={}
    for node in tree.body:
        if not isinstance(node,ast.ClassDef) or node.name=='Libero_Tabletop_Manipulation':continue
        updates={}
        for call in ast.walk(node):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='update':
                for arg in call.args:
                    value=ast.literal_eval(arg)
                    assert set(value)<= {'scene_xml','scene_properties'},(node.name,value)
                    updates.update(value)
        if set(updates)=={'scene_xml','scene_properties'}:
            definitions[node.name.lower()]=(node.name,updates)
    return path,definitions


def install_asset(case):
    """Register official scene selectors on the unchanged Vanilla base class.

    Plus base also changes camera rounding; importing that entire backend would
    break the reproduced native reference. Only explicit official scene settings
    are transferred, with source class/XML/texture receipts and physics checks.
    """
    if case['group'] not in ('Lighting','Background'):return
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING
    # Vanilla's registration decorator returns None; get the actual class from
    # the registry rather than the decorated module symbol.
    Libero_Tabletop_Manipulation=TASK_MAPPING['libero_tabletop_manipulation']
    from libero.libero.envs.arenas.style import STYLE_MAPPING
    from libero.libero.envs.regions import REGION_SAMPLERS
    name,settings=case['official_class'],case['scene_settings']
    scene=str(PLUS/'libero/libero/assets'/settings['scene_xml'])
    style_tree=ast.parse((PLUS/'libero/libero/envs/arenas/style.py').read_text())
    for node in style_tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('WALL_STYLE','FLOOR_STYLE'):
                    STYLE_MAPPING['wall' if target.id=='WALL_STYLE' else 'floor'].update(ast.literal_eval(node.value))
    def init(self,bddl_file_name,*args,**kwargs):
        kwargs['scene_xml']=scene
        kwargs['scene_properties']=dict(settings['scene_properties'])
        Libero_Tabletop_Manipulation.__init__(self,bddl_file_name,*args,**kwargs)
    TASK_MAPPING[name.lower()]=type(name,(Libero_Tabletop_Manipulation,),{'__init__':init})
    REGION_SAMPLERS[name.lower()]=dict(REGION_SAMPLERS['libero_tabletop_manipulation'])


def corruption_functions():
    # Execute verbatim official function/class AST, without importing Plus env.
    import cv2
    from io import BytesIO
    from wand.api import library as wandlibrary
    from wand.image import Image as WandImage
    from skimage.filters import gaussian
    from scipy.ndimage import zoom as scizoom
    wandlibrary.MagickMotionBlurImage.argtypes=(ctypes.c_void_p,ctypes.c_double,ctypes.c_double,ctypes.c_double)
    scope=dict(np=np,cv2=cv2,BytesIO=BytesIO,wandlibrary=wandlibrary,WandImage=WandImage,gaussian=gaussian,scizoom=scizoom)
    tree=ast.parse((PLUS/'libero/libero/envs/env_wrapper.py').read_text())
    names=set(FAMILIES)|{'MotionImage','clipped_zoom','plasma_fractal'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==names
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official-plus-corruptions>','exec'),scope)
    return {name:scope[name] for name in FAMILIES}


def perturb(obs,case,step,funcs):
    if case['group']!='Sensor':return obs
    from PIL import Image
    out=dict(obs)
    for view,key in enumerate(KEYS):
        image=np.asarray(obs[key]);assert image.dtype==np.uint8 and image.shape==(256,256,3)
        with isolated_seed(seed_for('phase3b',case['id'],step,view)):
            changed=funcs[case['subgroup']](Image.fromarray(image),severity=5)
        changed=np.asarray(changed).astype(np.uint8)
        assert changed.shape==image.shape and np.isfinite(changed).all()
        out[key]=changed
    return out


def setup(case):
    suite=FixedInitSuite(case['init_index']);spec=list(suite.iter_episodes(0,10))[case['task_id']]
    meta=suite._spec_index[spec.episode_id]
    assert spec.task==case['base_task'] and meta['language']==case['language']
    assert not re.search(r'\d',meta['language'])
    if case['group'] in ('Lighting','Background'):
        install_asset(case);meta['bddl_file']=str(PLUS/case['bddl'])
    random.seed(spec.seed);np.random.seed(spec.seed);ctypes.CDLL(None).srand(ctypes.c_uint(spec.seed))
    obs=suite.reset(spec)
    assert int(suite._env.noise)==0 and obs['language']==case['language']
    return suite,spec,obs


def manifest(out,calib):
    suite=FixedInitSuite(1);specs=list(suite.iter_episodes(0,10));tasks=[s.task for s in specs]
    devpath=Path('/hy-tmp/patch-diag/docs/data/static_visual_v2_suite/final/dev_manifest.json')
    dev=json.loads(devpath.read_text());assert dev['kind']!='blind'
    auditpath=Path('/hy-tmp/patch-diag/protocols/static_visual_v2/p0/p0_audit.json')
    audit=json.loads(auditpath.read_text())
    # The 100-case Dev subset omits some task/factor cells. Use the existing
    # historical official-pool audit, whose rows were ALL marked development-only
    # after the earlier full Frozen evaluation. Never read the protected pool.
    settings_path=Path('/hy-tmp/patch-diag/protocols/static_visual_v2/settings_suite_factor.json')
    settings=json.loads(settings_path.read_text())
    assert settings['historical_dev_reuse_authorized']
    assert any(q['suite']=='*' and q['factor']=='*' and 'all official instances are development-only' in q['reason']
               for q in settings['historical_scope_quarantine'])
    pool=[dict(c,role='target_dev') for c in audit['registry']]
    assert pool
    definitions_path,definitions=official_definitions();cases=[]
    # Existing Dev has uneven subgroup coverage by task. Find a 5/5 assignment
    # using only eligible Dev assets, before any model outcome is observed.
    assignment=next(combo for combo in itertools.product(('table_texture','scene_texture'),repeat=10)
        if combo.count('table_texture')==5 and all(any(c['base_task']==tasks[t] and c['factor']=='background'
            and c['sub_dimension']==combo[t] and c['role']=='target_dev' for c in pool) for t in range(10)))
    for task,spec in enumerate(specs):
        base={'task_id':task,'base_task':spec.task,'language':suite._spec_index[spec.episode_id]['language'],
              'policy_seed':0,'role':'development; previously accessed official pool; NOT independent confirmation'}
        cases.append(dict(base,id=f't{task:02d}_clean_i2',group='Clean',subgroup='Clean',init_index=2))
        for fi in (task%5,(task+1)%5):
            cases.append(dict(base,id=f't{task:02d}_{FAMILIES[fi]}_i1',group='Sensor',subgroup=FAMILIES[fi],init_index=1,severity=5))
        for group,factor,sub in [('Lighting','lighting','light_asset'),('Background','background',assignment[task])]:
            candidates=[c for c in pool if c['base_task']==spec.task and c['factor']==factor and c['sub_dimension']==sub and c['role']=='target_dev']
            assert candidates,(task,group,sub)
            dev_ids={c['case_id'] for c in dev['cases']}
            chosen=min(candidates,key=lambda c:(c['case_id'] not in dev_ids,c['case_id']))
            path=PLUS/chosen['bddl'];assert sha(path)==chosen['resolved_bddl_hash']
            problem=re.search(r'\(problem\s+([^\s)]+)',path.read_text()).group(1).lower()
            name,settings=definitions[problem];scene=PLUS/'libero/libero/assets'/settings['scene_xml']
            assert scene.is_file(),scene
            row=dict(base,id=f't{task:02d}_{sub}_i1',group=group,subgroup=sub,init_index=1,
                     bddl=chosen['bddl'],bddl_sha256=sha(path),dev_case_id=chosen['case_id'],
                     official_class=name,scene_settings=settings,scene_sha256=sha(scene))
            cases.append(row)
    assert len(cases)==50 and len({c['id'] for c in cases})==50
    assert all(sum(c['subgroup']==f for c in cases)==4 for f in FAMILIES)
    assert sum(c['subgroup']=='table_texture' for c in cases)==sum(c['subgroup']=='scene_texture' for c in cases)==5
    with np.load(calib,allow_pickle=False) as z:assert list(z['task_names'])==tasks
    source=out/'source_code'
    save_json(out/'manifest.json',dict(schema='smolvla_phase3b_v1',cases=cases,calib_sha256=sha(calib),
        dev_manifest_sha256=sha(devpath),official_pool_audit_sha256=sha(auditpath),historical_scope_settings_sha256=sha(settings_path),asset_definition_sha256=sha(definitions_path),
        corruption_source_sha256=sha(PLUS/'libero/libero/envs/env_wrapper.py'),
        source_sha256={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        protocol={'model':'native upstream SmolVLA, BF16 vision, FP32 operator then BF16, CPU normalization',
                  'E':1,'H':50,'Euler':10,'max_steps':220,'warmup':10,'alpha':.75,'layer':3,
                  'reference':'unchanged init0 full-trajectory outer9 / inner8+1 task LOTO',
                  'environment':'unchanged Vanilla base; official Plus BDDL scene selectors bridged onto Vanilla base',
                  'sensor':'official Plus severity5 (10-level scale); both cameras every policy step from t0; independent deterministic view seeds',
                  'pairing':'exact initial state/RGB and Torch CPU/CUDA common-prefix RNG hashes',
                  'comparison_scope':'exploratory one policy seed; not SALT-matched protocol or a noninferiority test',
                  'blind_access':False,'main_table_autostart':False}))
    suite.close()


def physics(suite):
    m=suite._env.sim.model
    names=('body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat','geom_size','geom_friction',
           'geom_contype','geom_conaffinity','jnt_pos','jnt_axis','dof_damping')
    result={key:np.asarray(getattr(m,key)).copy() for key in names}
    # Official scenes include many additional *unused* camera definitions.
    # Match the two policy cameras by name, not their shifted numeric IDs.
    for camera in ('agentview','robot0_eye_in_hand'):
        idx=m.camera_name2id(camera)
        for key in ('cam_pos','cam_quat','cam_fovy'):
            result[camera+'_'+key]=np.asarray(getattr(m,key)[idx]).copy()
    return result


def preflight(out,cases,calib):
    funcs=corruption_functions();audit=[]
    for case in cases:
        if case['group'] not in ('Lighting','Background'):continue
        clean=dict(case,group='Clean');s0,_,o0=setup(clean)
        try:p0=physics(s0);state0=np.asarray(s0._env.sim.get_state().flatten()).copy()
        finally:s0.close()
        s1,_,o1=setup(case)
        try:
            p1=physics(s1);state1=np.asarray(s1._env.sim.get_state().flatten()).copy()
            assert p0.keys()==p1.keys()
            for key in p0:assert np.array_equal(p0[key],p1[key]),(case['id'],'physics/camera changed',key)
            assert np.array_equal(state0,state1),(case['id'],'initial state changed')
            diff=[float(np.mean(np.abs(o0[k].astype(float)-o1[k].astype(float)))) for k in KEYS]
            assert max(diff)>0,(case['id'],'visual perturbation absent')
            audit.append({'id':case['id'],'physics_camera_exact':True,'initial_state_exact':True,'image_MAE':diff})
        finally:s1.close()
    sample=next(c for c in cases if c['group']=='Clean');s,_,obs=setup(sample)
    try:
        sensor_checks=[]
        for family in FAMILIES:
            c=dict(sample,group='Sensor',subgroup=family)
            before=np.random.get_state();a=perturb(obs,c,0,funcs);b=perturb(obs,c,0,funcs);after=np.random.get_state()
            assert all(np.array_equal(x,y) for x,y in zip(before,after))
            assert all(np.array_equal(a[k],b[k]) and not np.array_equal(a[k],obs[k]) for k in KEYS)
            sensor_checks.append({'family':family,'both_views_changed':True,'reproducible':True,'numpy_rng_preserved':True})
        actor=load_actor();refs=reference_for(calib,sample['task_id'])
        identity=[]
        for family in (None,)+FAMILIES:
            image=obs if family is None else perturb(obs,dict(sample,group='Sensor',subgroup=family),0,funcs)
            raw,_=predict_at(actor,image);zero,_=predict_at(actor,image,refs,3,0,layers=(3,))
            assert np.array_equal(raw,zero)
            identity.append(family or 'Clean')
        save_json(out/'preflight.json',{'status':'PASS','physical_assets':audit,'sensor':sensor_checks,
                 'native_alpha0_action_exact':True,'identity_conditions':identity,'script_sha256':sha(__file__)})
    finally:s.close()


def rollout(actor,case,arm,refs,funcs):
    suite,spec,obs=setup(case);raw_start=snapshot(suite,obs)
    observed=perturb(obs,case,0,funcs);observed_start=snapshot(suite,observed)
    actor.reset(0);seed_torch(0)
    h=SmolVLAV1Hook(actor._model,refs,layers=(3,),active_layer=3 if arm=='v1' else None,alpha=.75)
    frames=[];noise=[];actions=hashlib.sha256();began=time.monotonic()
    try:
        success=False
        for step in range(220):
            modelobs=observed if step==0 else perturb(obs,case,step,funcs)
            noise.append(rng_hash());h.begin();action=actor.act(modelobs);stats=h.end()
            assert action.shape==(7,) and np.isfinite(action).all()
            actions.update(action.tobytes())
            frames.append({'step':step,'active':[float(stats[3,v]['active_fraction']) for v in range(2)],
                           'gain_abs':[float(stats[3,v]['gain_abs_mean']) for v in range(2)],
                           'action':action.tolist(),'eef':np.asarray(obs['robot0_eef_pos']).tolist()})
            result=suite.step(action);obs=result.obs;success=bool(result.success)
            if result.done or success:break
        return dict(case_id=case['id'],task_id=case['task_id'],group=case['group'],subgroup=case['subgroup'],
                    init_index=case['init_index'],policy_seed=0,environment_seed=spec.seed,arm=arm,success=success,
                    steps=step+1,seconds=time.monotonic()-began,start=raw_start,observed_start=observed_start,
                    rng_before_replans=noise,actions_sha256=actions.hexdigest(),frames=frames,
                    finite_actions=True,two_camera_calls_every_replan=True)
    finally:h.close();suite.close()


def worker(out,cases,calib,shard,workers):
    pre=json.loads((out/'preflight.json').read_text())
    assert pre['status']=='PASS' and pre['script_sha256']==sha(__file__)
    actor=load_actor();versions=[p._version for p in actor._model.parameters()];funcs=corruption_functions()
    dest=out/'pairs';dest.mkdir(exist_ok=True)
    for index,case in enumerate(cases):
        if index%workers!=shard:continue
        refs=reference_for(calib,case['task_id']);rows=[]
        # Counterbalance order independently of condition outcome.
        for arm in (('frozen','v1') if index%2==0 else ('v1','frozen')):
            row=rollout(actor,case,arm,refs,funcs)
            save_json(dest/f"{case['id']}_{arm}.json",row);rows.append(row)
            print('ARM_DONE',case['id'],arm,row['success'],row['steps'],round(row['seconds'],1),flush=True)
        r0,r1=sorted(rows,key=lambda r:r['arm'])
        assert r0['start']==r1['start'] and r0['observed_start']==r1['observed_start'],case['id']
        n=min(r0['steps'],r1['steps']);assert r0['rng_before_replans'][:n]==r1['rng_before_replans'][:n],case['id']
        save_json(dest/f"{case['id']}_pair.json",dict(case_id=case['id'],start_exact=True,paired_rng_prefix_exact=True))
    assert versions==[p._version for p in actor._model.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('manifest','preflight','worker'));p.add_argument('--out',required=True,type=Path)
    p.add_argument('--calib',type=Path,default=Path('/hy-tmp/patch-diag/runs/phase3_smolvla/spatial_init0_calib.npz'))
    p.add_argument('--shard',type=int,default=0);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='manifest':manifest(a.out,a.calib)
    else:
        m=json.loads((a.out/'manifest.json').read_text());assert sha(a.calib)==m['calib_sha256']
        if a.mode=='worker':
            for relative,fingerprint in m['source_sha256'].items():
                assert sha(a.out/'source_code'/relative)==fingerprint,relative
        assert sha(PLUS/'libero/libero/envs/env_wrapper.py')==m['corruption_source_sha256']
        for c in m['cases']:
            assert c['init_index']>0
            if 'bddl' in c:
                assert sha(PLUS/c['bddl'])==c['bddl_sha256']
                assert sha(PLUS/'libero/libero/assets'/c['scene_settings']['scene_xml'])==c['scene_sha256']
        if a.mode=='preflight':preflight(a.out,m['cases'],a.calib)
        else:worker(a.out,m['cases'],a.calib,a.shard,a.workers)
