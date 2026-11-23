"""V2 entry point: audit, freeze, run Frozen/source smoke, summarize. Blind always locked."""
from __future__ import annotations
import argparse,ast,collections,json,re,subprocess,sys,time,functools,importlib.metadata,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import (FACTORS,PROTECTED,canonical,digest,file_hash,array_hash,seed_for,
    resolve_task,equivalence_key,authorize,append_event,EnvironmentRNG,sensor_transform,deterministic_partition,deterministic_dev_sample)


def write(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def git_commit(root):return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
def pinned_sources(root):return {p:file_hash(root/p) for p in ('libero/libero/benchmark/task_classification.json','libero/libero/benchmark/libero_suite_task_map.py','libero/libero/benchmark/__init__.py','libero/libero/envs/env_wrapper.py')}

@functools.lru_cache(maxsize=1)
def runtime_fingerprint():
    names=('torch','robosuite','mujoco','numpy','Pillow','transformers','PyOpenGL')
    versions={}
    for name in names:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    xml={}
    spec=importlib.util.find_spec('robosuite')
    if spec and spec.origin:
        root=Path(spec.origin).parent/'models/assets'
        for file in sorted(root.rglob('*.xml')):xml[str(file.relative_to(root))]=file_hash(file)
    return {'versions':versions,'robosuite_xml_asset_index_sha256':digest(xml),'robosuite_xml_count':len(xml)}


def scan_history(roots,names):
    """Conservatively quarantine exact logical references; no claim all references were rolled out."""
    hits=collections.defaultdict(list);scanned=[];skipped=[]
    for root in roots:
        root=Path(root)
        if not root.exists():skipped.append(str(root));continue
        for p in sorted(root.rglob('*')):
            if p.suffix not in ('.json','.jsonl','.md','.py','.log') or not p.is_file():continue
            if any(part in ('.git','.venv','node_modules','__pycache__','static_visual_v2') for part in p.parts):continue
            if p.stat().st_size>32_000_000:skipped.append(str(p));continue
            try:text=p.read_text(errors='replace')
            except OSError:skipped.append(str(p));continue
            matches=set(re.findall(r'[A-Za-z0-9_]+',text))&names
            if matches:
                evidence={'path':str(p),'sha256':file_hash(p),'kind':'historical_reference_quarantine'}
                for name in matches:hits[name].append(evidence)
            scanned.append(str(p))
    return hits,{'roots':[str(p) for p in roots],'files_scanned':len(scanned),'unread_or_large_files':skipped,
                 'scope':'exact references conservatively quarantined; cannot certify completeness without server/log review'}


def audit(args):
    import torch
    settings=json.loads(args.settings.read_text());src=args.libero_src;commit=git_commit(src)
    if commit!=settings['upstream_candidate_commit']:raise ValueError('upstream revision changed; do not silently rebase IDs')
    if file_hash(args.checkpoint)!=settings['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    source=pinned_sources(src);cls=json.loads((src/'libero/libero/benchmark/task_classification.json').read_text())
    tree=ast.parse((src/'libero/libero/benchmark/libero_suite_task_map.py').read_text());maps=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in node.targets):maps=ast.literal_eval(node.value)
    if maps is None:raise ValueError('missing task map')
    for suite,rows in cls.items():
        if len(rows)!=len(maps[suite]):raise ValueError('registry count mismatch')
        for index,row in enumerate(rows):
            if row['id']!=index+1 or row['name']!=maps[suite][index]:raise ValueError('classification ID / evaluator index mismatch')
    names={r['name'] for rows in cls.values() for r in rows};history,scan=scan_history(args.history_root,names)
    rows=[];errors=[];states={};seen={};suite=settings['candidate_suite']
    for item in cls[suite]:
        if item['category'] not in FACTORS:continue
        factor=FACTORS[item['category']];name=item['name']
        try:
            resolved=resolve_task(name,factor);base=resolved['base_task']
            bp=src/'libero/libero/bddl_files'/suite/resolved['resolved_bddl'];cp=bp.with_name(base+'.bddl')
            ip=src/'libero/libero/init_files'/suite/resolved['init_file']
            for p in (bp,cp,ip):
                if not p.is_file():raise FileNotFoundError(p)
            if base not in states:
                a=torch.load(ip,map_location='cpu',weights_only=False)
                if np.asarray(a).ndim!=2 or len(a)<31:raise ValueError('insufficient initial states')
                states[base]=np.asarray(a);seen[base]={'init_file':str(ip.relative_to(src)),'init_file_sha256':file_hash(ip),'clean_bddl':str(cp.relative_to(src)),'clean_bddl_sha256':file_hash(cp)}
            idx=settings['target_init_index'];ih=array_hash(states[base][idx]);rh=file_hash(bp)
            eq=equivalence_key(suite,base,rh,resolved['runtime_params'],file_hash(ip))
            reasons=[]
            if item['id'] in settings['known_accessed_classification_ids'].get(suite,[]):reasons.append('known_accessed_id')
            if name in history:reasons.append('historical_reference')
            for q in settings['historical_scope_quarantine']:
                if q['suite'] in ('*',suite) and q['factor'] in ('*',factor):reasons.append(q['reason'])
            row={'case_id':digest([commit,suite,name])[:24],'commit':commit,'suite':suite,'base_task':base,'logical_task_name':name,
                'classification_id':item['id'],'evaluator_index':item['id']-1,'factor':factor,'difficulty':item['difficulty_level'],
                'sub_dimension':resolved['sub_dimension'],'sub_dimension_source':resolved['sub_dimension_source'],
                'bddl':str(bp.relative_to(src)),'resolved_bddl_hash':rh,'clean_bddl':str(cp.relative_to(src)),'clean_bddl_sha256':file_hash(cp),
                'init_state_file':str(ip.relative_to(src)),'init_state_file_sha256':file_hash(ip),'init_state_index':idx,
                'raw_init_hash':ih,'physical_init_hash':None,'pairing_status':'raw_init_matched_post_reset_unverified',
                'pair_group_id':digest([suite,base,ih]),'equivalence_group':eq,'runtime_params':resolved['runtime_params'],
                'history_status':'quarantine' if reasons else 'candidate','history_reasons':reasons,
                'history_evidence':history.get(name,[])}
            rows.append(row)
        except (ValueError,FileNotFoundError) as e:errors.append({'logical_task_name':name,'error':str(e)})
    banned={r['equivalence_group'] for r in rows if r['history_status']!='candidate'}
    for r in rows:
        if r['equivalence_group'] in banned and r['history_status']=='candidate':r['history_status']='quarantine';r['history_reasons'].append('equivalent_to_historical_reference')
    cells=[];short=[]
    for base in sorted(states):
        for factor in settings['factors']:
            cell=[r for r in rows if (r['base_task'],r['factor'])==(base,factor)]
            n=len({r['equivalence_group'] for r in cell if r['history_status']=='candidate'})
            cells.append({'base_task':base,'factor':factor,'total':len(cell),'candidate_equivalence_groups':n,'quarantined':sum(r['history_status']!='candidate' for r in cell)})
            if settings.get('sampling_unit','base_task_factor')=='base_task_factor' and n<settings['dev_per_cell']+settings['blind_per_cell']:short.append(cells[-1])
    if settings.get('sampling_unit')=='suite_factor':
        for factor in settings['confirmatory_factors']:
            n=len({r['equivalence_group'] for r in rows if r['factor']==factor and r['history_status']=='candidate'})
            if n<settings['dev_per_factor']+settings['blind_per_factor']:short.append({'suite':suite,'factor':factor,'candidate_equivalence_groups':n,'required':settings['dev_per_factor']+settings['blind_per_factor']})
    # Predeclare source index roles; index0 remains benchmark target default. Not a claim of fresh historical Clean holdout.
    source_plan=[];source_cases=[]
    for base in sorted(states):
        order=sorted(range(1,len(states[base])),key=lambda i:digest([settings['sampling_seed'],base,'source_initialization',i]))
        for role,indices in [('source_fit',order[:10]),('source_calibration',order[10:20]),('clean_holdout',order[20:30])]:
            for ordinal,index in enumerate(indices):
                d=seen[base];row={'case_id':digest([commit,suite,base,role,index])[:24],'commit':commit,'suite':suite,'base_task':base,'logical_task_name':base,
                    'role':role,'factor':'clean','runtime_params':resolve_task(base+'_light_1','lighting')['runtime_params']|{'light_asset_id':0},
                    'bddl':d['clean_bddl'],'resolved_bddl_hash':d['clean_bddl_sha256'],'clean_bddl':d['clean_bddl'],'clean_bddl_sha256':d['clean_bddl_sha256'],
                    'init_state_file':d['init_file'],'init_state_file_sha256':d['init_file_sha256'],'init_state_index':index,'raw_init_hash':array_hash(states[base][index]),
                    'physical_init_hash':None,'pairing_status':'source_clean','pair_group_id':digest([suite,base,array_hash(states[base][index])]),
                    'historical_clean_freshness':'unverified_not_final_holdout','planned_smoke':role=='source_fit' and ordinal<settings['smoke_per_task']}
                source_plan.append(row)
                if row['planned_smoke']:source_cases.append(row)
    assets=src/'libero/libero/assets'
    blockers=[]
    if not settings['history_complete']:blockers.append('historical coverage not certified; original server logs unresolved')
    if short:blockers.append('insufficient independently unaccessed groups under declared sampling unit')
    if not assets.is_dir():blockers.append('simulator assets absent locally')
    blockers+=['Clean compatibility smoke not yet run','independent Clean Holdout historical freshness unresolved']
    out={'schema':'static_visual_v2_p0','status':'P0_AUDIT_ONLY','commit':commit,'settings':settings,'settings_sha256':file_hash(args.settings),
         'source_fingerprints':source,'runtime_fingerprint':runtime_fingerprint(),'checkpoint_sha256':file_hash(args.checkpoint),'training_coverage':'unknown',
         'runtime_patch':subprocess.check_output(['git','-C',str(src),'diff','--','libero/libero/envs/env_wrapper.py'],text=True),
         'classification_counts':{s:len(r) for s,r in cls.items()},'classification_id_convention':'1-based; evaluator list index=classification_id-1; name is identity',
         'candidate_tasks':sorted(states),'registry':rows,'source_plan':source_plan,'history_scan':scan,'parse_errors':errors,
         'cells':cells,'shortfalls':short,'blockers':blockers,'assets_present':assets.is_dir(),
         'code_sha256':{p:file_hash(ROOT/p) for p in ('scripts/static_visual_v2.py','dacsc/static_visual_v2.py','scripts/static_visual_v2_batch.py')}}
    args.out.mkdir(parents=True,exist_ok=False);write(args.out/'p0_audit.json',out)
    smoke={'schema':'static_visual_v2_manifest','kind':'dev','status':'P0_SMOKE_ONLY','audit_sha256':file_hash(args.out/'p0_audit.json'),'cases':source_cases,'note':'only predeclared source compatibility; no Target Dev authorized'}
    blind={'schema':'static_visual_v2_manifest','kind':'blind','status':'NOT_FROZEN_BLOCKED','cases':[],'blockers':blockers,'note':'empty is not a 400-instance frozen Blind manifest'}
    write(args.out/'dev_manifest.json',smoke);write(args.out/'blind_manifest.json',blind)
    write(args.out/'manifest_hashes.json',{p:file_hash(args.out/p) for p in ('p0_audit.json','dev_manifest.json','blind_manifest.json')})
    print(json.dumps({'tasks':len(states),'visual_rows':len(rows),'parse_errors':len(errors),'smoke_cases':len(source_cases),'shortfall_cells':len(short),'blockers':blockers},ensure_ascii=False,indent=2))


def freeze(args):
    audit_path=args.audit/'p0_audit.json';a=json.loads(audit_path.read_text());s=a['settings']
    if a['parse_errors']:raise ValueError('unresolved registry errors')
    if not s['history_complete']:raise PermissionError('history completeness not certified; cannot freeze Blind')
    if not a['assets_present']:raise PermissionError('asset audit incomplete')
    if a['shortfalls']:raise ValueError('insufficient per-cell capacity; do not silently change quotas')
    review=json.loads(args.review.read_text())
    if review.get('audit_sha256')!=file_hash(audit_path) or not review.get('independent_clean_holdout_verified'):raise PermissionError('missing same-audit Clean holdout certification')
    records=[json.loads(p.read_text()) for p in sorted(args.results.glob('*.json'))]
    for base in a['candidate_tasks']:
        rr=[r for r in records if r.get('base_task')==base and r.get('role')=='source_fit']
        expected={r['case_id'] for r in a['source_plan'] if r['base_task']==base and r['planned_smoke']}
        if len(rr)!=5 or {r['case_id'] for r in rr}!=expected or any(r.get('status')!='completed' for r in rr) or not any(r['success'] for r in rr):raise PermissionError('unresolved Clean compatibility: '+base)
        if any(r['audit_sha256']!=file_hash(audit_path) for r in rr):raise PermissionError('compatibility from another audit')
    selected=deterministic_partition(a['registry'],s['dev_per_cell'],s['blind_per_cell'],s['sampling_seed'])
    if len(selected['dev'])!=len(a['candidate_tasks'])*4*s['dev_per_cell'] or len(selected['blind'])!=len(a['candidate_tasks'])*4*s['blind_per_cell']:raise ValueError('missing cells')
    args.out.mkdir(parents=True,exist_ok=False)
    for kind in ('dev','blind'):
        cases=[];clean={}
        for row in selected[kind]:
            c=dict(row,role='target_dev' if kind=='dev' else 'target_blind');cases.append(c)
            pair=dict(c,case_id='clean_'+c['pair_group_id'][:24],role='dev_clean' if kind=='dev' else 'blind_clean',logical_task_name=c['base_task'],factor='clean',bddl=c['clean_bddl'],resolved_bddl_hash=c['clean_bddl_sha256'],runtime_params={'view':[0,0,100,0,0],'robot_initial_pose':0,'noise':0,'sensor_camera':'agentview'})
            clean[pair['case_id']]=pair
        if kind=='dev':cases+=sorted(clean.values(),key=lambda r:r['case_id'])
        if kind=='blind':cases+=[dict(r,historical_clean_freshness='verified_by_release_review') for r in a['source_plan'] if r['role']=='clean_holdout']
        write(args.out/(kind+'_manifest.json'),{'schema':'static_visual_v2_manifest','kind':kind,'status':'FROZEN','audit_sha256':file_hash(audit_path),'cases':cases,'review_sha256':file_hash(args.review),'clean_control_policy':'Dev uses paired twins; Blind uses separate independently certified Clean Holdout, never rebrands default-index0 Dev twins'})
    write(args.out/'manifest_hashes.json',{p:file_hash(args.out/p) for p in ('dev_manifest.json','blind_manifest.json')})


def run(args):
    # Deny BEFORE torch, simulator, asset, checkpoint or model access.
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('kind')=='blind':raise PermissionError('Blind manifest locked before case lookup or model access')
    case=next(c for c in manifest['cases'] if c['case_id']==args.case_id)
    hashes=json.loads((args.manifest.parent/'manifest_hashes.json').read_text())
    authorize(manifest,case,'render/inference/rollout',hashes.get(args.manifest.name),args.manifest)
    audit_path=args.audit/'p0_audit.json'
    if file_hash(audit_path)!=manifest['audit_sha256']:raise PermissionError('audit changed')
    a=json.loads(audit_path.read_text());s=a['settings'];src=args.libero_src
    if any(file_hash(ROOT/p)!=h for p,h in a['code_sha256'].items()):raise ValueError('audited execution code changed; regenerate P0 receipt')
    if pinned_sources(src)!=a['source_fingerprints']:raise ValueError('registry/wrapper code drift')
    if a.get('runtime_fingerprint')!=runtime_fingerprint():raise ValueError('runtime/dependency XML changed; regenerate audit and rerun compatibility')
    if file_hash(args.checkpoint)!=a['checkpoint_sha256']:raise ValueError('checkpoint drift')
    if not (src/'libero/libero/assets').is_dir():raise FileNotFoundError('simulator assets required; static source checkout is not runtime-ready')
    for field,path in [('resolved_bddl_hash',case['bddl']),('init_state_file_sha256',case['init_state_file'])]:
        if file_hash(src/path)!=case[field]:raise ValueError('case asset drift')
    import os,platform,traceback
    if os.environ.get('MUJOCO_GL')!='osmesa' or os.environ.get('PYOPENGL_PLATFORM')!='osmesa':raise ValueError('OSMesa required')
    sys.path[:0]=[str(src),str(args.corp_src),str(args.corp_src/'scripts'),str(ROOT/'scripts')]
    import torch
    from corp import load_policy
    from evaluate_libero import policy_observation,DUMMY_ACTION,SETTLING_STEPS,state_from_observation
    from collect_interaction_replay import official_plan
    from dacsc.interaction_records import step_flags
    from dacsc.state_pairing import snapshot,model_compatible
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.env_wrapper as wrapper
    import libero.libero.envs.bddl_utils as bddl
    if SETTLING_STEPS!=s['settling_steps']:raise ValueError('settling protocol mismatch')
    global _POLICY_CACHE
    key=(str(args.checkpoint.resolve()),a['checkpoint_sha256'])
    cache=globals().setdefault('_POLICY_CACHE',{})
    if key not in cache:
        loaded=load_policy(args.checkpoint,device='cuda:0');loaded.eval();loaded.requires_grad_(False);cache[key]=loaded
    policy=cache[key]
    if (policy.action_generator.action_horizon,policy.action_generator.action_dim,policy.action_generator.num_sample_steps)!=(12,7,10):raise ValueError('policy shape mismatch')
    initial=torch.load(src/case['init_state_file'],map_location='cpu',weights_only=False)[case['init_state_index']]
    if array_hash(np.asarray(initial))!=case['raw_init_hash']:raise ValueError('initialization differs')
    seeds={stream:seed_for(s['sampling_seed'],case['pair_group_id'],0,stream) for stream in ('environment_seed','sensor_noise_seed','policy_flow_seed','method_augmentation_seed')}
    envrng=EnvironmentRNG(seeds['environment_seed']);gen=torch.Generator(device='cuda:0');gen.manual_seed(seeds['policy_flow_seed'])
    logical=src/'libero/libero/bddl_files'/case['suite']/(case['logical_task_name']+'.bddl')
    language=bddl.get_problem_info(str(src/case['clean_bddl']))['language_instruction']
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/(case['case_id']+'.json')
    if output.exists():raise FileExistsError(output)
    ledger=args.manifest.parent/'access_ledger.jsonl';append_event(ledger,{'case_id':case['case_id'],'role':case['role'],'action':'rollout_start','manifest_sha256':file_hash(args.manifest),'checkpoint':a['checkpoint_sha256']})
    env=None;clean=None;result={};began=time.monotonic()
    try:
        with envrng.use():
            env=OffScreenRenderEnv(bddl_file_name=str(logical),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            noise=int(env.noise)
            if noise!=case['runtime_params']['noise']:raise ValueError('sensor parser parity failed')
            env.noise=0 # Disable native corruption BEFORE reset so it cannot consume environment RNG.
            env.seed(seeds['environment_seed']);env.reset();obs=env.set_init_state(initial)
        # Verify post-reset physical pairing using a Clean twin; twin never drives policy.
        with EnvironmentRNG(seeds['environment_seed']).use():
            clean=OffScreenRenderEnv(bddl_file_name=str(src/case['clean_bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            clean.seed(seeds['environment_seed']);clean.reset();clean.set_init_state(initial)
        def state_record(e):
            return {k:np.asarray(getattr(e.sim.data,k)).tolist() for k in ('qpos','qvel','act','time')}
        rawpair=state_record(env)==state_record(clean)
        compatible=model_compatible(snapshot(env),snapshot(clean),strict_bodies=True,strict_geometry=True,strict_physics=True)['pass']
        result['physical_pair']={'raw_state_equal':rawpair,'model_physics_compatible':compatible,'physical_init_hash':digest(state_record(env)),'scope':'listed simulator fields and static physics; not all hidden state'}
        if not rawpair or not compatible:raise ValueError('unpaired Clean/Shift; retained as exception, never replaced')
        clean.close();clean=None
        for step in range(s['settling_steps']):
            with envrng.use():obs,flags=step_flags(env.step(DUMMY_ACTION),env)
            if any(flags[k] for k in ('success','done','terminated','truncated')):raise ValueError('ended during settling')
        obs=sensor_transform(obs,noise,seeds['sensor_noise_seed'],0,wrapper)
        executed=[];chunks=[];steps=0;success=False
        while steps<s['max_steps']:
            z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=gen)
            plan,parity=official_plan(policy,policy_observation(obs,language,policy),z0,audit=(steps==0))
            begin=steps
            for action in np.clip(plan,-1,1):
                with envrng.use():obs,flags=step_flags(env.step(action),env)
                steps+=1;executed.append(action.copy());obs=sensor_transform(obs,noise,seeds['sensor_noise_seed'],steps,wrapper)
                success=flags['success']
                if any(flags[k] for k in ('success','done','terminated','truncated')) or steps>=s['max_steps']:break
            chunks.append({'start':begin,'end':steps,'z0_hash':array_hash(z0.cpu().numpy()),'official_parity':parity})
            if any(flags[k] for k in ('success','done','terminated','truncated')):break
        result.update(status='completed',success=success,steps=steps,chunks=chunks,action_sha256=array_hash(np.asarray(executed)),
            terminal_flags=flags,collector_truncated=steps>=s['max_steps'] and not success)
    except Exception as e:result.update(status='exception',success=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        if clean is not None:clean.close()
        if env is not None:env.close()
    result.update(case_id=case['case_id'],base_task=case['base_task'],factor=case['factor'],pair_group_id=case['pair_group_id'],role=case['role'],seeds=seeds,
        init_state_index=case['init_state_index'],audit_sha256=file_hash(audit_path),manifest_sha256=file_hash(args.manifest),checkpoint_sha256=a['checkpoint_sha256'],
        code_sha256={p:file_hash(ROOT/p) for p in ('scripts/static_visual_v2.py','dacsc/static_visual_v2.py')},wall_seconds=time.monotonic()-began,
        method='frozen',new_model_training=False,blind_accessed=False,
        runtime_versions={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'mujoco':__import__('mujoco').__version__,'robosuite':__import__('robosuite').__version__})
    write(output,result);append_event(ledger,{'case_id':case['case_id'],'role':case['role'],'action':'rollout_end','status':result['status'],'result_sha256':file_hash(output)})
    print(json.dumps(result,ensure_ascii=False))
    if result['status']!='completed':raise SystemExit(2)


def summarize(args):
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('kind')=='blind':raise PermissionError('Dev summary cannot open Blind')
    expected={c['case_id']:c for c in manifest['cases'] if c['role']=='target_dev'}
    records=[]
    for p in sorted(args.results.glob('*.json')):
        r=json.loads(p.read_text())
        if 'case_id' in r:records.append(r)
    if any(r['role'] in PROTECTED for r in records):raise PermissionError('Dev summary must not read Blind records')
    if len({r['case_id'] for r in records})!=len(records):raise ValueError('duplicate run results')
    lookup={r['case_id']:r for r in records};clean={r['pair_group_id']:r for r in records if r['role']=='dev_clean'};cells={}
    for case_id,case in expected.items():
        key=case['base_task']+'|'+case['factor'];c=cells.setdefault(key,{'expected_n':0,'completed_n':0,'paired':0,'clean_success_shift_fail':0,'clean_fail_shift_success':0,'both_success':0,'both_fail':0,'exceptions':0,'missing':0})
        c['expected_n']+=1;r=lookup.get(case_id)
        if r is None:c['missing']+=1;continue
        if r['manifest_sha256']!=file_hash(args.manifest):raise ValueError('results from different manifest')
        if r['status']!='completed':c['exceptions']+=1;continue
        c['completed_n']+=1;ref=clean.get(r['pair_group_id'])
        if ref is None or ref['status']!='completed':continue
        if ref['manifest_sha256']!=file_hash(args.manifest):raise ValueError('Clean control from different manifest')
        c['paired']+=1
        term='both_success' if r['success'] and ref['success'] else 'both_fail' if not r['success'] and not ref['success'] else 'clean_success_shift_fail' if ref['success'] else 'clean_fail_shift_success'
        c[term]+=1
    complete=bool(cells) and all(c['paired']==c['expected_n'] for c in cells.values())
    macro=float(np.mean([(c['clean_fail_shift_success']-c['clean_success_shift_fail'])/c['paired'] for c in cells.values()])) if complete else None
    factors={}
    for key,c in cells.items():
        factor=key.split('|')[-1];f=factors.setdefault(factor,{k:0 for k in c})
        for k,v in c.items():f[k]+=v
    factor_complete=bool(factors) and all(c['paired']==c['expected_n'] for c in factors.values())
    suite_macro=float(np.mean([(c['clean_fail_shift_success']-c['clean_success_shift_fail'])/c['paired'] for c in factors.values()])) if factor_complete else None
    write(args.out,{'factors':factors,'suite_factor_macro_shift_minus_clean':suite_macro,'cells':cells,'role':'dev_only','complete':complete,'frozen_shift_minus_clean_macro':macro,'method_vs_frozen_macro':None,
        'note':'no TTA method evaluated; exceptions/missing pairs retained, no complete macro when incomplete; n=3/cell exploratory'})


def freeze_suite(args):
    audit_path=args.audit/'p0_audit.json';a=json.loads(audit_path.read_text());s=a['settings']
    if s.get('sampling_unit')!='suite_factor':raise ValueError('suite/factor settings required')
    if a['parse_errors'] or not a['assets_present']:raise ValueError('registry/assets audit incomplete')
    records=[]
    for p in sorted(args.results.glob('*.json')):
        r=json.loads(p.read_text())
        if r.get('role')=='source_fit':records.append(r)
    expected={r['case_id'] for r in a['source_plan'] if r['planned_smoke']}
    if len(records)!=len(expected) or {r['case_id'] for r in records}!=expected:raise PermissionError('all predeclared Clean smoke results required')
    if any(r['status']!='completed' or r['audit_sha256']!=file_hash(audit_path) for r in records):raise PermissionError('unresolved execution errors or mismatched source protocol')
    for task in a['candidate_tasks']:
        if not any(r['success'] for r in records if r['base_task']==task):raise PermissionError('0/5 Clean compatibility unresolved: '+task)
    if not s.get('historical_dev_reuse_authorized'):raise PermissionError('development reuse not declared')
    selected=deterministic_dev_sample(a['registry'],s['dev_per_factor'],s['sampling_seed'],s['dev_factors'])
    args.out.mkdir(parents=True,exist_ok=False);cases=[];clean={}
    for r in selected:
        c=dict(r,role='target_dev',freshness='historically_accessed_development_reuse');cases.append(c)
        pair=dict(c,case_id='clean_'+c['pair_group_id'][:24],role='dev_clean',logical_task_name=c['base_task'],factor='clean',bddl=c['clean_bddl'],resolved_bddl_hash=c['clean_bddl_sha256'],runtime_params={'view':[0,0,100,0,0],'robot_initial_pose':0,'noise':0,'sensor_camera':'agentview'})
        clean[pair['case_id']]=pair
    cases+=sorted(clean.values(),key=lambda r:r['case_id'])
    write(args.out/'dev_manifest.json',{'schema':'static_visual_v2_manifest','kind':'dev','status':'FROZEN','audit_sha256':file_hash(audit_path),'cases':cases,'sampling_unit':'suite_factor','dev_per_factor':s['dev_per_factor'],'freshness':'new deterministic selection of historically accessed benchmark; not fresh data','Blind_access':False})
    write(args.out/'blind_manifest.json',{'schema':'static_visual_v2_manifest','kind':'blind','status':'NO_UNACCESSED_OFFICIAL_POOL','cases':[],'requested_per_factor':s['blind_per_factor'],'requested_factors':s['confirmatory_factors'],'reason':'historical full10030 benchmark access; cannot reserve fresh300 by resampling','Blind_access':False})
    write(args.out/'manifest_hashes.json',{p:file_hash(args.out/p) for p in ('dev_manifest.json','blind_manifest.json')})
    print(json.dumps({'dev_shift':len(selected),'dev_clean':len(clean),'blind_reserved':0,'status':'development-only official pool'}))


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('audit');a.add_argument('--libero-src',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--settings',type=Path,required=True);a.add_argument('--history-root',type=Path,action='append',required=True);a.add_argument('--out',type=Path,required=True)
    f=sp.add_parser('freeze');f.add_argument('--audit',type=Path,required=True);f.add_argument('--results',type=Path,required=True);f.add_argument('--review',type=Path,required=True);f.add_argument('--out',type=Path,required=True)
    fs=sp.add_parser('freeze-suite');fs.add_argument('--audit',type=Path,required=True);fs.add_argument('--results',type=Path,required=True);fs.add_argument('--out',type=Path,required=True)
    r=sp.add_parser('run');r.add_argument('--audit',type=Path,required=True);r.add_argument('--manifest',type=Path,required=True);r.add_argument('--case-id',required=True);r.add_argument('--libero-src',type=Path,required=True);r.add_argument('--corp-src',type=Path,required=True);r.add_argument('--checkpoint',type=Path,required=True);r.add_argument('--out',type=Path,required=True)
    s=sp.add_parser('summarize');s.add_argument('--manifest',type=Path,required=True);s.add_argument('--results',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    args=p.parse_args();{'audit':audit,'freeze':freeze,'freeze-suite':freeze_suite,'run':run,'summarize':summarize}[args.cmd](args)
if __name__=='__main__':main()
