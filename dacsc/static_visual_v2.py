"""Minimal immutable IDs, wrapper parsing, RNG isolation and role locks for V2."""
from __future__ import annotations
import contextlib,hashlib,json,random,re
from pathlib import Path
import numpy as np

FACTORS={'Light Conditions':'lighting','Background Textures':'background','Sensor Noise':'sensor_noise','Camera Viewpoints':'camera'}
PROTECTED={'target_blind','blind_clean','clean_holdout'}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def seed_for(*parts):return int(digest(parts)[:8],16)


def resolve_task(name,factor):
    """Mirror pinned ControlEnv filename parser; never infer wrapper settings from BDDL hash."""
    runtime={'view':[0,0,100,0,0],'robot_initial_pose':0,'noise':0,'sensor_camera':'agentview'}
    if '_view_' in name:
        match=re.fullmatch(r'(.+)_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)(?:_noise_(\d+))?',name)
        if not match:raise ValueError('unsupported runtime filename: '+name)
        base=match[1];runtime['view']=[int(match[i]) for i in range(2,7)]
        runtime['robot_initial_pose']=int(match[7]);runtime['noise']=int(match[8] or 0)
        if runtime['robot_initial_pose']!=0:raise ValueError('robot initial-state shift outside scope')
        if factor=='sensor_noise' and (runtime['view']!=[0,0,100,0,0] or not 1<=runtime['noise']<=50):raise ValueError('mixed or invalid sensor perturbation')
        if factor=='camera' and (runtime['noise']!=0 or runtime['view']==[0,0,100,0,0]):raise ValueError('mixed or empty camera perturbation')
        if factor not in ('sensor_noise','camera'):raise ValueError('factor/runtime mismatch')
        resolved=base+'.bddl';sub=('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')[(runtime['noise']-1)//10] if runtime['noise'] else 'camera_joint_parameters'
    else:
        match=re.fullmatch(r'(.+)_(light|table|tb)_(\d+)',name)
        if not match:raise ValueError('unsupported static task suffix: '+name)
        base=match[1];tag=match[2]
        if factor!=('lighting' if tag=='light' else 'background'):raise ValueError('factor/suffix mismatch')
        resolved=name+'.bddl';runtime[tag+'_asset_id']=int(match[3]);sub={'light':'light_asset','table':'table_texture','tb':'scene_texture'}[tag]
    return {'base_task':base,'resolved_bddl':resolved,'runtime_params':runtime,
            'init_file':base+'.pruned_init','sub_dimension':sub,'sub_dimension_source':'parsed wrapper/asset family; not an invented official label'}


def equivalence_key(suite,base,bddl_hash,runtime,init_hash):
    # Exact asset selectors already resolved in BDDL; cosmetic alias number is not a runtime effect.
    actual={k:v for k,v in runtime.items() if not k.endswith('_asset_id')}
    return digest([suite,base,bddl_hash,actual,init_hash])


def authorize(manifest,case,action,expected_sha=None,manifest_path=None):
    if case.get('role') in PROTECTED or manifest.get('kind')=='blind':raise PermissionError('Blind/Clean Holdout locked: '+action)
    if case.get('role') not in ('source_fit','target_dev','dev_clean'):raise PermissionError('role is not runnable')
    if manifest.get('status') not in ('P0_SMOKE_ONLY','FROZEN'):raise PermissionError('manifest not executable')
    if manifest['status']=='P0_SMOKE_ONLY' and case['role']!='source_fit':raise PermissionError('P0 can run Source Fit only')
    if manifest_path is not None and (not expected_sha or file_hash(manifest_path)!=expected_sha):raise PermissionError('manifest fingerprint mismatch')
    if case not in manifest.get('cases',[]):raise PermissionError('case absent from manifest')


@contextlib.contextmanager
def isolated_seed(seed):
    n,p=np.random.get_state(),random.getstate()
    np.random.seed(seed);random.seed(seed)
    try:yield
    finally:np.random.set_state(n);random.setstate(p)


class EnvironmentRNG:
    def __init__(self,seed):
        self.n=np.random.RandomState(seed).get_state();self.p=random.Random(seed).getstate()
    @contextlib.contextmanager
    def use(self):
        n,p=np.random.get_state(),random.getstate();np.random.set_state(self.n);random.setstate(self.p)
        try:yield
        finally:self.n=np.random.get_state();self.p=random.getstate();np.random.set_state(n);random.setstate(p)


def sensor_transform(obs,code,seed,step,wrapper):
    if code==0:return obs
    from PIL import Image
    out=dict(obs);pixels=np.asarray(obs['agentview_image']).copy()
    if pixels.dtype!=np.uint8:pixels=(pixels*255).astype(np.uint8)
    image=Image.fromarray(pixels);family=(code-1)//10;severity=(code-1)%10+1
    funcs=[wrapper.motion_blur,wrapper.gaussian_blur,wrapper.zoom_blur,wrapper.fog,wrapper.glass_blur]
    with isolated_seed(seed_for(seed,step,'agentview')):result=funcs[family](image,severity=severity)
    if family!=0:result=result.astype(np.uint8)
    out['agentview_image']=result
    return out


def append_event(path,record):
    import fcntl,os,time
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(canonical(dict(timestamp_unix=time.time(),**record))+'\n');f.flush();os.fsync(f.fileno());fcntl.flock(f,fcntl.LOCK_UN)


def deterministic_partition(rows,dev,blind,seed):
    if min(dev,blind)<=0:raise ValueError('positive quotas required')
    cells={}
    for r in rows:
        if r.get('history_status')!='candidate':continue
        groups=cells.setdefault((r['base_task'],r['factor']),{})
        previous=groups.get(r['equivalence_group'])
        if previous is None or r['logical_task_name']<previous['logical_task_name']:groups[r['equivalence_group']]=r
    result={'dev':[],'blind':[]};shortfalls=[]
    for cell,groups in sorted(cells.items()):
        ordered=sorted(groups.values(),key=lambda r:(digest([seed,*cell,r['equivalence_group']]),r['logical_task_name']))
        if len(ordered)<dev+blind:shortfalls.append({'cell':cell,'available':len(ordered),'required':dev+blind});continue
        result['blind']+=ordered[:blind];result['dev']+=ordered[blind:blind+dev]
    if shortfalls:raise ValueError('insufficient independent groups: '+canonical(shortfalls))
    return result
