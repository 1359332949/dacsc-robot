"""CPU-only historical timing audit; never invent risk onset from episode end."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np


def opportunities_before(event_step,horizon=12):
    return list(range(horizon,int(event_step),horizon))


def describe(values):
    return {'n':len(values),'min':min(values),'median':float(np.median(values)),'max':max(values)} if values else {'n':0}


def first_event(trajectory, predicate):
    return next((int(r['step']) for r in trajectory if predicate(r)),None)


def rotation_distance(a,b):
    def matrix(v):
        theta=np.linalg.norm(v)
        if theta<1e-12:return np.eye(3)
        x,y,z=v/theta;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        return np.eye(3)+np.sin(theta)*k+(1-np.cos(theta))*(k@k)
    return float(np.arccos(np.clip((np.trace(matrix(a).T@matrix(b))-1)/2,-1,1)))


def audit(root, legacy_path=None):
    files=['perturb_dev_clean_v2.json','perturb_dev_v2.json','gate1_first_chunk.json','failure_trajectories2.json']
    paths={f:(legacy_path if f=='failure_trajectories2.json' and legacy_path else root/f) for f in files}
    docs={f:json.loads(paths[f].read_text()) for f in files};episodes=[]
    for domain,file in [('clean',files[0]),('lighting',files[1])]:
        seen=set()
        for r in docs[file]['records']:
            if r.get('method')!='frozen':continue
            key=(r['init_index'],r['seed'])
            if key in seen:raise ValueError('duplicate episode')
            seen.add(key);end=int(r['steps']);success=bool(r['success'])
            episodes.append({'domain':domain,'state_id':key[0],'seed':key[1],'success':success,'endpoint_step':end,
                'endpoint_kind':'success' if success else ('timeout_censored' if r.get('hit_max_steps') else 'unspecified_non_success_end'),
                'complete_chunks':end//12,'partial_steps':end%12,
                'observer_boundaries_before_end':len(opportunities_before(end)),
                'first_action_deviation_step':None,'first_task_risk_step':None,'effective_recovery_opportunities':None})
    summary={d:{'success_end_steps':describe([r['endpoint_step'] for r in episodes if r['domain']==d and r['success']]),
                'non_success_end_steps':describe([r['endpoint_step'] for r in episodes if r['domain']==d and not r['success']]),
                'all_ended_by_first_chunk':sum(r['endpoint_step']<=12 for r in episodes if r['domain']==d)} for d in ('clean','lighting')}
    gate=docs[files[2]]['records'];core=[r for r in gate if r['clean_success'] and not r['shift_success']]
    categories={}
    for r in core:
        name=f"clean_{int(r['clean_continuation'])}_shift_{int(r['shift_recovery'])}"
        categories.setdefault(name,[]).append({'state_id':r['state_id'],'seed':r['seed'],'clean_end':r['clean_steps'],'shift_end':r['shift_steps']})
    raw=docs[files[3]]['records'];legacy=[];natural={}
    for r in raw:
        if r['mode']!='frozen':continue
        t=r['trajectory']
        if [s['step'] for s in t]!=list(range(1,len(t)+1)):raise ValueError('noncontiguous legacy steps')
        if len(t)!=r['steps']:raise ValueError('legacy step count')
        close=first_event(t,lambda s:s['clipped_action'][6]>0)
        contact=first_event(t,lambda s:bool(s.get('target_finger_contact',False)))
        legacy.append({'state_id':r['state_id'],'domain':r['domain'],'success':r['success'],'steps':r['steps'],
            'first_close_command':close,'first_target_contact':contact,'first_task_risk_step':None,
            'provenance':'legacy per-step diagnostic; seed/renderer/checkpoint hashes not persisted in artifact',
            'interpretation':'close/contact absence or timing is descriptive, not an established risk event'})
        natural[(r['state_id'],r['domain'])]=r
    divergence=[]
    for sid in sorted({k[0] for k in natural}):
        if (sid,'clean') not in natural or (sid,'shift') not in natural:continue
        c,s=natural[(sid,'clean')],natural[(sid,'shift')]
        ca=np.array([r['clipped_action'] for r in c['trajectory']]);sa=np.array([r['clipped_action'] for r in s['trajectory']]);n=min(len(ca),len(sa));ca,sa=ca[:n],sa[:n]
        xyz=np.linalg.norm(.05*(ca[:,:3]-sa[:,:3]),axis=1)*1000
        rot=np.degrees([rotation_distance(.5*x,.5*y) for x,y in zip(ca[:,3:6],sa[:,3:6])]);grip=np.abs(ca[:,6]-sa[:,6])
        def onset(a,threshold):
            hits=np.flatnonzero(a>threshold)
            if not len(hits):return None
            step=int(hits[0]+1)
            return {'step':step,'chunk_zero_based':(step-1)//12,'observer_boundaries_strictly_before':opportunities_before(step)}
        divergence.append({'state_id':sid,'clean_success':c['success'],'shift_success':s['success'],
            'xyz_command_mm':{str(x):onset(xyz,x) for x in (1,5,10)},
            'rotation_command_deg':{str(x):onset(rot,x) for x in (1,5,10)},'gripper_command_0.5':onset(grip,.5),
            'scope':'illustrative threshold sensitivity, not validated task-risk labels; independent trajectories diverge physically',
            'controller_scale_assumption':'current OSC_POSE mapping XYZ .05 m, rotvec .5 rad; historical artifact lacks saved controller provenance'})
    return {'schema':'historical_failure_timing_v1','gpu_used':False,'H':12,'E':12,'first_observer_available_step':12,
        'sources':{f:hashlib.sha256(paths[f].read_bytes()).hexdigest() for f in files},'legacy_input_path':str(paths['failure_trajectories2.json']),'baseline_summary':summary,'episodes':episodes,
        'gate1':{'core_n':len(core),'categories':categories,'baseline_max_steps':220,'continuation_max_steps':300,
            'shift_success_before_original_cutoff':[{'state_id':r['state_id'],'seed':r['seed'],'step':r['shift_steps']} for r in core if r['shift_recovery'] and r['shift_steps']<=220],
            'baseline_reproduction_warning':'Some same-Shift continuations succeed before step220 despite baseline failure; cannot explain this solely by longer cutoff or claim causal recovery.',
            'scope':'legacy privileged continuation at step12; no last-recoverable-time sweep or deployable recovery guarantee'},
        'legacy_events':legacy,'legacy_action_divergence':divergence,
        'missing':['first task-risk time for current baseline episodes','last recoverable time','effective replan opportunities after risk'],
        'decision':'nonzero nominal time budget exists, but useful intervention lead time remains unmeasured'}


def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,domain in zip(axes,('clean','lighting')):
        records=sorted((r for r in report['episodes'] if r['domain']==domain),key=lambda r:(r['state_id'],r['seed']))
        for i,r in enumerate(records):
            ax.scatter(r['endpoint_step'],i,marker='o' if r['success'] else '>',color='#167d9a' if r['success'] else '#d67524',s=28)
        ax.axvline(12,color='#555',ls='--',lw=1,label='first residual available: 12')
        ax.set(title=domain.title()+' recorded endpoints',xlabel='Executed step (not failure onset)',ylabel='State × seed episode',xlim=(0,235))
        ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Historical timing audit: circles = success; arrows = non-success timeout')
    fig.savefig(path,dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--plot',action='store_true');p.add_argument('--legacy',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d=audit(a.data,a.legacy);(a.out/'report.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    with (a.out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(d['episodes'][0]),lineterminator='\n');w.writeheader();w.writerows(d['episodes'])
    if a.plot:plot(d,a.out/'endpoints.png')
    print(json.dumps({k:d[k] for k in ('baseline_summary','gate1','legacy_action_divergence','missing')},indent=2))
