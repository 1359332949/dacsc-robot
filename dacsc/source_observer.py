"""Source Observer V1: fit-only preprocessing and immutable numpy inference."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .source_interactions import role_ids, validate_source_episode
from .source_provenance import require_same_collection

MODELS=('M0','M1','M2')
CONTROLS=('zero_delta','fit_mean_next_minus_current')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_weights(ids):
    ids=np.asarray(ids);states=np.unique(ids)
    return np.asarray([1./len(states)/np.sum(ids==s) for s in ids])


@dataclass
class Data:
    role: str
    states: np.ndarray
    q: np.ndarray
    actions: np.ndarray
    current: dict
    following: dict
    boundaries: dict
    boundary_states: np.ndarray
    timing: list
    provenance: list
    excluded_partial: int


def load_role(root, split_path, role, views):
    if role not in ('source_fit','source_validation'):raise ValueError('source-only data role')
    root=Path(root);split=json.loads(Path(split_path).read_text());expected=sorted(role_ids(split,role))
    state=[];q=[];act=[];cur={v:[] for v in views};nxt={v:[] for v in views};bounds={v:[] for v in views}
    bs=[];timing=[];prov=[];seen=[];partial=0;collection=None
    # Deliberately do not open the other role's metadata or arrays.
    for mp in sorted(root.glob(f'*/{role}_state*_seed*/metadata.json')):
        m=json.loads(mp.read_text());npz=mp.parent/'episode.npz'
        collection=require_same_collection(m,collection)
        if m['role']!=role or m['split_sha256']!=sha(split_path):raise ValueError('role/split mismatch')
        if m['checkpoint_sha256']!=split['policy']['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        if sha(npz)!=m['npz_sha256']:raise ValueError('data hash mismatch')
        if m['seed']!=2014:raise ValueError('V1 freezes source seed 2014')
        with np.load(npz,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        validate_source_episode(a,m,split)
        ii=np.flatnonzero(a['fixed_horizon_eligible'])
        if not len(ii):raise ValueError('source state has no full chunks')
        if m['state_id'] in seen:raise ValueError('duplicate state/seed')
        seen.append(m['state_id']);partial+=int((~a['fixed_horizon_eligible']).sum())
        bb=np.unique(np.r_[ii,ii+1]);bs.extend([m['state_id']]*len(bb));state.extend([m['state_id']]*len(ii))
        q.append(a['state8'][ii].astype(float))
        act.append(a['actions_executed'][ii].reshape(-1,3,4,7).mean(axis=2).reshape(-1,21).astype(float))
        for v in views:
            pooled=a['tokens_'+v].astype(float).mean(axis=1)
            cur[v].append(pooled[ii]);nxt[v].append(pooled[ii+1]);bounds[v].append(pooled[bb])
        for j in ii:
            timing.append({'state_id':m['state_id'],'start_step':int(a['boundary_step'][j]),
                'end_step':int(a['boundary_step'][j+1]),'remaining_steps':int(m['steps']-a['boundary_step'][j+1]),
                'episode_steps':m['steps'],'episode_success':m['success'],'chunk':int(j)})
        prov.append({'state_id':m['state_id'],'metadata_path':str(mp),'metadata_sha256':sha(mp),'npz_sha256':sha(npz),
                     'full_chunks':len(ii),'selected_boundaries':len(bb),'collection_signature':collection})
    if sorted(seen)!=expected:raise ValueError(f'incomplete role {role}: {sorted(seen)}')
    return Data(role,np.asarray(state),np.concatenate(q),np.concatenate(act),
                {v:np.concatenate(cur[v]) for v in views},{v:np.concatenate(nxt[v]) for v in views},
                {v:np.concatenate(bounds[v]) for v in views},np.asarray(bs),timing,prov,partial)


def weighted_pca(x,weights,rank):
    if rank>min(x.shape):raise ValueError('insufficient PCA rows')
    mu=np.sum(weights[:,None]*x,axis=0)
    _,s,vt=np.linalg.svd((x-mu)*np.sqrt(weights[:,None]),full_matrices=False)
    p=vt[:rank].copy()
    signs=np.sign(p[np.arange(rank),np.argmax(np.abs(p),axis=1)]);p*=signs[:,None]
    return mu,p,float(np.sum(s[:rank]**2)/np.sum(s**2))


def design(kind,z,q,actions):
    if kind=='M0':return np.c_[q,actions]
    if kind=='M1':return np.c_[z,q,actions]
    if kind=='M2':return z.copy()
    raise ValueError(kind)


def fit(train,config):
    if train.role!='source_fit':raise ValueError('fit refuses non-Fit data')
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    weights=state_weights(train.states);bw=state_weights(train.boundary_states)
    arrays={};summary={}
    for view in config['views']:
        mu,p,capture=weighted_pca(train.boundaries[view],bw,config['pca_rank'])
        z=(train.current[view]-mu)@p.T;zn=(train.following[view]-mu)@p.T;delta=zn-z
        ym=np.sum(weights[:,None]*delta,axis=0)
        raw_scale=np.sqrt(np.sum(weights[:,None]*(delta-ym)**2,axis=0))
        floor=max(config['target_scale_floor_absolute'],config['target_scale_floor_median_fraction']*float(np.median(raw_scale)))
        scale=np.maximum(raw_scale,floor);target=delta/scale
        arrays[view+'_mu']=mu;arrays[view+'_pca']=p;arrays[view+'_target_scale']=scale
        arrays[view+'_mean_next']=np.sum(weights[:,None]*zn,axis=0)
        yraw=train.following[view]-train.current[view]
        summary[view]={'pca_boundary_variance_capture':capture,
            'fit_delta_energy_capture':float(np.sum(weights[:,None]*delta**2)/np.sum(weights[:,None]*yraw**2)),
            'target_raw_scale':raw_scale.tolist(),'target_scale':scale.tolist(),'target_scale_floor':floor,'models':{}}
        for kind in MODELS:
            x=design(kind,z,train.q,train.actions)
            scaler=StandardScaler().fit(x,sample_weight=weights)
            sd=np.maximum(scaler.scale_,config['feature_scale_floor']);xs=(x-scaler.mean_)/sd
            reg=Ridge(alpha=config['ridge_alpha'],fit_intercept=True,solver='svd').fit(xs,target,sample_weight=weights)
            prefix=view+'_'+kind
            for k,value in [('x_mu',scaler.mean_),('x_scale',sd),('coef',reg.coef_.T),('intercept',reg.intercept_)]:arrays[prefix+'_'+k]=value
            summary[view]['models'][kind]={'input_dim':x.shape[1],'output_dim':target.shape[1],
                'parameter_count':int(reg.coef_.size+reg.intercept_.size)}
    arrays['fit_row_weights']=weights;arrays['fit_states']=train.states.copy()
    return arrays,summary


def predict(arrays,view,kind,current,q,actions):
    z=(current-arrays[view+'_mu'])@arrays[view+'_pca'].T
    if kind=='zero_delta':return np.zeros_like(z)
    if kind=='fit_mean_next_minus_current':return arrays[view+'_mean_next'][None,:]-z
    prefix=view+'_'+kind;x=design(kind,z,q,actions)
    y=((x-arrays[prefix+'_x_mu'])/arrays[prefix+'_x_scale'])@arrays[prefix+'_coef']+arrays[prefix+'_intercept']
    return y*arrays[view+'_target_scale']


def state_metrics(data,arrays,view,prediction,config,mask=None):
    if mask is None:mask=np.ones(len(data.states),bool)
    delta_raw=data.following[view]-data.current[view]
    delta=delta_raw@arrays[view+'_pca'].T
    scale=arrays[view+'_target_scale'];target=delta/scale;pred=prediction/scale
    err=pred-target;rawerr=prediction@arrays[view+'_pca']-delta_raw
    result={}
    for s in np.unique(data.states[mask]):
        take=mask&(data.states==s);t=target[take];p=pred[take]
        valid=(np.linalg.norm(t,axis=1)>config['cosine_min_true_scaled_norm'])&(np.linalg.norm(p,axis=1)>config['cosine_min_pred_norm'])
        cosine=(1-np.sum(t[valid]*p[valid],axis=1)/(np.linalg.norm(t[valid],axis=1)*np.linalg.norm(p[valid],axis=1))) if valid.any() else None
        result[str(int(s))]={'n':int(take.sum()),'rmse':float(np.sqrt(np.mean(err[take]**2))),
            'rmse_768':float(np.sqrt(np.mean(rawerr[take]**2))),
            'cosine_error':float(cosine.mean()) if cosine is not None else None,
            'cosine_valid_fraction':float(valid.mean()),'target_scaled_norm_mean':float(np.linalg.norm(t,axis=1).mean())}
    return result


def comparison(metrics,reference,config,bootstrap=True):
    keys=sorted(set(metrics)&set(reference),key=int)
    gain=np.array([1-metrics[k]['rmse']/max(reference[k]['rmse'],1e-12) for k in keys])
    if not len(gain):return {'n_states':0,'median_gain':None,'wins':0}
    out={'n_states':len(keys),'per_state_gain':dict(zip(keys,gain.tolist())),
         'median_gain':float(np.median(gain)),'mean_gain':float(gain.mean()),'wins':int((gain>0).sum())}
    if bootstrap:
        rng=np.random.default_rng(config['bootstrap_seed']);draw=rng.integers(0,len(gain),(config['bootstrap_resamples'],len(gain)))
        out['mean_gain_state_bootstrap_ci95']=np.quantile(gain[draw].mean(axis=1),[.025,.975]).tolist()
    return out


def gate(view_report,config):
    c=config['continue_gate_each_view'];checks={}
    for ref in c['comparisons']:
        x=view_report['comparisons'][ref]
        checks[ref+'_median_gain_ge_5pct']=x['median_gain']>=c['minimum_median_relative_rmse_gain']
        checks[ref+'_wins_ge_8']=x['wins']>=c['minimum_state_wins']
        checks[ref+'_CI_lower_positive']=x['mean_gain_state_bootstrap_ci95'][0]>0
        for subset in ('common_early','nonterminal'):
            x=view_report['subsets'][subset]['comparisons'][ref]
            checks[ref+'_'+subset+'_positive']=x['n_states']==10 and x['median_gain']>0
    for control in CONTROLS:checks[control+'_median_gain_positive']=view_report['comparisons'][control]['median_gain']>0
    return {'pass':all(checks.values()),'checks':checks,'failed_checks':[k for k,v in checks.items() if not v]}
