"""LOTO feature scoring/diagnosis and descriptive paired-cluster comparisons."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.global_shift_diag import fit_moments,distance,loto_stats
from dacsc.static_visual_v2 import file_hash
from dacsc.flow_geometry import auc


def concordance(rows,score):
    pairs=[float(score[i]>score[j])+.5*float(score[i]==score[j]) for i,r in enumerate(rows) if not r['baseline_success']
           for j,s in enumerate(rows) if s['baseline_success'] and r['base_task']==s['base_task']]
    return {'pairs':len(pairs),'rate':float(np.mean(pairs)) if pairs else None}


def bootstrap_auc(y,s,weights):
    f=weights[:,y];p=weights[:,~y];m=s[y,None]-s[None,~y]
    comp=(m>0)+.5*(m==0);den=f.sum(1)*p.sum(1)
    result=np.full(len(weights),np.nan);ok=den>0
    result[ok]=np.sum(f[ok]*(p[ok]@comp.T),axis=1)/den[ok]
    return result


def finite_corr(x,y):
    if np.std(x)<1e-15 or np.std(y)<1e-15:return None
    c=float(spearmanr(x,y).statistic)
    return c if np.isfinite(c) else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());paths=sorted(a.features.glob('*.json'));rows=[json.loads(p.read_text()) for p in paths]
    if len(rows)!=100 or {r['case_id'] for r in rows}!={c['case_id'] for c in m['cases']} or any(r['status']!='completed' for r in rows):raise ValueError('all100 completed features required')
    if any(r['manifest_sha256']!=file_hash(a.manifest) for r in rows):raise ValueError('manifest drift')
    clean={r['pair_group_id']:r for r in rows if r['factor']=='clean'};clean_list=list(clean.values())
    if len(clean_list)!=10 or len({r['base_task'] for r in clean_list})!=10:raise ValueError('ten unique Clean tasks required')
    tasks=sorted({r['base_task'] for r in rows});keys=sorted(rows[0]['moments']);stats={};sources={};arrays={};thresholds={}
    for ti,task in enumerate(tasks):
        source=[r for r in clean_list if r['base_task']!=task]
        sources[task]=[r['case_id'] for r in source]
        if len(source)!=9:raise ValueError('LOTO source count')
        for key in keys:
            s=fit_moments([r['moments'][key] for r in source])
            if key.startswith('rgb__'):
                mean=np.mean([r['moments'][key][0] for r in source],axis=0)
                second=np.mean([r['moments'][key][1] for r in source],axis=0);s['mean']=mean;s['std']=np.sqrt(np.maximum(second-mean*mean,1e-8));s['variance']=s['std']**2
            stats[(task,key)]=s
            for name in ('mean','std','variance'):arrays[f'{ti}__{key}__{name}']=s[name]
            if not key.startswith(('rgb__','suffix__')):
                inner=[]
                for held in source:
                    fit=[r['moments'][key] for r in source if r['case_id']!=held['case_id']]
                    if len(fit)!=8:raise ValueError('inner LOO source count')
                    inner.append(distance(held['moments'][key],fit_moments(fit))['z_rms'])
                thresholds[(task,key)]=max(inner)
    scored=[]
    for r in rows:
        result={k:r[k] for k in ['case_id','factor','sub_dimension','base_task','pair_group_id','baseline_success','noise_code']};scores={}
        for key in keys:
            if key.startswith('rgb__'):continue
            d=distance(r['moments'][key],stats[(r['base_task'],key)])
            for metric,value in d.items():scores[key+'__'+metric]=value
        classifications={}
        for loc in ('dino','dino_mean','resampler','key','value'):
            av=scores[loc+'__agentview__z_rms'];eye=scores[loc+'__eye_in_hand__z_rms']
            scores[loc+'__asym']=av/(eye+1e-6)
            aa=av>thresholds[(r['base_task'],loc+'__agentview')];ee=eye>thresholds[(r['base_task'],loc+'__eye_in_hand')]
            classifications[loc]='both' if aa and ee else 'agent_only' if aa else 'wrist_only' if ee else 'neither'
        scores['flow_m4_variance']=r['flow_m4_variance'];scores['flow_m4_relative']=r['flow_m4_relative'];result['scores']=scores
        ref=clean[r['pair_group_id']]
        if r['z0_hashes']!=ref['z0_hashes']:raise ValueError('paired noise differs')
        current=np.asarray(r['actions']);original=np.asarray(ref['actions']);errors={}
        for name,sl in [('xyz',slice(0,3)),('rotation',slice(3,6)),('all',slice(None))]:
            errors[name]=float(np.linalg.norm((current-original)[:,:,sl].reshape(4,-1),axis=1).mean())
        changed={view:r['baseline_initial_rgb'][view]['sha256']!=ref['baseline_initial_rgb'][view]['sha256'] for view in ('agentview','eye_in_hand')}
        result.update(reference_errors=errors,predicted_view_change=classifications,actual_rgb_changed=changed);scored.append(result)
    rng=np.random.default_rng(20261124);freq=np.stack([np.bincount(rng.choice(len(tasks),len(tasks),replace=True),minlength=len(tasks)) for _ in range(1000)])
    groups={'sensor_noise':lambda r:r['factor']=='sensor_noise','background':lambda r:r['factor']=='background',
            'background_table':lambda r:r['factor']=='background' and r['sub_dimension']=='table_texture',
            'background_scene':lambda r:r['factor']=='background' and r['sub_dimension']=='scene_texture','lighting':lambda r:r['factor']=='lighting'}
    metrics={}
    for group,select in groups.items():
        rr=[r for r in scored if select(r)];y=np.array([not r['baseline_success'] for r in rr]);w=freq[:,[tasks.index(r['base_task']) for r in rr]]
        flow=np.array([r['scores']['flow_m4_variance'] for r in rr]);bf=bootstrap_auc(y,flow,w);entries={}
        for score in scored[0]['scores']:
            ss=np.array([r['scores'][score] for r in rr]);bs=bootstrap_auc(y,ss,w);valid=np.isfinite(bs)&np.isfinite(bf)
            entries[score]={'failure_auc':auc(y,ss),'same_task':concordance(rr,ss),
                'spearman_xyz_reference':finite_corr(ss,[r['reference_errors']['xyz'] for r in rr]),
                'spearman_rotation_reference':finite_corr(ss,[r['reference_errors']['rotation'] for r in rr]),
                'task_bootstrap_auc_95':np.quantile(bs[np.isfinite(bs)],[.025,.975]).tolist() if np.isfinite(bs).any() else None,
                'task_bootstrap_auc_minus_flow_95':np.quantile((bs-bf)[valid],[.025,.975]).tolist() if valid.any() else None,'valid_bootstraps':int(valid.sum())}
        metrics[group]={'n':len(rr),'failures':int(y.sum()),'tasks':len({r['base_task'] for r in rr}),'scores':entries}
    asym={};predictions={}
    for loc in ('dino','dino_mean','resampler','key','value'):
        asym[loc]={};predictions[loc]={}
        for factor in ('clean','sensor_noise','background','lighting'):
            rr=[r for r in scored if r['factor']==factor];v=[r['scores'][loc+'__asym'] for r in rr]
            asym[loc][factor]={'median':float(np.median(v)),'quartiles':np.quantile(v,[.25,.75]).tolist(),'min':min(v),'max':max(v)}
            predictions[loc][factor]=dict(Counter(r['predicted_view_change'][loc] for r in rr))
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'loto_stats.npz',**arrays)
    meta={'tasks':tasks,'source_case_ids':sources,'independent_source_tasks':9,'source_role':'other-task Dev Clean reference, not independent Source Calibration',
          'thresholds':{task:{key:float(v) for (t,key),v in thresholds.items() if t==task} for task in tasks},
          'method':'pooled token-channel diagonal moments, shrinkage .1; RGB empirical std; inner8/outer9 Clean thresholds',
          'source_feature_json_sha256':{p.name:file_hash(p) for p in paths}}
    (a.out/'loto_stats.json').write_text(json.dumps(meta,indent=2)+'\n');(a.out/'scored_cases.json').write_text(json.dumps(scored,indent=2)+'\n')
    report={'case_count':100,'metrics':metrics,'asym_distributions':asym,'source_only_view_predictions':predictions,
        'max_native_predict_error':max(r['official_max_abs'] for r in rows),'max_v1_action_error':max(r['v1_actions_max_abs'] for r in rows),
        'limits':['nine source Clean scenes; tokens are correlated','many descriptive metrics; no multiple-comparison-corrected superiority claim',
                  'suffix is fused; asymmetry only defined for view-separable locations','RGB changed is not proof of task-relevant damage','paired Clean predictions are not expert true actions']}
    (a.out/'scan_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'scan_complete':True,'cases':100,'native_error':report['max_native_predict_error']}))

if __name__=='__main__':main()
