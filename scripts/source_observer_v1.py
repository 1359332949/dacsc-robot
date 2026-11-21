"""Two-process Source Observer V1. Fit finishes before Validation is opened."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.source_observer import (MODELS,CONTROLS,load_role,fit,predict,state_metrics,comparison,gate,sha)


def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def summary(metrics):
    a=np.array([r['rmse'] for r in metrics.values()])
    return {'n_states':len(a),'mean_rmse':float(a.mean()),'median_rmse':float(np.median(a)),
            'min_rmse':float(a.min()),'max_rmse':float(a.max())} if len(a) else {'n_states':0}


def correlation(x,y):
    from scipy.stats import spearmanr
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)


def evaluate(data,arrays,config):
    reports={};chunk_rows=[]
    ends=np.array([r['end_step'] for r in data.timing]);remaining=np.array([r['remaining_steps'] for r in data.timing])
    masks={'common_early':ends<=72,'nonterminal':remaining>24,
           'end_le_24':ends<=24,'end_25_to_72':(ends>24)&(ends<=72),'end_gt_72':ends>72,'remaining_le_24':remaining<=24}
    for view in config['views']:
        preds={k:predict(arrays,view,k,data.current[view],data.q,data.actions) for k in MODELS+CONTROLS}
        metrics={k:state_metrics(data,arrays,view,p,config) for k,p in preds.items()}
        comps={k:comparison(metrics['M1'],metrics[k],config) for k in ('M0','M2')+CONTROLS}
        subsets={}
        for name,mask in masks.items():
            mm={k:state_metrics(data,arrays,view,p,config,mask) for k,p in preds.items()}
            subsets[name]={'models':mm,'comparisons':{k:comparison(mm['M1'],mm[k],config,bootstrap=False) for k in ('M0','M2')},
                           'summary':{k:summary(m) for k,m in mm.items()}}
        raw=data.following[view]-data.current[view];delta=raw@arrays[view+'_pca'].T;scale=arrays[view+'_target_scale']
        errors={k:np.sqrt(np.mean(((p-delta)/scale)**2,axis=1)) for k,p in preds.items()}
        correlations={}
        for k in MODELS:
            correlations[k]={str(int(s)):{'error_vs_end':correlation(ends[data.states==s],errors[k][data.states==s]),
                'error_vs_remaining':correlation(remaining[data.states==s],errors[k][data.states==s]),
                'error_vs_target_norm':correlation(np.linalg.norm(delta[data.states==s]/scale,axis=1),errors[k][data.states==s])} for s in np.unique(data.states)}
        shuffled=data.actions.copy()
        for s in np.unique(data.states):
            ii=np.flatnonzero(data.states==s);shuffled[ii]=np.roll(data.actions[ii],1,axis=0)
        ps=predict(arrays,view,'M1',data.current[view],data.q,shuffled)
        ms=state_metrics(data,arrays,view,ps,config)
        action_probe={'per_state':ms,'summary':summary(ms),
            'relative_error_increase':{s:ms[s]['rmse']/metrics['M1'][s]['rmse']-1 for s in ms},
            'interpretation':'fixed model, within-state action mismatch; may be off-support; not a causal effect'}
        support={}
        from dacsc.source_observer import design
        z=(data.current[view]-arrays[view+'_mu'])@arrays[view+'_pca'].T
        for k in MODELS:
            pre=view+'_'+k;standard=(design(k,z,data.q,data.actions)-arrays[pre+'_x_mu'])/arrays[pre+'_x_scale']
            support[k]={'max_absolute_standardized_input':float(np.abs(standard).max()),
                        'fraction_rows_any_abs_z_gt3':float((np.abs(standard)>3).any(axis=1).mean())}
        # Explain how much of the original pooled delta is represented at all.
        energy={str(int(s)):float(np.sum(delta[data.states==s]**2)/np.sum(raw[data.states==s]**2)) for s in np.unique(data.states)}
        view_report={'models':metrics,'summary':{k:summary(m) for k,m in metrics.items()},'comparisons':comps,'subsets':subsets,
                     'time_correlations':correlations,'action_probe':action_probe,'input_support':support,'delta_energy_capture_by_state':energy}
        reports[view]=view_report
        for i,t in enumerate(data.timing):
            for k in MODELS+CONTROLS:
                chunk_rows.append(dict(t,view=view,model=k,error_scaled_rmse=float(errors[k][i]),
                    true_scaled_delta_norm=float(np.linalg.norm(delta[i]/scale))))
    return reports,chunk_rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['fit','evaluate'])
    ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--split',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model-dir',type=Path)
    args=ap.parse_args();config=json.loads(args.config.read_text())
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    code={p:sha(ROOT/p) for p in ['dacsc/source_observer.py','scripts/source_observer_v1.py','dacsc/source_interactions.py','dacsc/interaction_records.py']}
    if args.command=='fit':
        data=load_role(args.source_root,args.split,'source_fit',config['views'])
        arrays,stats=fit(data,config)
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite model')
        model_path=args.out/'model.npz';np.savez_compressed(model_path,**arrays)
        fit_metrics,rows=evaluate(data,arrays,config)
        # Training diagnostics never determine the preregistered parameters.
        import sklearn,scipy
        meta={'schema':'source_observer_v1_frozen_model','config':config,'config_sha256':sha(args.config),
              'split_sha256':sha(args.split),'model_sha256':sha(model_path),'code_sha256':code,
              'fit_states':sorted(set(data.states.tolist())),'fit_rows':len(data.states),'excluded_partial':data.excluded_partial,
              'fit_boundaries':len(data.boundary_states),'inputs':data.provenance,'fit_statistics':stats,
              'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__},
              'validation_opened_in_fit':False,'target_data_opened':False,'hyperparameter_search':False,
              'frozen_design_commit':'bbbaa85'}
        dump(args.out/'model.json',meta);dump(args.out/'fit_diagnostics.json',fit_metrics)
        print(json.dumps({'fit_done':True,'rows':len(data.states),'states':meta['fit_states'],'model_sha256':meta['model_sha256']}),flush=True)
    else:
        if args.model_dir is None:raise ValueError('--model-dir required')
        meta=json.loads((args.model_dir/'model.json').read_text());model_path=args.model_dir/'model.npz'
        if meta['config_sha256']!=sha(args.config) or meta['split_sha256']!=sha(args.split):raise ValueError('frozen config/split mismatch')
        if meta['code_sha256']!=code:raise ValueError('fit/evaluate code drift')
        before=sha(model_path)
        if before!=meta['model_sha256']:raise ValueError('model artifact changed')
        with np.load(model_path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        data=load_role(args.source_root,args.split,'source_validation',config['views'])
        if set(meta['fit_states'])&set(data.states.tolist()):raise ValueError('fit/validation overlap')
        reports,rows=evaluate(data,arrays,config)
        for v in config['views']:reports[v]['gate']=gate(reports[v],config)
        overall=all(r['gate']['pass'] for r in reports.values())
        after=sha(model_path)
        if before!=after:raise ValueError('evaluation mutated model')
        report={'schema':'source_observer_v1_gate_A','decision':'PASS_A' if overall else 'STOP_V1',
                'config_sha256':sha(args.config),'frozen_design_commit':meta['frozen_design_commit'],
                'model_sha256_before':before,'model_sha256_after':after,'code_sha256':code,
                'validation_states':sorted(set(data.states.tolist())),'validation_rows':len(data.states),
                'excluded_partial':data.excluded_partial,'validation_inputs':data.provenance,'views':reports,
                'refit_on_validation':False,'gates_B_C_executed':False,'target_data_opened':False,
                'interpretation':'gate A only; per-state bootstrap is exploratory; no cross-seed/task/domain or risk claim'}
        dump(args.out/'report.json',report)
        print(json.dumps({'decision':report['decision'],'views':{v:{'comparisons':reports[v]['comparisons'],'gate':reports[v]['gate']} for v in config['views']}},indent=2),flush=True)
    with (args.out/'chunk_diagnostics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

if __name__=='__main__':main()
