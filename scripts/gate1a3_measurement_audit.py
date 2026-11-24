"""Gate 1A-3: action/attention measurement validity audit on paired patches."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa
from scripts.gate1a2_attention_diagnostic import resampler_tokens_attention, attention_descriptor  # noqa


def js_divergence(p,q):
    p=p.clamp_min(1e-12); q=q.clamp_min(1e-12); m=0.5*(p+q)
    return 0.5*(p*(p.log()-m.log())).sum(-1) + 0.5*(q*(q.log()-m.log())).sum(-1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--manifest',type=Path,default=None)
    ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--executed-k',type=int,default=12)
    args=ap.parse_args()
    man_path=args.manifest or args.pairs.with_suffix('.manifest.json')
    man=json.loads(man_path.read_text()); records=man['records']; npz=np.load(args.pairs,allow_pickle=False)
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    rs=policy.representation_extractor.cross_attention_resampler
    z0s=torch.tensor(npz['z0'],dtype=torch.float32,device=device)
    shift_lang=torch.tensor(npz['shift_language_tokens'],dtype=torch.float32,device=device)
    shift_pooled=torch.tensor(npz['shift_pooled_language'],dtype=torch.float32,device=device)
    shift_state=torch.tensor(npz['shift_state_token'],dtype=torch.float32,device=device)
    views=list(man['view_names']); rows=[]; parity=[]
    with torch.no_grad():
        for r in records:
            i=int(r['feature_row'])
            clean={v:torch.tensor(npz[f'clean_{v}'][i],dtype=torch.float32,device=device)[None] for v in views}
            shift={v:torch.tensor(npz[f'shift_{v}'][i],dtype=torch.float32,device=device)[None] for v in views}
            pc=torch.tensor(npz['clean_pooled_language'][i],dtype=torch.float32,device=device)[None]
            ps_=shift_pooled[i:i+1]
            tokc,ac,sc,ec=resampler_tokens_attention(rs,clean,pc)
            toks,as_,ss,es=resampler_tokens_attention(rs,shift,ps_)
            parity.append(max(ec+es))
            def ctrl(patch):
                visual=rs(patch,ps_)
                rep=Representation(tokens=torch.cat([visual,shift_lang[i:i+1].to(visual.dtype),shift_state[i:i+1]],dim=1),
                                   pooled_language=ps_,action_dtype=visual.dtype)
                return policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0s[i:i+1]))[0]
            a_c=ctrl(clean); a_s=ctrl(shift)
            k=min(args.executed_k,a_c.shape[0])
            ac_clip=a_c[:k].clamp(-1,1); as_clip=a_s[:k].clamp(-1,1)
            d=(ac_clip-as_clip).float().cpu().numpy()
            metrics={}
            for j,v in enumerate(views):
                acw=ac[j]; asw=as_[j]
                js=js_divergence(acw,asw)  # H,Q
                ent_c=-(acw.clamp_min(1e-12)*acw.clamp_min(1e-12).log()).sum(-1)
                ent_s=-(asw.clamp_min(1e-12)*asw.clamp_min(1e-12).log()).sum(-1)
                max_c=acw.max(-1).values; max_s=asw.max(-1).values
                metrics[v]={'js_mean':float(js.mean()),'js_std':float(js.std()),
                            'entropy_diff_abs_mean':float((ent_c-ent_s).abs().mean()),
                            'maxprob_diff_abs_mean':float((max_c-max_s).abs().mean()),
                            'attention_clean':attention_descriptor(acw,sc[j]),
                            'attention_shift':attention_descriptor(asw,ss[j])}
            rows.append({'feature_row':i,'state_id':int(r['state_id']),'role':r['role'],'window':r['window'],'chunk_index':int(r['chunk_index']),
                         'pairing_pass':bool(r.get('pairing_pass', False)),
                         'attention_metrics':metrics,
                         'action_exec':{'norm':float(np.sqrt(np.mean(d**2))),
                                        'xyz':float(np.sqrt(np.mean(d[:,:3]**2))),
                                        'rot':float(np.sqrt(np.mean(d[:,3:6]**2))),
                                        'grip':float(np.sqrt(np.mean(d[:,6]**2))),
                                        'k':int(k)}})
    # ---- Gate per single-observation attention metric ----
    metric_names=['entropy_mean','maxprob_mean','lse_mean']
    gate={}
    for v in views:
        gate[v]={}
        for win in ('early','late'):
            for metric in metric_names:
                def sd(domain):
                    g={}
                    for row in rows:
                        if row['window']!=win: continue
                        key=(row['role'],row['state_id'])
                        g.setdefault(key,[]).append(np.array([row['attention_metrics'][v][f'attention_{domain}'][metric]]))
                    return {k:np.mean(x,axis=0)[0] for k,x in g.items()}
                c=sd('clean'); s=sd('shift')
                fit_c={k[1]:val for k,val in c.items() if k[0]=='source_fit'}
                val_c={k[1]:val for k,val in c.items() if k[0]=='source_validation'}
                fit_s={k[1]:val for k,val in s.items() if k[0]=='source_fit'}
                val_s={k[1]:val for k,val in s.items() if k[0]=='source_validation'}
                if not fit_c or not val_c: continue
                ids=sorted(fit_c); vals=np.array([fit_c[x] for x in ids])
                mu=float(vals.mean()); scale=float(vals.std()+1e-6)
                fcs={k:abs(v-mu)/scale for k,v in fit_c.items()}; vcs={k:abs(v-mu)/scale for k,v in val_c.items()}
                fss={k:abs(v-mu)/scale for k,v in fit_s.items()}; vss={k:abs(v-mu)/scale for k,v in val_s.items()}
                tau=float(np.max(list(fcs.values())))
                gate[v][f'{win}/{metric}']={'tau':tau,'fit_clean_fpr':float(np.mean(np.array(list(fcs.values()))>tau)),
                    'val_clean_fpr':float(np.mean(np.array(list(vcs.values()))>tau)),
                    'fit_shift_activation':float(np.mean(np.array(list(fss.values()))>tau)),
                    'val_shift_activation':float(np.mean(np.array(list(vss.values()))>tau)),
                    'val_auroc':float(roc_auc_score([0]*len(vcs)+[1]*len(vss),list(vcs.values())+list(vss.values())))}
    # ---- state-level action/attention correlations ----
    metrics=['js_mean','entropy_diff_abs_mean','maxprob_diff_abs_mean']
    dims=['norm','xyz','rot','grip']
    corr={}
    for v in views:
        for win in ('early','late'):
            groups={}
            for row in rows:
                if row['window']!=win: continue
                key=(row['role'],row['state_id'])
                groups.setdefault(key,[]).append(row)
            keys=sorted(groups)
            for metric in metrics:
                for dim in dims:
                    xs=[]; ys=[]
                    for k in keys:
                        xs.append(np.mean([r['attention_metrics'][v][metric] for r in groups[k]]))
                        ys.append(np.mean([r['action_exec'][dim] for r in groups[k]]))
                    xs=np.array(xs); ys=np.array(ys)
                    if len(xs)<3 or xs.std()==0 or ys.std()==0: continue
                    rho=float(np.corrcoef(xs,ys)[0,1]); spear=float(spearmanr(xs,ys).statistic)
                    rng=np.random.default_rng(20261119); boots=[]
                    for _ in range(2000):
                        idx=rng.integers(0,len(xs),len(xs))
                        if xs[idx].std()>0 and ys[idx].std()>0: boots.append(float(np.corrcoef(xs[idx],ys[idx])[0,1]))
                    # role split
                    fit_idx=[i for i,k in enumerate(keys) if k[0]=='source_fit']; val_idx=[i for i,k in enumerate(keys) if k[0]=='source_validation']
                    def sub(idx):
                        if len(idx)<3: return None
                        return float(np.corrcoef(xs[idx],ys[idx])[0,1])
                    corr[f'{v}/{win}/{metric}/{dim}']={'n_states':len(keys),'pearson':rho,'spearman':spear,
                        'pearson_ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))] if boots else None,
                        'fit_pearson':sub(fit_idx),'val_pearson':sub(val_idx)}
    report={'schema':'gate1a3_measurement_audit_v1','pairs':str(args.pairs),'manifest':str(man_path),
            'n_records':len(rows),'attention_parity_max_error':max(parity) if parity else None,
            'action_definition':'controller actions clipped to [-1,1], first executed_k steps, dimension-separated',
            'gate':gate,'correlation':corr,'records':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'n':len(rows),'parity':report['attention_parity_max_error'],'gate':gate,'n_corr':len(corr)},indent=2)[:4000])

if __name__=='__main__': main()
