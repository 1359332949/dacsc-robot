"""Gate 1A: test the TP-TVA per-view source-prior domain score and gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def descriptor(patches):
    p = np.asarray(patches, dtype=np.float64)
    mu = p.mean(axis=0)
    sd = p.std(axis=0)
    return mu, np.log(sd + 1e-6)

def score(desc, prior_mu, prior_mu_scale, prior_logsd, alpha=1.0):
    mu, logsd = desc
    mean_term = np.mean((mu - prior_mu) ** 2 / (prior_mu_scale ** 2 + 1e-6))
    std_term = np.mean((logsd - prior_logsd) ** 2)
    return float(mean_term + alpha * std_term)

def state_desc(npz, records, view, domain, window):
    groups = {}
    for r in records:
        if r['role'] not in ('source_fit', 'source_validation') or r['window'] != window:
            continue
        if not bool(r.get('pairing_pass', False)):
            raise RuntimeError(f"Gate 1A received failed pair {r.get('role')}/{r.get('state_id')}/{r.get('chunk_index')}")
        groups.setdefault((r['role'], int(r['state_id'])), []).append(
            descriptor(npz[f'{domain}_{view}'][int(r['feature_row'])]))
    out = {}
    for key, vals in groups.items():
        mu = np.mean([v[0] for v in vals], axis=0)
        logsd = np.mean([v[1] for v in vals], axis=0)
        out[key] = (mu, logsd)
    return out

def run_view(npz, records, view, window):
    cd = state_desc(npz, records, view, 'clean', window)
    sd = state_desc(npz, records, view, 'shift', window)
    fit_clean = {s: d for (r, s), d in cd.items() if r == 'source_fit'}
    val_clean = {s: d for (r, s), d in cd.items() if r == 'source_validation'}
    fit_shift = {s: d for (r, s), d in sd.items() if r == 'source_fit'}
    val_shift = {s: d for (r, s), d in sd.items() if r == 'source_validation'}
    if not fit_clean or not val_clean:
        return {'status': 'insufficient'}
    if set(fit_clean) != set(fit_shift) or set(val_clean) != set(val_shift):
        return {'status': 'state_id_mismatch'}
    fit_ids = sorted(fit_clean)
    fit_mu = np.stack([fit_clean[s][0] for s in fit_ids])
    fit_logsd = np.stack([fit_clean[s][1] for s in fit_ids])
    prior_mu = fit_mu.mean(axis=0)
    prior_mu_scale = np.maximum(fit_mu.std(axis=0), 1e-3)
    prior_logsd = fit_logsd.mean(axis=0)
    def scores(mapping):
        return {s: score(d, prior_mu, prior_mu_scale, prior_logsd, alpha=args.alpha) for s, d in mapping.items()}
    fcs = scores(fit_clean); vcs = scores(val_clean)
    fss = scores(fit_shift); vss = scores(val_shift)
    fit_scores = np.array(list(fcs.values()))
    tau95 = float(np.quantile(fit_scores, 0.95))
    tau_max = float(np.max(fit_scores))
    def report(tau):
        return {
            'threshold': tau,
            'fit_clean_fpr': float(np.mean(fit_scores > tau)),
            'val_clean_fpr': float(np.mean(np.array(list(vcs.values())) > tau)),
            'fit_shift_activation': float(np.mean(np.array(list(fss.values())) > tau)),
            'val_shift_activation': float(np.mean(np.array(list(vss.values())) > tau)),
            'fit_clean_mean': float(np.mean(fit_scores)),
            'fit_shift_mean': float(np.mean(np.array(list(fss.values())))),
            'val_clean_mean': float(np.mean(np.array(list(vcs.values())))),
            'val_shift_mean': float(np.mean(np.array(list(vss.values())))),
            'fit_auroc_clean_vs_shift': float(roc_auc_score([0]*len(fcs)+[1]*len(fss), list(fcs.values())+list(fss.values()))),
            'val_auroc_clean_vs_shift': float(roc_auc_score([0]*len(vcs)+[1]*len(vss), list(vcs.values())+list(vss.values()))),
            'n_fit_states': len(fcs), 'n_val_states': len(vcs),
        }
    return {'status':'ok','tau_quantile95':report(tau95),'tau_train_max':report(tau_max),
            'alpha': float(args.alpha),
            'per_state': {'fit_clean':fcs,'fit_shift':fss,'val_clean':vcs,'val_shift':vss},
            'prior_mu_norm':float(np.linalg.norm(prior_mu)),
            'prior_mu_scale_mean':float(np.mean(prior_mu_scale)),
            'prior_logsd_mean':float(np.mean(prior_logsd))}

def main():
    global args
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=None)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--alpha',type=float,default=1.0)
    args=ap.parse_args()
    man_path=args.manifest or args.pairs.with_suffix('.manifest.json')
    man=json.loads(man_path.read_text()); records=man['records']; npz=np.load(args.pairs,allow_pickle=False)
    report={'schema':'gate1a_domain_gate_v1','pairs':str(args.pairs),'manifest':str(man_path),'views':{}}
    for view in man['view_names']:
        report['views'][view]={}
        for window in ('early','late'):
            report['views'][view][window]=run_view(npz,records,view,window)
    # Simple fused gate: max of per-view z-scored scores is not used yet;
    # report indicates which view should dominate.
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    for view,v in report['views'].items():
        for w,r in v.items():
            if r.get('status')!='ok': print(view,w,r['status']); continue
            for name in ('tau_quantile95','tau_train_max'):
                x=r[name]; print(view,w,name,'fit_fpr',round(x['fit_clean_fpr'],3),'val_fpr',round(x['val_clean_fpr'],3),'fit_act',round(x['fit_shift_activation'],3),'val_act',round(x['val_shift_activation'],3),'val_auc',round(x['val_auroc_clean_vs_shift'],3))
    print(json.dumps({'out':str(args.out)}))

if __name__=='__main__': main()
