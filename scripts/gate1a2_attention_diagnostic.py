"""Gate 1A-2: Resampler attention response vs global-stat gate, and action deviation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path('/hy-tmp/patch-diag')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'repos/CoRP')); sys.path.insert(0, str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa


def resampler_tokens_attention(resampler, patches, pooled):
    """Replicate Resampler forward and return output + per-view attention [H,Q,N]."""
    gamma, beta = resampler.film(pooled).unsqueeze(1).chunk(2, dim=-1)
    batch_size = pooled.shape[0]
    outs = []; attns = []; score_stats = []
    for index, name in enumerate(resampler.view_names):
        p = resampler.input_projection(patches[name]) + resampler.view_embedding[index].unsqueeze(0)
        q = resampler.queries[index].unsqueeze(0).expand(batch_size, -1, -1)
        q = q * (1.0 + gamma) + beta
        W = resampler.cross_attention.in_proj_weight
        b = resampler.cross_attention.in_proj_bias
        h = resampler.cross_attention.num_heads
        E = W.shape[1]; D = E // h
        qkv = F.linear(q, W, b)
        qq, kk, vv = qkv.chunk(3, dim=-1)
        kk = F.linear(p, W[h * E:2 * h * E], b[h * E:2 * h * E]) if False else kk
        # Above is intentionally unused; generic projections are below.
        Wq, Wk, Wv = W[:E], W[E:2 * E], W[2 * E:]
        bq, bk, bv = b[:E], b[E:2 * E], b[2 * E:]
        qq = F.linear(q, Wq, bq); kk = F.linear(p, Wk, bk); vv = F.linear(p, Wv, bv)
        qq = qq.view(batch_size, -1, h, D).transpose(1, 2)
        kk = kk.view(batch_size, -1, h, D).transpose(1, 2)
        vv = vv.view(batch_size, -1, h, D).transpose(1, 2)
        scores = torch.matmul(qq, kk.transpose(-2, -1)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, vv).transpose(1, 2).reshape(batch_size, -1, E)
        ctx = resampler.cross_attention.out_proj(ctx)
        outs.append(resampler.norm(ctx)); attns.append(attn.detach()); score_stats.append(scores.detach())
    return torch.cat(outs, dim=1), attns, score_stats


def attention_descriptor(attn, scores):
    # attn [H,Q,N]
    logp = torch.log(attn.clamp_min(1e-12))
    ent = -(attn * logp).sum(dim=-1)  # H,Q
    eff = torch.exp(ent)
    lse = torch.logsumexp(scores, dim=-1)
    maxp = attn.max(dim=-1).values
    return {
        'entropy_mean': float(ent.mean()), 'entropy_std': float(ent.std()),
        'effective_mean': float(eff.mean()), 'effective_std': float(eff.std()),
        'maxprob_mean': float(maxp.mean()), 'maxprob_std': float(maxp.std()),
        'lse_mean': float(lse.mean()), 'lse_std': float(lse.std()),
    }


def descriptor_vec(d):
    return np.array([d[k] for k in ('entropy_mean','entropy_std','effective_mean','effective_std','maxprob_mean','maxprob_std','lse_mean','lse_std')], dtype=float)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=None)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    man_path=args.manifest or args.pairs.with_suffix('.manifest.json')
    man=json.loads(man_path.read_text()); records=man['records']; npz=np.load(args.pairs,allow_pickle=False)
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    resampler=policy.representation_extractor.cross_attention_resampler
    z0s=torch.tensor(npz['z0'],dtype=torch.float32,device=device)
    shift_lang=torch.tensor(npz['shift_language_tokens'],dtype=torch.float32,device=device)
    shift_pooled=torch.tensor(npz['shift_pooled_language'],dtype=torch.float32,device=device)
    shift_state=torch.tensor(npz['shift_state_token'],dtype=torch.float32,device=device)
    view_names=list(man['view_names'])
    rows=[]; verify=[]
    with torch.no_grad():
        for r in records:
            i=int(r['feature_row'])
            clean={v:torch.tensor(npz[f'clean_{v}'][i],dtype=torch.float32,device=device)[None] for v in view_names}
            shift={v:torch.tensor(npz[f'shift_{v}'][i],dtype=torch.float32,device=device)[None] for v in view_names}
            pool_c=torch.tensor(npz['clean_pooled_language'][i],dtype=torch.float32,device=device)[None]
            pool_s=shift_pooled[i:i+1]
            tok_c, att_c, score_c = resampler_tokens_attention(resampler, clean, pool_c)
            tok_s, att_s, score_s = resampler_tokens_attention(resampler, shift, pool_s)
            orig_c=resampler(clean,pool_c); orig_s=resampler(shift,pool_s)
            err=max(float((tok_c-orig_c).abs().max()), float((tok_s-orig_s).abs().max()))
            verify.append(err)
            desc={v:{'clean':attention_descriptor(att_c[j], score_c[j]),'shift':attention_descriptor(att_s[j], score_s[j])} for j,v in enumerate(view_names)}
            # B1 shift vs B2 clean patches, same Shift context/z0
            z0=z0s[i:i+1]
            def act(patch_tokens):
                visual=resampler(patch_tokens,pool_s)
                rep=Representation(tokens=torch.cat([visual,shift_lang[i:i+1].to(visual.dtype),shift_state[i:i+1]],dim=1),
                                   pooled_language=pool_s, action_dtype=visual.dtype)
                return policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))[0]
            a_shift=act(shift); a_clean=act(clean)
            d=(a_clean-a_shift).float().cpu().numpy()
            rows.append({'feature_row':i,'state_id':int(r['state_id']),'role':r['role'],'window':r['window'],'chunk_index':int(r['chunk_index']),
                         'attention':desc,
                         'action_delta_norm':float(np.sqrt(np.mean(d**2))),
                         'action_delta_xyz':float(np.sqrt(np.mean(d[:,:3]**2))),
                         'action_delta_rot':float(np.sqrt(np.mean(d[:,3:6]**2))),
                         'action_delta_grip':float(np.sqrt(np.mean(d[:,6]**2)))})
    # descriptors to vectors and domain gate
    def state_vectors(view, domain):
        out={}
        for r,row in zip(records,rows):
            key=(r['role'],int(r['state_id']),r['window'])
            out.setdefault(key,[]).append(descriptor_vec(row['attention'][view][domain]))
        return {k:np.mean(v,axis=0) for k,v in out.items()}
    gate={}
    for view in view_names:
        gate[view]={}
        for window in ('early','late'):
            c=state_vectors(view,'clean'); s=state_vectors(view,'shift')
            fit_c={k[1]:v for k,v in c.items() if k[0]=='source_fit' and k[2]==window}
            val_c={k[1]:v for k,v in c.items() if k[0]=='source_validation' and k[2]==window}
            fit_s={k[1]:v for k,v in s.items() if k[0]=='source_fit' and k[2]==window}
            val_s={k[1]:v for k,v in s.items() if k[0]=='source_validation' and k[2]==window}
            if not fit_c or not val_c: gate[view][window]={'status':'insufficient'}; continue
            prior=np.mean([fit_c[x] for x in sorted(fit_c)],axis=0)
            sd=np.maximum(np.std([fit_c[x] for x in sorted(fit_c)],axis=0),1e-4)
            def sc(d): return float(np.mean(((d-prior)/sd)**2))
            fcs={x:sc(v) for x,v in fit_c.items()}; vcs={x:sc(v) for x,v in val_c.items()}
            fss={x:sc(v) for x,v in fit_s.items()}; vss={x:sc(v) for x,v in val_s.items()}
            tau=float(np.max(list(fcs.values())))
            gate[view][window]={'fit_clean_mean':float(np.mean(list(fcs.values()))),'fit_shift_mean':float(np.mean(list(fss.values()))),
                                'val_clean_mean':float(np.mean(list(vcs.values()))),'val_shift_mean':float(np.mean(list(vss.values()))),
                                'tau':tau,'fit_clean_fpr':float(np.mean(np.array(list(fcs.values()))>tau)),
                                'val_clean_fpr':float(np.mean(np.array(list(vcs.values()))>tau)),
                                'fit_shift_activation':float(np.mean(np.array(list(fss.values()))>tau)),
                                'val_shift_activation':float(np.mean(np.array(list(vss.values()))>tau)),
                                'val_auroc':float(roc_auc_score([0]*len(vcs)+[1]*len(vss),list(vcs.values())+list(vss.values())))}
    # correlation attention shift-clean descriptor distance vs action deviation
    corr={}
    for view in view_names:
        for window in ('early','late'):
            sel=[r for r in rows if r['window']==window]
            x=[]; y=[]
            for r in sel:
                a=r['attention'][view]['clean']; b=r['attention'][view]['shift']
                x.append(float(np.linalg.norm(descriptor_vec(a)-descriptor_vec(b)))); y.append(r['action_delta_norm'])
            x=np.array(x); y=np.array(y)
            corr[f'{view}/{window}']={'n':len(x),'pearson':float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>0 else None,
                                      'x_mean':float(x.mean()),'y_mean':float(y.mean()),
                                      'x_high_action':float(x[y>np.median(y)].mean()) if len(x) else None,
                                      'x_low_action':float(x[y<=np.median(y)].mean()) if len(x) else None}
    report={'schema':'gate1a2_attention_diagnostic_v1','pairs':str(args.pairs),'manifest':str(man_path),
            'n_records':len(rows),'instrumentation_max_error':max(verify) if verify else None,
            'instrumentation_pass':bool(max(verify)<1e-5) if verify else False,
            'attention_gate':gate,'attention_action_correlation':corr,'records':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'n':len(rows),'instrumentation_max_error':report['instrumentation_max_error'],
                      'gate':gate,'corr':corr},indent=2)[:5000])

if __name__=='__main__': main()
