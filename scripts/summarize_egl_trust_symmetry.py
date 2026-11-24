"""Verify all paired runs, quantify choice of eye, and directly compare trust weights."""
import argparse,collections,json,math
from pathlib import Path
import numpy as np

def pairs(base,method,ids):
    changes=np.array([int(method[i]['success'])-int(base[i]['success']) for i in ids]);b=int((changes==1).sum());c=int((changes==-1).sum());n=b+c
    return dict(n=len(ids),base=sum(int(base[i]['success']) for i in ids),success=sum(int(method[i]['success']) for i in ids),rescued=b,harmed=c,net=b-c,exact_p=min(1.,2*sum(math.comb(n,j) for j in range(min(b,c)+1))/2**n) if n else 1.)

def bootstrap(rows):
    tasks=sorted({r['task'] for r in rows});counts=np.array([sum(r['task']==t for r in rows) for t in tasks]);net=np.array([sum(r['delta'] for r in rows if r['task']==t) for t in tasks]);index=np.random.default_rng(20261124).integers(len(tasks),size=(5000,len(tasks)))
    return np.quantile(100*net[index].sum(1)/counts[index].sum(1),[.025,.975]).tolist()

def summarize(path,manifest,previous):
    mm=json.loads(manifest.read_text());cases={c['case_id']:c for c in mm['cases']};assert len(cases)==100 and mm['kind']=='dev';assert not list(path.glob('*.pending'));prior={(r['seed'],r['case_id']):r for r in json.loads(previous.read_text())['rows']};seeds={};all_rows=[];legacy=[]
    for regime,seed in [('standard',i) for i in range(3)]+[('wrist',0)]:
        ids={i for i,c in cases.items() if regime=='standard' or c['factor']=='sensor_noise'};arms=('frozen','continuous','fixed','symmetric') if regime=='standard' else ('frozen','symmetric');d={m:{} for m in arms}
        for p in path.glob(f'{regime}_s{seed}_*.json'):
            r=json.loads(p.read_text());i=r['case_id'];m=r['method'];assert i not in d[m] and r['factor']==cases[i]['factor'] and r['base_task']==cases[i]['base_task'] and r['regime']==regime and r['seed']==seed
            assert isinstance(r['success'],bool) and 0<r['steps']<=220 and [q['step'] for q in r['trace']]==list(range(0,r['steps'],12));d[m][i]=r
        assert all(set(rr)==ids for rr in d.values());base=d['frozen'];groups={};rejections=collections.Counter();corrected=collections.Counter()
        for i in sorted(ids):
            b=base[i];assert b['gate'] is None
            for m in arms:
                r=d[m][i];assert (r['initial_sha'],r['wrist_sha'],r['trace'][0]['state'])==(b['initial_sha'],b['wrist_sha'],b['trace'][0]['state']);s=r['applied_logs'];assert len(s)==2 and max(s)==0 and min(s)>=-4.606
                assert all(0<n<=64 and 0<x<=1 for _,_,n,x in r['roundings']);corrected[m]+=len(r['roundings']);g=r['gate']
                if m!='frozen':
                    if m=='symmetric':
                        assert max(g['candidate'])==0 and g['accepted']==(g['before']>1e-12 and g['after']<=.8*g['before']);fit,valid=g['train_hash'],g['val_hash'];assert g['logs']==s
                    else:
                        assert g['accepted']==(g['validation_before']>1e-12 and g['validation_after']<=.8*g['validation_before']);fit,valid=g['training_noise_hashes'],g['validation_noise_hashes']
                    assert len(set(fit+valid))==8
                    if not g['accepted']:
                        assert r['trace']==b['trace']
                        assert s==[0,0] and (r['steps'],r['success'],r['action_sha'])==(b['steps'],b['success'],b['action_sha']);rejections[m]+=1
                    elif m=='fixed':assert abs(math.exp(s[0])-.1)<1e-7 and s[1]==0
                    elif m=='continuous':assert s==[g['candidate_log_trust'],0.]
                all_rows.append(dict(regime=regime,seed=seed,case_id=i,task=r['base_task'],factor=r['factor'],subtype=cases[i]['sub_dimension'],method=m,accepted=bool(g and g['accepted']),eye='agent' if s[0]<0 else 'wrist' if s[1]<0 else 'none',trust=[math.exp(x) for x in s],base=b['success'],success=r['success'],delta=int(r['success'])-int(b['success'])))
            if regime=='standard':
                f=d['fixed'][i];c=d['continuous'][i];g=c['gate'];sg=d['symmetric'][i]['gate'];assert f['gate']==g and sg['train_hash']==g['training_noise_hashes'] and sg['val_hash']==g['validation_noise_hashes']
                old=prior[seed,i]
                if (old['baseline'],old['method'],old['accepted'])!=(int(b['success']),int(c['success']),g['accepted']):legacy.append(dict(seed=seed,case_id=i,old=old,current=[b['success'],c['success'],g['accepted']]))
        for m in arms[1:]:
            group={};factors=sorted({base[i]['factor'] for i in ids})
            for f in ['all',*factors]:
                subset=sorted(i for i in ids if f=='all' or base[i]['factor']==f);entry=pairs(base,d[m],subset);accepted=[i for i in subset if d[m][i]['gate']['accepted']];entry.update(candidate_eyes=dict(collections.Counter(('none' if min(d[m][i]['gate']['candidate'])==0 else 'wrist' if d[m][i]['gate']['candidate'][1]<0 else 'agent') for i in subset)) if m=='symmetric' else {},accepted=len(accepted),agent=sum(d[m][i]['applied_logs'][0]<0 for i in accepted),wrist=sum(d[m][i]['applied_logs'][1]<0 for i in accepted));group[f]=entry
            groups[m]=group
        summary=dict(methods=groups,rollback_exact=dict(rejections),rounding_corrections=dict(corrected))
        diag={}
        for eye in ('none','agent','wrist'):
            gg=[r['gate'] for r in d['symmetric'].values() if ('none' if min(r['gate']['candidate'])==0 else 'agent' if r['gate']['candidate'][0]<0 else 'wrist')==eye]
            diag[eye]={'n':len(gg),'accepted':sum(g['accepted'] for g in gg),'validation_drop_pct':[100*(1-g['after']/g['before']) for g in gg if g['before']>1e-12]}
        summary['symmetric_before_validation']=diag
        if regime=='standard':
            accepted=sorted(i for i in ids if d['continuous'][i]['gate']['accepted']);summary.update(fixed_vs_continuous=pairs(d['continuous'],d['fixed'],sorted(ids)),fixed_vs_continuous_accepted=pairs(d['continuous'],d['fixed'],accepted),sym_vs_continuous=pairs(d['continuous'],d['symmetric'],sorted(ids)))
        seeds[f'{regime}_{seed}']=summary
    assert len(all_rows)==1260
    aggregate={}
    for m in ('continuous','fixed','symmetric'):
        for f in ['all','sensor_noise','clean','background','lighting']:
            rr=[r for r in all_rows if r['regime']=='standard' and r['method']==m and (f=='all' or r['factor']==f)];aggregate[m+'/'+f]=dict(n=len(rr),base=sum(r['base'] for r in rr),success=sum(r['success'] for r in rr),rescued=sum(r['delta']>0 for r in rr),harmed=sum(r['delta']<0 for r in rr),accepted=sum(r['accepted'] for r in rr),gain_pp=100*np.mean([r['delta'] for r in rr]),task_bootstrap_95=bootstrap(rr))
    sx=[seeds['standard_'+str(i)]['methods']['symmetric'] for i in range(3)];w=seeds['wrist_0']['methods']['symmetric']['all'];standard=bool(aggregate['symmetric/sensor_noise']['gain_pp']>=20 and all(x['sensor_noise']['net']>0 and x['all']['accepted']-x['sensor_noise']['accepted']<=1 and x['all']['harmed']==x['sensor_noise']['harmed'] for x in sx));reverse=bool(w['net']>0 and 2*w['wrist']>w['accepted'])
    direct=[]
    for seed in range(3):
        base={r['case_id']:r for r in all_rows if r['regime']=='standard' and r['seed']==seed and r['method']=='continuous'}
        for r in all_rows:
            if r['regime']=='standard' and r['seed']==seed and r['method']=='fixed' and r['accepted']:
                direct.append(dict(seed=seed,case_id=r['case_id'],task=r['task'],delta=int(r['success'])-int(base[r['case_id']]['success'])))
    direct_info=dict(n=len(direct),unique_cases=len({x['case_id'] for x in direct}),fixed_only=sum(r['delta']>0 for r in direct),continuous_only=sum(r['delta']<0 for r in direct),fixed_minus_continuous_pp_95=bootstrap(direct) if direct else None)
    return dict(seeds=seeds,aggregate=aggregate,direct_fixed_vs_continuous_accepted=direct_info,prior_differences=legacy,standard_screen=standard,wrist_screen=reverse,both_pass=standard and reverse,rows=all_rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();summary=summarize(a.raw,a.manifest,a.previous);a.out.write_text(json.dumps(summary,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');print('standard',summary['standard_screen'],'wrist',summary['wrist_screen'])
