"""Offline only: attach paired-Clean distances after all gate optimization outputs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import file_hash


def distance(actions,reference,sl):
    d=np.asarray(actions)[:,:,sl]-np.asarray(reference)[:,:,sl]
    return float(np.linalg.norm(d.reshape(len(d),-1),axis=1).mean())


def evaluate(a,base,reference):
    result={}
    for space,key in [('controller','controller_actions'),('clipped','clipped_controller_actions')]:
        ref=np.asarray(reference)
        if space=='clipped':ref=np.clip(ref,-1,1)
        d={}
        for name,sl in [('xyz',slice(0,3)),('rotation',slice(3,6)),('all',slice(None))]:
            b=distance(base[key],ref,sl);v=distance(a[key],ref,sl)
            d[name]={'step0_distance':b,'distance':v,'distance_change_pct':100*(v/b-1) if b>1e-4 else None}
        norm0=np.linalg.norm(np.asarray(base[key]).reshape(4,-1),axis=1)
        norm1=np.linalg.norm(np.asarray(a[key]).reshape(4,-1),axis=1)
        d['action_norm_ratio_mean']=float(np.mean(norm1/np.maximum(norm0,1e-12)))
        d['action_norm_ratio_min_noise']=float(np.min(norm1/np.maximum(norm0,1e-12)))
        xyz0=np.linalg.norm(np.asarray(base[key])[:,:,:3].reshape(4,-1),axis=1)
        xyz1=np.linalg.norm(np.asarray(a[key])[:,:,:3].reshape(4,-1),axis=1)
        d['xyz_action_norm_ratio_mean']=float(np.mean(xyz1/np.maximum(xyz0,1e-12)))
        d['xyz_action_norm_ratio_min_noise']=float(np.min(xyz1/np.maximum(xyz0,1e-12)))
        result[space]=d
    for field in ('raw_curvature','relative_curvature'):
        b=float(np.mean(base[field]));v=float(np.mean(a[field]))
        result[field]={'step0':b,'after':v,'change_pct':100*(v/b-1)}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    sel=json.loads(a.selection.read_text());paths=sorted(a.results.glob('*.json'));rows=[json.loads(p.read_text()) for p in paths]
    if len(rows)!=10 or {r['case_id'] for r in rows}!=set(sel['case_ids']) or any(r['status']!='completed' for r in rows):raise ValueError('all ten completed selected cases required')
    outputs=[];groups={}
    for row in rows:
        if len(row['configurations'])!=18:raise ValueError('incomplete fixed grid')
        base=row['step0'];reference=row['offline_clean_reference']['actions']
        for config in row['configurations']:
            for step in ('5','10'):
                ck=config['checkpoints'][step];metric=evaluate(ck['adapted'],base,reference)
                null=[evaluate(x,base,reference) for x in ck['random_nulls']]
                if len(null)!=5:raise ValueError('five null draws required')
                nullxyz=[x['clipped']['xyz']['distance'] for x in null];nd=float(np.median(nullxyz));ad=metric['clipped']['xyz']['distance']
                b=metric['clipped']['xyz']['step0_distance']
                name=f"{config['knob']}|{config['objective']}|lr{config['learning_rate']}|step{step}"
                result={'case_id':row['case_id'],'factor':row['factor'],'base_task':row['base_task'],'baseline_success':row['baseline_success'],'configuration':name,
                        'metrics':metric,'null_median_xyz_distance':nd,'null_xyz_distances':nullxyz,
                        'null_median_xyz_change_pct':100*(nd/b-1) if b>1e-4 else None,
                        'beats_null_median':bool(ad<nd-1e-7),'ties_null_median':bool(abs(ad-nd)<=1e-7)}
                outputs.append(result);groups.setdefault(name,[]).append(result)
    summary={};passed=[]
    for name,rr in groups.items():
        fail=[r for r in rr if not r['baseline_success']];clean=[r for r in rr if r['factor']=='clean']
        if len(fail)!=8 or len(clean)!=2:raise ValueError('selected roles changed')
        gains=np.array([-r['metrics']['clipped']['xyz']['distance_change_pct'] for r in fail])
        nullgains=np.array([-r['null_median_xyz_change_pct'] for r in fail])
        # Preregistered exploratory existence screen, not hyperparameter selection.
        wins=sum(r['beats_null_median'] for r in fail);normmin=min(r['metrics']['clipped']['action_norm_ratio_min_noise'] for r in fail)
        go=bool(name.endswith('step10') and sum(gains>=10)>=6 and gains.mean()>=10 and wins>=6 and normmin>=.8)
        if go:passed.append(name)
        summary[name]={'n_failure':8,'n_material_xyz_improvement':int(sum(gains>=10)),'n_any_xyz_improvement':int(sum(gains>0)),
          'mean_xyz_improvement_pct':float(gains.mean()),'median_xyz_improvement_pct':float(np.median(gains)),
          'mean_null_xyz_improvement_pct':float(nullgains.mean()),'beats_null_median':wins,'ties_null_median':sum(r['ties_null_median'] for r in fail),
          'min_noise_action_norm_ratio':normmin,'mean_raw_curvature_change_pct':float(np.mean([r['metrics']['raw_curvature']['change_pct'] for r in fail])),
          'mean_relative_curvature_change_pct':float(np.mean([r['metrics']['relative_curvature']['change_pct'] for r in fail])),
          'clean_xyz_absolute_distance':[r['metrics']['clipped']['xyz']['distance'] for r in clean],
          'exploratory_screen_pass':go}
    verdict='EXPLORATORY_REFERENCE_ALIGNMENT_GO' if passed else 'NO_GO_FOR_STRAIGHTENING_REPAIR_V1'
    report={'status':verdict,'screen_pass_configurations':passed,'selection_sha256':file_hash(a.selection),
            'summary':summary,'per_case':outputs,'input_sha256':{p.name:file_hash(p) for p in paths},
            'limits':['extreme failures selected using outcomes and privileged Clean; not representative or confirmatory',
                      'Clean policy reference distance is not expert action correctness or closed-loop success',
                      '18 configurations examined; no Clean-distance-based production hyperparameter selection',
                      'view-only null necessarily identical; joint null preserves scalar and randomizes spatial block',
                      'negative repair does not certify a calibrated deployable risk gate']}
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(verdict,'passed',passed)
    for name,s in sorted(summary.items()):
        if name.endswith('step10'):print(name,'xyz gain',round(s['mean_xyz_improvement_pct'],2),'material',s['n_material_xyz_improvement'],'nullwins',s['beats_null_median'],'normmin',round(s['min_noise_action_norm_ratio'],3))

if __name__=='__main__':main()
