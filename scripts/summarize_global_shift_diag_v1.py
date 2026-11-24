"""Join descriptive LOTO scan with oracle ceilings and source-only probe results."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import file_hash
from dacsc.flow_geometry import auc


def action_metrics(actions,base,reference):
    out={}
    for space,clip in [('controller',False),('clipped',True)]:
        a=np.array(actions);b=np.array(base);r=np.array(reference)
        if clip:a=np.clip(a,-1,1);b=np.clip(b,-1,1);r=np.clip(r,-1,1)
        d={}
        for name,sl in [('xyz',slice(0,3)),('rotation',slice(3,6)),('all',slice(None))]:
            bv=float(np.linalg.norm((b-r)[:,:,sl].reshape(4,-1),axis=1).mean());v=float(np.linalg.norm((a-r)[:,:,sl].reshape(4,-1),axis=1).mean())
            d[name]={'baseline_distance':bv,'distance':v,'reduction_pct':100*(1-v/bv) if bv>1e-4 else None}
        norm=np.linalg.norm(a.reshape(4,-1),axis=1)/np.maximum(np.linalg.norm(b.reshape(4,-1),axis=1),1e-12)
        d['action_norm_ratio_mean']=float(norm.mean());d['action_norm_ratio_min']=float(norm.min());out[space]=d
    return out


def main():
    ap=argparse.ArgumentParser()
    for k in ['analysis','probes','both-value','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();scan=json.loads((a.analysis/'scan_summary.json').read_text());scored=json.loads((a.analysis/'scored_cases.json').read_text());stat=json.loads((a.analysis/'loto_stats.json').read_text())
    pp=sorted(a.probes.glob('*.json'));rows=[json.loads(p.read_text()) for p in pp]
    extra=json.loads(a.both_value.read_text());ev={r['case_id']:r for r in extra['rows']}
    if len(rows)!=29 or len(ev)!=29 or any(r['status']!='completed' for r in rows):raise ValueError('29 complete probe cases and matched both-Value controls required')
    expected={r['case_id'] for r in scored if r['factor']=='clean' or (r['factor'] in ('background','lighting') and not r['baseline_success'])}
    if {r['case_id'] for r in rows}!=expected or set(ev)!=expected:raise ValueError('probe population mismatch')
    ledger=[json.loads(l) for l in (a.probes/'access_ledger.jsonl').read_text().splitlines()]
    detail=[];max_dual_error=0
    for p,r in zip(pp,rows):
        if not any(e.get('action')=='global_probe_end' and e.get('case_id')==r['case_id'] and e.get('result_sha256')==file_hash(p) for e in ledger):raise ValueError('probe end receipt missing')
        if r['source_case_ids']!=stat['source_case_ids'][r['base_task']] or r['source_n']!=9:raise ValueError('source mismatch')
        if r['policy_weights_updated'] or r['adapted_actions_executed'] or r['blind_accessed']:raise ValueError('forbidden action')
        if len(r['variants'])!=11:raise ValueError('missing fixed probe conditions')
        baseline=float(np.mean(r['step0']['relative_curvature']))
        for variant in r['variants']+[ev[r['case_id']]]:
            am=action_metrics(variant['actions'],r['step0']['actions'],r['paired_clean_actions']);after=float(np.mean(variant['relative_curvature']))
            entry={'case_id':r['case_id'],'factor':r['factor'],'sub_dimension':r['sub_dimension'],'base_task':r['base_task'],
                   'method':variant['name'],'privileged_oracle':variant['privileged_oracle'],'actions':am,
                   'relative_curvature_before':baseline,'relative_curvature_after':after,'relative_curvature_change_pct':100*(after/baseline-1),
                   'nonworsening_curvature_accept':after<=baseline*(1+1e-6)}
            if 'altered_components' in variant:entry['altered_fraction']=variant['altered_components']/variant['total_components']
            detail.append(entry)
            if variant['name']=='oracle_both_kv':max_dual_error=max(max_dual_error,am['controller']['all']['distance'])
    groups={'background_fail':lambda r:r['factor']=='background','background_table_fail':lambda r:r['factor']=='background' and r['sub_dimension']=='table_texture',
            'background_scene_fail':lambda r:r['factor']=='background' and r['sub_dimension']=='scene_texture','lighting_fail':lambda r:r['factor']=='lighting','clean':lambda r:r['factor']=='clean'}
    methods=sorted(set(r['method'] for r in detail));summary={}
    for group,select in groups.items():
        summary[group]={}
        for method in methods:
            rr=[r for r in detail if r['method']==method and select(r)];d={'n':len(rr),'privileged_oracle':rr[0]['privileged_oracle'],'metrics':{}}
            for space in ('controller','clipped'):
                d['metrics'][space]={}
                for channel in ('xyz','rotation','all'):
                    z=[r['actions'][space][channel] for r in rr];v=[x['reduction_pct'] for x in z if x['reduction_pct'] is not None]
                    d['metrics'][space][channel]={'eligible_pct_n':len(v),'mean_reduction_pct':float(np.mean(v)) if v else None,
                        'median_reduction_pct':float(np.median(v)) if v else None,'material_improvement_n':sum(x>=10 for x in v),'material_harm_n':sum(x<=-10 for x in v),
                        'mean_absolute_distance':float(np.mean([x['distance'] for x in z])),'max_absolute_distance':max(x['distance'] for x in z)}
            accepted=[r for r in rr if r['nonworsening_curvature_accept']]
            d['curvature_guard']={'accepted_n':len(accepted),'accepted_xyz_harm_ge10_n':sum(r['actions']['clipped']['xyz']['reduction_pct'] is not None and r['actions']['clipped']['xyz']['reduction_pct']<=-10 for r in accepted),
                'accepted_clean_mean_xyz_drift':float(np.mean([r['actions']['clipped']['xyz']['distance'] for r in accepted])) if group=='clean' and accepted else None}
            d['mean_relative_curvature_change_pct']=float(np.mean([r['relative_curvature_change_pct'] for r in rr]));summary[group][method]=d
    asym_auc={}
    shift=[r for r in scored if r['factor']!='clean']
    for loc in ('dino','dino_mean','resampler','key','value'):
        asym_auc[loc]=auc([r['factor']=='sensor_noise' for r in shift],[r['scores'][loc+'__asym'] for r in shift])
    combined={'status':'EXPLORATORY_GLOBAL_DIAG_COMPLETE','scan':scan,'probe_summary':summary,'probe_per_case':detail,
              'sensor_vs_background_lighting_asym_auc':asym_auc,'oracle_both_kv_max_full_chunk_controller_l2':max_dual_error,
              'source_only':'LOTO other9 Dev Clean; no same-task Clean in statistical fitting or target-only operators',
              'limits':['oracle replacement is privileged and not a deployable method','nine source tasks and many metrics; no confirmed manifold or superiority claim',
                        'failure-selected 29-case probes are not closed-loop validation','fused suffix cannot be split into per-view asymmetry',
                        'paired Clean predictions are reference outputs, not expert ground truth']}
    a.out.write_text(json.dumps(combined,indent=2,allow_nan=False)+'\n');print('global scan +29x12 probes joined; dualKV max L2',max_dual_error)

if __name__=='__main__':main()
