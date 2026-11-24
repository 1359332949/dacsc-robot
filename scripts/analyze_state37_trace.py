"""Summarize three-input state37 trace."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def mean(xs): return float(statistics.mean(xs)) if xs else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trace',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    d=json.loads(args.trace.read_text()); rs=d['records']
    summary={'schema':'state37_trace_summary_v1','n_steps':len(rs),'pairing_pass':sum(bool(r['pairing_pass']) for r in rs),
             'target_contact_steps':sum(bool(r['target_finger_contact']) for r in rs),'target_geometry':d.get('target_geometry'),'steps':{}}
    for r in rs:
        rec={'step':r['step'],'eef_target_distance':r['eef_target_distance'],'theta_rel':r.get('theta_rel'),
             'target_contact':bool(r['target_finger_contact']),
             'dino_T':mean([r['dino'][v]['clean_vs_transform'] for v in r['dino']]),
             'dino_S':mean([r['dino'][v]['clean_vs_shift'] for v in r['dino']]),
             'dino_T_cos':mean([r['dino'][v]['clean_vs_transform_cosine'] for v in r['dino']]),
             'dino_S_cos':mean([r['dino'][v]['clean_vs_shift_cosine'] for v in r['dino']]),
             'patch_T_mean':mean([r['dino'][v]['clean_vs_transform_patch']['patch_l2_mean'] for v in r['dino']]),
             'patch_S_mean':mean([r['dino'][v]['clean_vs_shift_patch']['patch_l2_mean'] for v in r['dino']]),
             'res_T':r['resampler']['clean_vs_transform'],'res_S':r['resampler']['clean_vs_shift'],
             'act_T_pre':r['action']['clean_vs_transform']['pre'],'act_S_pre':r['action']['clean_vs_shift']['pre'],
             'act_T_post':r['action']['clean_vs_transform']['post'],'act_S_post':r['action']['clean_vs_shift']['post']}
        summary['steps'][str(r['step'])]=rec
    st=list(summary['steps'].values())
    def agg(path):
        vals=[]
        for x in st:
            v=x
            for k in path: v=v[k]
            vals.append(v)
        return {'mean':mean(vals),'min':min(vals),'max':max(vals)}
    summary['aggregate']={}
    for branch in ('T','S'):
        summary['aggregate'][branch]={
            'dino_rel':agg(('dino_'+branch,)),'dino_cosine':agg(('dino_'+branch+'_cos',)),
            'patch_l2_mean':agg(('patch_'+branch+'_mean',)),'resampler_rel':agg(('res_'+branch,)),
            'action_pre':{k:agg(('act_'+branch+'_pre',k)) for k in ('xyz','rot','grip','norm')},
            'action_post':{k:agg(('act_'+branch+'_post',k)) for k in ('xyz','rot','grip','norm')}}
    summary['target_contact_steps']=sum(1 for r in st if r['target_contact'])
    summary['first_significant_action_step_xyz_1e-3']=next((r['step'] for r in st if r['act_T_pre']['xyz']>1e-3),None)
    summary['first_significant_action_step_shift_xyz_1e-3']=next((r['step'] for r in st if r['act_S_pre']['xyz']>1e-3),None)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'aggregate':summary['aggregate'],'first_action':summary['first_significant_action_step_xyz_1e-3'],'contact_steps':summary['target_contact_steps']},indent=2))

if __name__=='__main__': main()
