"""Standalone selected-case heatmaps; no hyperparameter selection from the figure."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    d=json.loads(a.summary.read_text());sel=json.loads(a.selection.read_text());cases=sel['case_ids'][:8]
    names=[f'{k}|{o}|lr{lr}|step10' for k in ['spatial','view','joint'] for o in ['relative','late','anchored'] for lr in [.3,.8]]
    lookup={(r['configuration'],r['case_id']):r for r in d['per_case']}
    gains=np.array([[-lookup[(n,c)]['metrics']['clipped']['xyz']['distance_change_pct'] for c in cases] for n in names])
    flow=np.array([[-lookup[(n,c)]['metrics']['relative_curvature']['change_pct'] for c in cases] for n in names])
    fig,axes=plt.subplots(1,2,figsize=(13,9),layout='constrained')
    for ax,mat,title in zip(axes,[flow,gains],['Held-out relative-flow variance reduction (%)','Held-out paired-Clean XYZ distance reduction (%)']):
        im=ax.imshow(mat,cmap='RdYlGn',vmin=-60,vmax=60,aspect='auto')
        for i in range(len(names)):
            for j in range(8):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=7)
        ax.set_title(title,fontsize=10);ax.set_xticks(range(8),['S1','S2','S3','S4','S5','B1','B2','B3']);ax.set_yticks(range(len(names)),[n.replace('|step10','').replace('|lr',' / ').replace('|',' / ') for n in names],fontsize=8)
    fig.colorbar(im,ax=axes,shrink=.6,label='Positive = reduction; not task success')
    fig.suptitle('Post-LayerNorm Value-trust repair probe — selected Dev failures\n10 Adam steps, 4 held-out noises; all fixed configurations shown',fontsize=12)
    a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180);fig.savefig(a.out.with_suffix('.pdf'))
