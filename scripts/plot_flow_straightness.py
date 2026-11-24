"""Standalone scientific figure from frozen flow diagnostic outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--gradients',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    rows=[json.loads(p.read_text()) for p in sorted(a.results.glob('*.json'))];gr=[json.loads(p.read_text()) for p in sorted(a.gradients.glob('*.json'))]
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained');factors=['clean','lighting','background','sensor_noise'];rng=np.random.default_rng(1)
    for panel,metric,title in [(0,'velocity_variance','Native-flow velocity variance'),(1,'relative_variance','Variance / velocity energy')]:
        for i,f in enumerate(factors):
            rr=[r for r in rows if r['factor']==f]
            vals=np.array([r['metrics'][0]['all'][metric] for r in rr]);jitter=rng.uniform(-.16,.16,len(rr))
            colors=['#23856d' if r['baseline_success'] else '#ca4c42' for r in rr]
            ax[panel].scatter(i+jitter,vals,c=colors,s=24,alpha=.8)
            ax[panel].plot([i-.2,i+.2],[np.median(vals)]*2,color='black',lw=2)
        ax[panel].set_yscale('log');ax[panel].set_xticks(range(4),['Clean','Lighting','Background','Sensor'],rotation=20);ax[panel].set_title(title);ax[panel].grid(axis='y',alpha=.2)
    for i,r in enumerate(gr):
        g=r['gradient'];before=g['before']['all']['velocity_variance'];after=g['after']['all']['velocity_variance']
        hb=g['heldout_before']['all']['velocity_variance'];ha=g['heldout_after']['all']['velocity_variance']
        ax[2].scatter(i-.1,100*(after/before-1),c='#3b66a1',marker='o',label='optimized z0' if i==0 else None)
        ax[2].scatter(i+.1,100*(ha/hb-1),c='#ce8b30',marker='x',label='held-out z0' if i==0 else None)
    ax[2].axhline(0,color='black',lw=1);ax[2].set_title('One gate step: curvature change (%)');ax[2].set_xticks(range(len(gr)),[r['factor'].replace('_noise','')[:5]+(' S' if r['baseline_success'] else ' F') for r in gr],rotation=45);ax[2].legend(fontsize=8);ax[2].grid(axis='y',alpha=.2)
    fig.suptitle('Flow geometry diagnostic — development data, t=0, frozen policy\nGreen = eventual success; red = eventual failure. No candidate actions executed.',fontsize=11)
    a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180);fig.savefig(a.out.with_suffix('.pdf'))
if __name__=='__main__':main()
