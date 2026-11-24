"""Paired closed-loop reclassifications, not unpaired success-rate error bars."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(summary,path):
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.2),layout='constrained',sharey=True)
    factors=['clean','sensor_noise','background','lighting'];extent=1+max(v for arm in summary['methods'].values() for f in factors for v in [arm[f]['rescued'],arm[f]['harmed']]);
    for ax,(key,title) in zip(axes,[('rgb','RGB alignment'),('trust','Gated Value trust'),('joint','RGB + gated trust')]):
        d=summary['methods'][key];a=d['all'];ax.axvline(0,c='0.5',lw=.8);ax.barh(range(4),[-d[f]['harmed'] for f in factors],color='#c55c53',label='Harmed');ax.barh(range(4),[d[f]['rescued'] for f in factors],color='#358a77',label='Rescued');ax.set_yticks(range(4),['Clean','Sensor','Background','Lighting']);ax.set_xlim(-extent,extent);ax.set_title(f'{title}\n{a["success"]}/100, net {a["net"]:+d}');ax.grid(axis='x',alpha=.16);ax.invert_yaxis() if ax is axes[0] else None
    axes[0].legend(loc='lower left',fontsize=8);fig.supxlabel('Lost Frozen successes  ←   paired episodes   →  recovered Frozen failures',fontsize=9)
    fig.suptitle('EGL Dev, seed 0 — paired changes from contemporaneous Frozen',fontsize=12);fig.savefig(path,dpi=170);plt.close(fig)
if __name__=='__main__':plot(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
