"""Closed-loop success per seed, plus camera selection on standard and reverse stress."""
import sys,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot(s,path):
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');ax=axes[0];colors=['#999999','#237377','#bf843b','#505e9a'];keys=['frozen','continuous','fixed','symmetric'];x=np.arange(3);width=.18
    for j,(m,color,name) in enumerate(zip(keys,colors,['Frozen','Continuous','Fixed 0.10','Symmetric'])):
        yy=[]
        for seed in range(3):
            grp=s['seeds'][f'standard_{seed}']['methods'];yy.append(grp['continuous']['sensor_noise']['base'] if m=='frozen' else grp[m]['sensor_noise']['success'])
        ax.bar(x+(j-1.5)*width,yy,width,color=color,label=name);ax.bar_label(ax.containers[-1],fontsize=8)
    ax.set_ylim(0,30.8);ax.set_xticks(x,['Seed 0','Seed 1','Seed 2']);ax.set_ylabel('Sensor successes / 30');ax.set_title('Standard Dev: corrupt agentview');ax.legend(loc='lower center',fontsize=8,ncol=2);ax.grid(axis='y',alpha=.15)
    ax=axes[1];aa=s['seeds']['standard_0']['methods']['symmetric']['sensor_noise'];bb=s['seeds']['wrist_0']['methods']['symmetric']['sensor_noise'];groups=[aa,bb];left=np.zeros(2)
    for field,color,label in [('agent','#237377','Lower agentview'),('wrist','#bf843b','Lower wrist'),('none','#d5d5d5','No update')]:
        yy=np.array([g[field] if field!='none' else g['n']-g['accepted'] for g in groups]);ax.barh([1,0],yy,left=left,color=color,label=label)
        for i,n in enumerate(yy):
            if n:ax.text(left[i]+n/2,1-i,str(n),ha='center',va='center',color='black' if field=='none' else 'white',fontsize=10)
        left+=yy
    ax.set_yticks([1,0],['Agent corrupted','Wrist corrupted']);ax.set_xlim(0,30);ax.set_ylim(-.8,1.8);ax.set_xlabel('Number of episodes (Seed 0)');ax.set_title('Which camera did the symmetric gate select?');ax.legend(loc='lower center',bbox_to_anchor=(.5,-.37),fontsize=8,ncol=3)
    fig.savefig(path,dpi=150);plt.close(fig)
if __name__=='__main__':plot(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
