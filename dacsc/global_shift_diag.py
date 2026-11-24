"""LOTO Clean-only moment statistics and fixed pullbacks; no outcome inputs."""
import numpy as np

LAYERS=('dino','dino_mean','resampler','key','value','suffix')
VIEWS=('agentview','eye_in_hand')


def moments(tokens):
    x=np.asarray(tokens,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite tokens x channels required')
    return np.stack([x.mean(0),(x*x).mean(0)])


def fit_moments(source):
    s=np.asarray(source,dtype=np.float64)
    if s.ndim!=3 or s.shape[1]!=2 or len(s)<2:raise ValueError('source must be N x 2 x channels')
    mu=s[:,0].mean(0);var=np.maximum(s[:,1].mean(0)-mu*mu,0)
    scale=max(float(var.mean()),1e-12)
    variance=.9*var+.1*scale+1e-8*scale
    return {'mean':mu,'variance':variance,'std':np.sqrt(variance),'source_n':len(s),'shrinkage':.1}


def distance(target,statistics):
    t=np.asarray(target,dtype=np.float64);mu=statistics['mean'];var=statistics['variance']
    z2=(t[1]-2*mu*t[0]+mu*mu)/var
    z=float(np.sqrt(np.maximum(z2,0).mean()))
    denom=np.linalg.norm(t[0])*np.linalg.norm(mu)
    cos=float(1-np.clip(np.dot(t[0],mu)/max(denom,1e-12),-1,1))
    return {'z_rms':z,'cosine':cos}


def loto_stats(records,held_task,layer,view):
    clean=[r for r in records if r['factor']=='clean' and r['base_task']!=held_task]
    if len(clean)!=9 or len({r['base_task'] for r in clean})!=9:raise ValueError('exactly nine other Clean tasks required')
    key=layer+'__'+view
    return fit_moments([r['moments'][key] for r in clean]),[r['case_id'] for r in clean]


def pullback(x,mean,std,tau,beta):
    x=np.asarray(x);mean=np.asarray(mean);std=np.asarray(std)
    z=(x-mean)/std;mask=np.abs(z)>tau
    boundary=mean+np.sign(z)*tau*std
    return np.where(mask,x-beta*(x-boundary),x),mask
