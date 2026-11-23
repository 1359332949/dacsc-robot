"""Zero-training spatial patch bias for the frozen CoRP resampler."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def boundary_bg_weights(patches: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """patches [B,N,C] -> [B,N] background weight from border-prototype similarity."""
    B,N,C=patches.shape
    if N!=grid*grid: raise ValueError(f'expected {grid*grid} patches, got {N}')
    x=F.normalize(patches.float(),dim=-1)
    g=x.view(B,grid,grid,C)
    border=torch.cat([g[:,0,:,:],g[:,:,0,:],g[:,:,-1,:]],dim=1)  # B,3*grid,C
    proto=F.normalize(border.mean(dim=1),dim=-1)  # B,C
    sim=(x*proto.unsqueeze(1)).sum(-1)  # B,N
    med=sim.median(dim=1,keepdim=True).values
    return torch.sigmoid((sim-med)/0.10)

def _manual_mha(mha, q, k, v, attn_bias=None):
    # q [B,L,H], k/v [B,N,H]; bias [B,1,1,N] or [B,N]
    B,L,H=q.shape; N=k.shape[1]; D=H//mha.num_heads
    w=mha.in_proj_weight; b=mha.in_proj_bias
    qq=F.linear(q,w[:H],b[:H]); kk=F.linear(k,w[H:2*H],b[H:2*H]); vv=F.linear(v,w[2*H:],b[2*H:])
    qq=qq.view(B,L,mha.num_heads,D).transpose(1,2)
    kk=kk.view(B,N,mha.num_heads,D).transpose(1,2)
    vv=vv.view(B,N,mha.num_heads,D).transpose(1,2)
    scores=torch.matmul(qq,kk.transpose(-2,-1))/math.sqrt(D)
    if attn_bias is not None:
        bias=attn_bias
        if bias.dim()==2: bias=bias[:,None,None,:]
        scores=scores+bias
    attn=torch.softmax(scores,dim=-1)
    out=torch.matmul(attn,vv).transpose(1,2).contiguous().view(B,L,H)
    return mha.out_proj(out)

def resampler_forward_with_bias(resampler, patch_tokens, pooled_language, spatial_bias=None):
    """spatial_bias: dict view -> [B,N] additive logit bias (negative suppresses)."""
    gamma,beta=resampler.film(pooled_language).unsqueeze(1).chunk(2,dim=-1)
    B=pooled_language.shape[0]; out=[]
    for i,name in enumerate(resampler.view_names):
        patches=resampler.input_projection(patch_tokens[name])+resampler.view_embedding[i].unsqueeze(0)
        q=resampler.queries[i].unsqueeze(0).expand(B,-1,-1)
        q=q*(1.0+gamma)+beta
        bias=None if spatial_bias is None else spatial_bias.get(name)
        tokens=_manual_mha(resampler.cross_attention,q,patches,patches,bias)
        out.append(resampler.norm(tokens))
    return torch.cat(out,dim=1)
