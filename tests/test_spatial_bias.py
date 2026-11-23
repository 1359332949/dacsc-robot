import torch
from corp.representation.cross_attention_resampler import CrossAttentionResampler
from dacsc.spatial_patch_bias import resampler_forward_with_bias,boundary_bg_weights
torch.manual_seed(0)
r=CrossAttentionResampler(view_names=['agentview','eye_in_hand'],hidden_dim=768,tokens_per_view=48,num_heads=12,film_rank=64).eval()
p={'agentview':torch.randn(2,256,384),'eye_in_hand':torch.randn(2,256,384)}; l=torch.randn(2,768)
with torch.no_grad(): a=r(p,l); b=resampler_forward_with_bias(r,p,l)
print('parity',float((a-b).abs().max()))
w=boundary_bg_weights(p['agentview']); print('bg stats',float(w.min()),float(w.mean()),float(w.max()),w.shape)
assert torch.allclose(a,b,atol=1e-5)
