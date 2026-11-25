"""Load SmolVLA only after confirmation; print SigLIP module and layer shapes."""
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p='/hy-tmp/smolvla_ckpt/HuggingFaceVLA_smolvla_libero'
policy=SmolVLAPolicy.from_pretrained(p).eval()
vlm=policy.model.vlm_with_expert.vlm.model.vision_model
print('vision module:',vlm)
print('depth:',len(vlm.encoder.layers))
for i,l in enumerate(vlm.encoder.layers): print('layer',i,type(l).__name__)
# expected [B,1024,768] per image; no model forward here
