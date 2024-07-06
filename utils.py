import torch
import torch.nn as nn

# contrastive loss has been adopted from HF -> 
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L53
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
