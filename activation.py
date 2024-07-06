import torch
import torch.nn as nn

# Refer transformers implementation of Quick GELU at 
# https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)