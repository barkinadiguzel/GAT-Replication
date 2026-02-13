import torch
import torch.nn as nn
from .graph_attention_layer import GraphAttentionLayer


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, concat=True):
        super().__init__()
        self.concat = concat

        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, out_dim)
            for _ in range(heads)
        ])

    def forward(self, x, adj):
        outputs = [head(x, adj) for head in self.heads]

        if self.concat:
            return torch.cat(outputs, dim=1)
        else:
            return torch.mean(torch.stack(outputs), dim=0)
