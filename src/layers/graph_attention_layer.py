import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_mechanism import AttentionMechanism


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = AttentionMechanism(out_dim, alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        Wh = self.W(x)

        e = self.attention(Wh)

        mask = (adj > 0)
        e = e.masked_fill(~mask, float("-inf"))

        alpha = torch.softmax(e, dim=1)
        alpha = self.dropout(alpha)

        h_prime = alpha @ Wh

        return h_prime
