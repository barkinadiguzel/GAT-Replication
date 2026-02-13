import torch.nn as nn
from ..layers.multi_head import MultiHeadGAT


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()

        self.layer1 = MultiHeadGAT(in_dim, hidden_dim, heads, concat=True)
        self.layer2 = MultiHeadGAT(hidden_dim * heads, out_dim, 1, concat=False)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x
