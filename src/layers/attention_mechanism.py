import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):
    def __init__(self, out_dim, negative_slope=0.2):
        super().__init__()
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a.data)
        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, Wh):
        N = Wh.size(0)

        Wh_i = Wh.repeat_interleave(N, dim=0)
        Wh_j = Wh.repeat(N, 1)

        e = torch.cat([Wh_i, Wh_j], dim=1)
        e = self.leakyrelu(e @ self.a)

        return e.view(N, N)
