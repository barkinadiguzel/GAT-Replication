import torch.nn as nn
from ..encoder.gat_encoder import GATEncoder
from ..encoder.readout import global_mean_pool


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.encoder = GATEncoder(in_dim, hidden_dim, out_dim)

    def forward(self, x, adj, graph_level=False):
        node_emb = self.encoder(x, adj)

        if graph_level:
            return global_mean_pool(node_emb)

        return node_emb
