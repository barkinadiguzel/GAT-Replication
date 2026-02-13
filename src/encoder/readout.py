import torch


def global_mean_pool(x):
    return torch.mean(x, dim=0, keepdim=True)
