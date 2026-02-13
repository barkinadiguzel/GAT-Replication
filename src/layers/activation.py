import torch.nn as nn


def get_activation(name="leakyrelu", negative_slope=0.2):
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU(negative_slope)
    elif name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "elu":
        return nn.ELU()
    else:
        return nn.Identity()
