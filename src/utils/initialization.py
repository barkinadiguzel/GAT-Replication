import torch.nn as nn


def glorot_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


def attention_init(param):
    nn.init.xavier_uniform_(param.data)
