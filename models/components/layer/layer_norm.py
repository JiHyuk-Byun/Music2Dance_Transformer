"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, hid_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hid_dim))
        self.beta = nn.Parameter(torch.zeros(hid_dim))
        self.eps = eps

    def forward(self, x):
        # 'x': [batch, sequence length, hid_dim]: 고정된 hidden dimension 사이즈에 대해 input 크기에 관계없이 구할 수 있다.
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        # Normalize
        out = (x - mean) / torch.sqrt(var + self.eps)
        # Learnable distribution
        out = self.gamma * out + self.beta
        return out
