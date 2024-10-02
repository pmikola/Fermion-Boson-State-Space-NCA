import torch
from torch import nn


class NCA(nn.Module):
    def __init__(self, in_channels):
        super(NCA, self).__init__()
        self.nca_layer = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(3, 3),stride=1,padding=(1, 1))
        self.tanh = nn.Tanh()

    def nca_update(self, x, num_steps: int):
        for _ in range(num_steps):
            dx = self.tanh(self.nca_layer(x))
            x = x + dx
        return x

    def forward(self, x, num_steps: int):
        x = self.nca_update(x, num_steps)
        return x