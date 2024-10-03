import torch
from torch import nn


class NCA(nn.Module):
    def __init__(self, in_channels,num_steps):
        super(NCA, self).__init__()
        self.num_steps =num_steps
        self.nca_layer_3 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(3, 3),stride=1,padding=(1, 1))
        self.nca_layer_3_dil = nn.Conv2d(in_channels=in_channels,out_channels=in_channels, kernel_size=(3, 3), dilation=2,padding=2)
        #self.nca_layer_5 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(5, 5),stride=1,padding=(2, 2))
        #self.nca_layer_7 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(7, 7),stride=1,padding=(3, 3))
        self.nca_fusion = nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
        self.act = nn.LeakyReLU(0.1)
        # self.act = nn.Tanh()
        self.step_param = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)


    def nca_update(self, x,spiking_probabilities):
        step_weight = torch.sigmoid(self.step_param)
        for i in range(self.num_steps):
            dx_3 = self.act(self.nca_layer_3(x))
            dx_3_dil = self.act(self.nca_layer_3_dil(x))
            # dx_5 = self.act(self.nca_layer_5(x))
            # dx_7 = self.act(self.nca_layer_7(x))
            dx = torch.cat([dx_3,dx_3_dil],dim=1)
            dx = self.act(self.nca_fusion(dx))
            x = x + dx * step_weight[i] + torch.rand_like(x) * spiking_probabilities[i]*self.spike_scale[i]
            # TODO: Create loss value that validate criticality of this NCA network
        return x

    def forward(self, x,spiking_probabilities):
        x = self.nca_update(x,spiking_probabilities)
        return x