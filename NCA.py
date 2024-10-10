import torch
from torch import nn


class NCA(nn.Module):
    def __init__(self, in_channels,num_steps,device):
        super(NCA, self).__init__()
        self.device = device
        self.num_steps =num_steps
        self.nca_layer_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.nca_layer_3 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.nca_layer_3_dil = nn.Conv2d(in_channels=in_channels,out_channels=in_channels, kernel_size=3, dilation=2,padding=2)
        self.nca_layer_5 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(5, 5),stride=1,padding=(2, 2))
        self.nca_layer_7 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(7, 7),stride=1,padding=(3, 3))
        self.nca_fusion = nn.Conv2d(in_channels=4*in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
        self.layer_norm_1 = nn.LayerNorm([in_channels, 15, 15])
        self.act = nn.ELU(alpha=1.0)
        # self.act = nn.GELU()
        self.step_param = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.residual_weights = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)

    def nca_update(self, x,spiking_probabilities):
        nca_var = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
        dx_3 = self.act(self.nca_layer_3(x))
        dx_3_dil = self.act(self.nca_layer_3_dil(x))
        dx_5 = self.act(self.nca_layer_5(x))
        dx_7 = self.act(self.nca_layer_7(x))
        x = torch.cat([dx_3,dx_3_dil,dx_5,dx_7], dim=1)
        x = self.act(self.nca_fusion(x))
        for i in range(self.num_steps):
            dx = self.act(self.layer_norm_1(self.nca_layer_1(x)))
            nca_var[:,i] = torch.var(dx, dim=[1 ,2, 3])
            x = (x + dx * self.step_param[i] +
                 torch.rand_like(x) * spiking_probabilities[i]*self.spike_scale[i] +
                 x*self.residual_weights[i])
        return x,nca_var

    def forward(self, x,spiking_probabilities):
        x,nca_var = self.nca_update(x,spiking_probabilities)
        return x,nca_var