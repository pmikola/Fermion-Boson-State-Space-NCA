import torch
from torch import nn
import torch_dct as dct
from torch_dct import dct_3d


class NCA(nn.Module):
    def __init__(self, in_channels,num_steps,device):
        super(NCA, self).__init__()
        self.device = device
        self.num_steps =num_steps
        self.nca_layer_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.nca_layer_3 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.nca_layer_3_dil = nn.Conv2d(in_channels=in_channels,out_channels=in_channels, kernel_size=3, dilation=2,padding=2)
        self.nca_layer_5 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=5,stride=1,padding=2)
        self.nca_layer_7 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=7,stride=1,padding=3)
        self.nca_layer_9 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=9,stride=1,padding=4)
        self.nca_layer_11 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=11,stride=1,padding=5)
        self.nca_layer_13 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=13,stride=1,padding=6)
        self.nca_layer_15 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=15,stride=1,padding=7)

        self.nca_fusion = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
        self.evolve_nca_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.layer_norm = nn.LayerNorm([in_channels, 15, 15])
        self.act = nn.ELU(alpha=2.)
        # self.act = nn.GELU()
        self.step_param = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.residual_weights = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)


    def nca_update(self, x,meta_embeddings,spiking_probabilities):
        nca_var = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
        x = dct_3d(x)
        x_1 = self.act(self.nca_layer_1(x))
        x_3 = self.act(self.nca_layer_3(x))
        x_3_dil = self.act(self.nca_layer_3_dil(x))
        x_5 = self.act(self.nca_layer_5(x))
        x_7 = self.act(self.nca_layer_7(x))
        x_9 = self.act(self.nca_layer_9(x))
        x_11 = self.act(self.nca_layer_11(x))
        x_13 = self.act(self.nca_layer_13(x))
        x_15 = self.act(self.nca_layer_15(x))
        x = x_1 + x_3 + x_3_dil + x_5 + x_7 + x_9 + x_11 + x_13 + x_15 # Note: Bosonic response (superposition)
        x = self.act(self.nca_fusion(x))+meta_embeddings
        for i in range(self.num_steps):
            dx = self.act(self.layer_norm(self.evolve_nca_layer(x)))
            nca_var[:,i] = torch.var(dx, dim=[1 ,2, 3])
            x = (x + dx * self.step_param[i] +
                 torch.rand_like(x) * spiking_probabilities[i]*self.spike_scale[i] +
                 x*self.residual_weights[i]) # Note: Progressing NCA dynamics by dx
        x = dct.idct_3d(x)
        return x,nca_var

    def forward(self, x,meta_embeddings,spiking_probabilities):
        x,nca_var = self.nca_update(x,meta_embeddings,spiking_probabilities)
        return x,nca_var