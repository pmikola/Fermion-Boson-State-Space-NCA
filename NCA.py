import time

import torch
from performer_pytorch import Performer
from torch import nn
import torch_dct as dct
from torch_dct import dct_3d

class NCA(nn.Module):
    def __init__(self, in_channels,num_steps,device):
        super(NCA, self).__init__()
        self.device = device
        self.num_steps =num_steps
        self.fermion_number = 10
        self.boson_number = 10
        self.in_channels = in_channels
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
        self.kernel_size = 3
        self.fermionic_NCA = FermionConvLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=self.kernel_size)
        self.bosonic_NCA = BosonConvLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=self.kernel_size)

        self.fermion_features = Performer(dim=self.fermion_number,dim_head=self.fermion_number, depth=1, heads=self.fermion_number)
        self.boson_features = Performer(dim=self.boson_number,dim_head=self.boson_number, depth=1, heads=self.boson_number)

        self.project_fermions = nn.Linear(in_features=in_channels*15*15,out_features=self.fermion_number*self.fermion_number*self.kernel_size**2)
        self.project_bosons = nn.Linear(in_features=in_channels*15*15,out_features=self.boson_number*self.fermion_number*self.kernel_size**2)

        self.lnorm_fermion = nn.LayerNorm([in_channels, 15, 15])
        self.lnorm_boson = nn.LayerNorm([in_channels, 15, 15])

        self.learned_fermion_kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(self.fermion_number,in_channels, self.kernel_size, self.kernel_size), requires_grad=False) for _ in range(num_steps)])
        self.learned_boson_kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(self.boson_number,in_channels, self.kernel_size, self.kernel_size), requires_grad=False) for _ in range(num_steps)])

        self.act = nn.ELU(alpha=2.)
        # self.act = nn.GELU()
        self.step_param = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.residual_weights = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)

    def nca_update(self, x,meta_embeddings,spiking_probabilities):
        nca_var = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
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
        energy_spectrum = dct_3d(x)
        energy_spectrum = self.act(self.nca_fusion(energy_spectrum))+meta_embeddings
        for i in range(self.num_steps):
            if self.training:
                reshaped_energy_spectrum = energy_spectrum.contiguous().reshape(energy_spectrum.shape[0], -1, energy_spectrum.shape[1])
                fermion_kernels = self.fermion_features(reshaped_energy_spectrum)
                boson_kernels = self.boson_features(reshaped_energy_spectrum)
                fermion_kernels = self.project_fermions(fermion_kernels.flatten(start_dim=1))
                boson_kernels = self.project_bosons(boson_kernels.flatten(start_dim=1))
                #fermion_kernels, _ = torch.qr(fermion_kernels) # Note: check Orthogonality and if this is wanted behavior
                fermion_kernels = fermion_kernels.mean(dim=0).view(self.fermion_number,self.in_channels, self.kernel_size, self.kernel_size)
                boson_kernels=boson_kernels.mean(dim=0).view(self.fermion_number,self.in_channels, self.kernel_size, self.kernel_size)
                self.learned_fermion_kernels[i].data = fermion_kernels
                self.learned_boson_kernels[i].data = boson_kernels
            else:
                with torch.no_grad():
                    fermion_kernels = self.learned_fermion_kernels[i]
                    boson_kernels = self.learned_boson_kernels[i]
                fermionic_response = self.fermionic_NCA(energy_spectrum,weights=fermion_kernels)
                fermion_energy_states = self.act(self.lnorm_fermion(fermionic_response))
                bosonic_response = self.bosonic_NCA(fermion_energy_states, weights=boson_kernels)
                energy_spectrum = self.act(self.lnorm_boson(bosonic_response))
                nca_var[:,i] = torch.var(fermionic_response+bosonic_response, dim=[1 ,2, 3])
                energy_spectrum = (energy_spectrum + energy_spectrum * self.step_param[i] +
                     torch.rand_like(x) * spiking_probabilities[i]*self.spike_scale[i] +
                     energy_spectrum*self.residual_weights[i]) # Note: Progressing NCA dynamics by dx
        energy_spectrum = dct.idct_3d(x)
        return energy_spectrum,nca_var

    def forward(self, x,meta_embeddings,spiking_probabilities):
        x,nca_var = self.nca_update(x,meta_embeddings,spiking_probabilities)
        return x,nca_var

class FermionConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FermionConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.act = nn.ELU(alpha=2.)
        self.kernel_size= kernel_size

    def forward(self, x, weights=None):
        if weights is not None:
            return  self.act(torch.nn.functional.conv2d(x, weights, padding=self.kernel_size//2))
        else:
            return  self.act(self.conv(x))

class BosonConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BosonConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.act = nn.ELU(alpha=2.)
        self.kernel_size = kernel_size

    def forward(self, x, weights=None):
        if weights is not None:
            return  self.act(torch.nn.functional.conv2d(x, weights, padding=self.kernel_size//2))
        else:
            return  self.act(self.conv(x))