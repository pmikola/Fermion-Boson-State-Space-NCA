import time
import torch
from linformer import Linformer
# from transformers import XTransformer
from torch import nn
import torch_dct as dct
from torch_dct import dct_3d

class NCA(nn.Module):
    def __init__(self,batch_size, channels, num_steps, device):
        super(NCA, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_steps =num_steps
        self.particle_number = channels
        self.fermion_number =  self.particle_number
        self.boson_number =  self.particle_number
        self.patch_size_x = 15
        self.patch_size_y = 15
        self.clamp_low,self.clamp_high = -2.,2.
        self.channels = channels
        self.nca_layer_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1)
        self.nca_layer_3 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        self.nca_layer_3_dil = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, dilation=2, padding=2)
        self.nca_layer_5 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=5, stride=1, padding=2)
        self.nca_layer_7 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=7, stride=1, padding=3)
        self.nca_layer_9 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=9, stride=1, padding=4)
        self.nca_layer_11 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=11, stride=1, padding=5)
        self.nca_layer_13 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=13, stride=1, padding=6)
        self.nca_layer_15 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=15, stride=1, padding=7)

        self.nca_fusion = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0)
        self.kernel_size = 3
        self.fermionic_NCA = FermionConvLayer(channels=self.channels, kernel_size=self.kernel_size)
        self.bosonic_NCA = BosonConvLayer(channels=self.channels, kernel_size=self.kernel_size)

        # self.particle_features = Performer(dim=self.particle_number, dim_head=self.particle_number, depth=1, heads=self.particle_number)

        self.fermion_features = Linformer(
            dim=self.fermion_number,
            seq_len=self.patch_size_x * self.patch_size_y,
            depth=1,
            heads=self.fermion_number,
            dim_head =self.fermion_number//2,
            one_kv_head = False,
            share_kv = False,
            reversible = True,
            dropout = 0.05,
            k=self.patch_size_x * self.patch_size_y
        )

        self.boson_features = Linformer(
            dim=self.boson_number,
            seq_len=self.patch_size_x * self.patch_size_y,
            depth=1,
            heads=self.boson_number,
            dim_head=self.boson_number // 2,
            one_kv_head=False,
            share_kv=False,
            reversible=True,
            dropout=0.05,
            k=self.patch_size_x * self.patch_size_y
        )

        self.project_fermions = nn.Linear(in_features=self.channels * self.patch_size_x * self.patch_size_y, out_features=self.fermion_number * self.fermion_number * self.kernel_size ** 2)
        self.project_bosons = nn.Linear(in_features=self.channels * self.patch_size_x * self.patch_size_y, out_features=self.boson_number * self.fermion_number * self.kernel_size ** 2)

        self.lnorm_fermion = nn.LayerNorm([self.channels, self.patch_size_x, self.patch_size_y])
        self.lnorm_boson = nn.LayerNorm([self.channels, self.patch_size_x, self.patch_size_y])

        self.learned_fermion_kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(self.fermion_number, self.channels, self.kernel_size, self.kernel_size), requires_grad=False) for _ in range(num_steps)])
        self.learned_boson_kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(self.boson_number, self.channels, self.kernel_size, self.kernel_size), requires_grad=False) for _ in range(num_steps)])

        self.act = nn.ELU(alpha=2.)
        # self.act = nn.GELU()
        self.step_param = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.residual_weights = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)

    def nca_update(self, x,meta_embeddings,spiking_probabilities,batch_size):
        self.batch_size = batch_size
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
        x = x_1 + x_3 + x_3_dil + x_5 + x_7 + x_9 + x_11 + x_13 + x_15 # Note: Bosonic behavior (superposition)
        #energy_spectrum = dct_3d(x)
        energy_spectrum = self.act(self.nca_fusion(x))+meta_embeddings
        for i in range(self.num_steps):
            if self.training:
                reshaped_energy_spectrum = energy_spectrum.view(energy_spectrum.shape[0], -1, energy_spectrum.shape[1])
                fermion_kernels = self.act(self.fermion_features(reshaped_energy_spectrum))
                ortho_mean,ortho_max = self.validate_channel_orthogonality(fermion_kernels)
                fermion_kernels = fermion_kernels.flatten(start_dim=1)

                boson_kernels = self.act(self.boson_features(reshaped_energy_spectrum))
                boson_kernels = boson_kernels.flatten(start_dim=1)

                fermion_kernels = self.act(self.project_fermions(fermion_kernels))
                boson_kernels = self.act(self.project_bosons(boson_kernels))

                fermion_kernels = fermion_kernels.mean(dim=0).view(self.particle_number,self.channels, self.kernel_size, self.kernel_size)
                boson_kernels = boson_kernels.mean(dim=0).view(self.particle_number,self.channels, self.kernel_size, self.kernel_size)
                
                fermionic_response = self.fermionic_NCA(energy_spectrum,meta_embeddings, weights=fermion_kernels)
                fermion_energy_states = self.act(self.lnorm_fermion(fermionic_response))

                bosonic_response = self.bosonic_NCA(fermion_energy_states,meta_embeddings, weights=boson_kernels)
                bosonic_energy_states = self.act(self.lnorm_boson(bosonic_response))

                energy_spectrum = energy_spectrum + (bosonic_energy_states * self.step_param[i] +
                                                     torch.rand_like(bosonic_energy_states) * spiking_probabilities[i] *
                                                     self.spike_scale[i] +
                                                     bosonic_energy_states * self.residual_weights[
                                                         i])  # Note: Progressing NCA dynamics by dx
                nca_var[:, i] = torch.var(energy_spectrum, dim=[1, 2, 3],unbiased=False)
                #energy_spectrum = torch.clamp(energy_spectrum, min=self.clamp_low, max=self.clamp_high)
                with torch.no_grad():
                    self.learned_fermion_kernels[i].copy_(fermion_kernels)
                    self.learned_boson_kernels[i].copy_(boson_kernels)
            else:
                fermion_kernels = self.learned_fermion_kernels[i]
                boson_kernels = self.learned_boson_kernels[i]
                fermionic_response = self.fermionic_NCA(energy_spectrum,meta_embeddings,weights=fermion_kernels)
                fermion_energy_states = self.act(self.lnorm_fermion(fermionic_response))
                bosonic_response = self.bosonic_NCA(fermion_energy_states,meta_embeddings, weights=boson_kernels)
                bosonic_energy_states = self.act(self.lnorm_boson(bosonic_response))
                energy_spectrum = energy_spectrum + (bosonic_energy_states * self.step_param[i] +
                     torch.rand_like(bosonic_energy_states) * spiking_probabilities[i]*self.spike_scale[i] +
                     bosonic_energy_states*self.residual_weights[i]) # Note: Progressing NCA dynamics by dx
                #energy_spectrum = torch.clamp(energy_spectrum, min=self.clamp_low, max=self.clamp_high)
                nca_var, ortho_mean, ortho_max = None,None,None
                #energy_spectrum = dct.idct_3d(energy_spectrum)
        return energy_spectrum,nca_var,ortho_mean,ortho_max

    def forward(self, x,meta_embeddings,spiking_probabilities,batch_size):
        x,nca_var,ortho_mean,ortho_max = self.nca_update(x,meta_embeddings,spiking_probabilities,batch_size)
        return x,nca_var,ortho_mean,ortho_max

    def validate_channel_orthogonality(self,particles):
        particles = particles.view(self.batch_size, self.channels, -1)
        dot_products = torch.bmm(particles, particles.transpose(1, 2))
        identity_mask = torch.eye(self.channels, device=particles.device).unsqueeze(0)
        identity_mask = identity_mask.expand(self.batch_size, -1, -1)
        off_diagonal_elements = dot_products * (1 - identity_mask)
        mean_off_diagonal = torch.mean(torch.abs(off_diagonal_elements), dim=(1, 2))
        max_off_diagonal = torch.amax(torch.abs(off_diagonal_elements), dim=(1, 2))
        ortho_mean = torch.mean(mean_off_diagonal)
        ortho_max = torch.mean(max_off_diagonal)

        return ortho_mean, ortho_max


class FermionConvLayer(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FermionConvLayer, self).__init__()
        self.fermion_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)  # Convolutional gate
        self.act = nn.ELU(alpha=2.)
        self.kernel_size= kernel_size

    def forward(self, x,meta, weights):
        f_o = self.act(torch.nn.functional.conv2d(x, weights, padding=self.kernel_size // 2))
        gating_input = torch.mean(x+meta, dim=[2, 3], keepdim=True)
        gate = torch.sigmoid(self.fermion_gate(gating_input))
        gated_output = f_o * gate
        return gated_output


class BosonConvLayer(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(BosonConvLayer, self).__init__()
        self.boson_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.ELU(alpha=2.)
        self.kernel_size = kernel_size

    def forward(self, x,meta, weights):
        b_o = self.act(torch.nn.functional.conv2d(x, weights, padding=self.kernel_size // 2))
        gating_input = torch.mean(x+meta, dim=[2, 3], keepdim=True)
        gate = torch.sigmoid(self.boson_gate(gating_input))
        gated_output = b_o * gate
        return gated_output