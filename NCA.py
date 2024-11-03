import time
import torch
from linformer import Linformer
# from transformers import XTransformer
from torch import nn
import torch_dct as dct
from torch.distributions.constraints import symmetric
from torch_dct import dct_3d
from pytorch_wavelets import DWTForward
import torch.nn.functional as F

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
        self.channels = channels
        self.kernel_size = 3
        self.target_seq_len = 243
        self.feature_dim = 4 * self.channels ** 2 # 4 components of wavelet dwt transform

        self.nca_layers = nn.ModuleList([
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1, padding=k // 2)
            for k in [1, 3, 5, 7, 9, 11, 13, 15]
        ])
        self.nca_fusion = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0)
        self.fermionic_NCA = FermionConvLayer(channels=self.channels,propagation_steps = num_steps, kernel_size=self.kernel_size)
        self.bosonic_NCA = BosonConvLayer(channels=self.channels,propagation_steps = num_steps, kernel_size=self.kernel_size)

        # self.particle_features = Performer(dim=self.particle_number, dim_head=self.particle_number, depth=1, heads=self.particle_number)
        self.dwt_space = DWTForward(J=1, wave='db2', mode='periodization').to(self.device)
        self.dwt_freq = DWTForward(J=5, wave='coif1', mode='periodization').to(self.device)

        self.fermion_features = Linformer(
            dim=self.feature_dim,
            seq_len= self.target_seq_len,
            depth=2,
            heads=2*self.fermion_number,
            dim_head =2*self.fermion_number,
            one_kv_head = False,
            share_kv = False,
            reversible = True,
            dropout = 0.05,
            k=self.patch_size_x * self.patch_size_y
        )

        self.boson_features = Linformer(
            dim=self.feature_dim,
            seq_len= self.target_seq_len,
            depth=2,
            heads=2*self.boson_number,
            dim_head=2*self.boson_number,
            one_kv_head=False,
            share_kv=False,
            reversible=True,
            dropout=0.05,
            k=self.patch_size_x * self.patch_size_y
        )

        self.project_fermions_seq = nn.Conv1d(self.target_seq_len, self.fermion_number  * self.kernel_size ** 3,kernel_size=1)
        self.project_bosons_seq = nn.Conv1d(self.target_seq_len, self.boson_number  * self.kernel_size ** 3,kernel_size=1)

        self.project_fermions_feature = nn.Conv1d(self.feature_dim, 5,kernel_size=1)
        self.project_bosons_feature = nn.Conv1d(self.feature_dim,5, kernel_size=1)

        self.lnorm_fermion = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y])
        self.lnorm_boson = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y])
        self.wvl_layer_norm = nn.LayerNorm(self.feature_dim).to(self.device)

        self.learned_fermion_kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(self.fermion_number,self.channels, self.kernel_size, self.kernel_size, self.kernel_size), requires_grad=False) for _ in range(num_steps)])
        self.learned_boson_kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(self.boson_number,self.channels, self.kernel_size, self.kernel_size, self.kernel_size), requires_grad=False) for _ in range(num_steps)])

        self.act = nn.ELU(alpha=2.)
        # self.act = nn.GELU()
        self.step_param = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)

    def nca_update(self, x,meta_embeddings,spiking_probabilities,batch_size):
        self.batch_size = batch_size
        nca_var = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
        nca_out = [self.act(layer(x)) for layer in self.nca_layers]
        x = torch.sum(torch.stack(nca_out), dim=0)
        energy_spectrum = self.act(self.nca_fusion(x))
        #energy_spectrum = dct_3d(x)
        for i in range(self.num_steps):
            if self.training:
                #reshaped_energy_spectrum = energy_spectrum.view(energy_spectrum.shape[0], self.patch_size_x * self.patch_size_y*self.channels, energy_spectrum.shape[1])
                reshaped_energy_spectrum = energy_spectrum.view(energy_spectrum.shape[0],self.channels**2, self.patch_size_x , self.patch_size_y)
                wvl_low,wvl_high = self.dwt(reshaped_energy_spectrum)
                wh_0 = wvl_high[0].flatten(start_dim=2)
                wh_1 = wvl_high[1].flatten(start_dim=2)
                wh_2 = wvl_high[2].flatten(start_dim=2)
                wl_0 = wvl_low.flatten(start_dim=2)
                wh_1 = F.pad(wh_1,(0, self.target_seq_len - wh_1.size(-1)))
                wh_2 = F.pad(wh_2,(0, self.target_seq_len - wh_2.size(-1)))
                wl_0 = F.pad(wl_0,(0, self.target_seq_len - wl_0.size(-1)))
                wavelet_space = torch.cat([wl_0,wh_0,wh_1,wh_2],dim=1).permute(0, 2, 1)
                wavelet_space = self.wvl_layer_norm(wavelet_space)
                fermion_kernels = self.act(self.fermion_features(wavelet_space))
                ortho_mean,ortho_max = self.validate_channel_orthogonality(fermion_kernels)
                # fermion_kernels = fermion_kernels.flatten(start_dim=1)
                boson_kernels = self.act(self.boson_features(wavelet_space))
                # boson_kernels = boson_kernels.flatten(start_dim=1)

                fermion_kernels = self.act(self.project_fermions_seq(fermion_kernels)).permute(0, 2, 1)
                boson_kernels = self.act(self.project_bosons_seq(boson_kernels)).permute(0, 2, 1)
                fermion_kernels = self.act(self.project_fermions_feature(fermion_kernels))
                boson_kernels = self.act(self.project_bosons_feature(boson_kernels))
                # fermion_kernels = torch.sum(torch.nn.functional.softplus(fermion_kernels), dim=0).view(self.particle_number,self.channels, self.kernel_size, self.kernel_size)
                # boson_kernels = torch.sum(torch.nn.functional.softplus(boson_kernels), dim=0).view(self.particle_number,self.channels, self.kernel_size, self.kernel_size)
                fermion_kernels = fermion_kernels.mean(dim=0).view(self.particle_number,self.channels,self.kernel_size, self.kernel_size, self.kernel_size)
                boson_kernels = boson_kernels.mean(dim=0).view(self.particle_number,self.channels,self.kernel_size, self.kernel_size, self.kernel_size)
                fermionic_response = self.fermionic_NCA(energy_spectrum,meta_embeddings,i, weights=fermion_kernels)
                fermion_energy_states = self.act(self.lnorm_fermion(fermionic_response))
                bosonic_response = self.bosonic_NCA(fermion_energy_states,meta_embeddings,i, weights=boson_kernels)
                bosonic_energy_states = self.act(self.lnorm_boson(bosonic_response))

                energy_spectrum = energy_spectrum + (bosonic_energy_states * self.step_param[i] +
                                                     torch.rand_like(bosonic_energy_states) * spiking_probabilities[i] *
                                                     self.spike_scale[i])  # Note: Progressing NCA dynamics by dx
                nca_var[:, i] = torch.var(energy_spectrum, dim=[1, 2, 3, 4],unbiased=False)
                #energy_spectrum = torch.clamp(energy_spectrum, min=self.clamp_low, max=self.clamp_high)
                with torch.no_grad():
                    self.learned_fermion_kernels[i].copy_(fermion_kernels)
                    self.learned_boson_kernels[i].copy_(boson_kernels)
            else:
                fermion_kernels = self.learned_fermion_kernels[i]
                boson_kernels = self.learned_boson_kernels[i]
                fermionic_response = self.fermionic_NCA(energy_spectrum,meta_embeddings,i,weights=fermion_kernels)
                fermion_energy_states = self.act(self.lnorm_fermion(fermionic_response))
                bosonic_response = self.bosonic_NCA(fermion_energy_states,meta_embeddings,i, weights=boson_kernels)
                bosonic_energy_states = self.act(self.lnorm_boson(bosonic_response))
                energy_spectrum = energy_spectrum + (bosonic_energy_states * self.step_param[i]) # Note: Progressing NCA dynamics by dx

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
    def __init__(self, channels,propagation_steps, kernel_size=3):
        super(FermionConvLayer, self).__init__()
        self.fermion_gate_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)  # Convolutional gate
        self.fermion_gate_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)  # Convolutional gate
        #self.threshold = nn.Parameter(torch.rand(propagation_steps))
        self.act = nn.ELU(alpha=2.)
        self.kernel_size= kernel_size

    def forward(self, x,meta,idx, weights):
        f_o = self.act(torch.nn.functional.conv3d(x, weights, padding=self.kernel_size//2 ))
        gating_input = torch.mean(x+meta.unsqueeze(1), dim=[2, 3], keepdim=True)
        gate = self.act(self.fermion_gate_0(gating_input))
        #gate = torch.where(gate > self.threshold[idx], gate, torch.zeros_like(gate))
        gate = torch.sigmoid(self.fermion_gate_1(gate))
        gated_output = f_o * gate
        return gated_output


class BosonConvLayer(nn.Module):
    def __init__(self, channels,propagation_steps, kernel_size=3):
        super(BosonConvLayer, self).__init__()
        self.boson_gate_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.boson_gate_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        #self.threshold = nn.Parameter(torch.rand(propagation_steps))
        self.act = nn.ELU(alpha=2.)
        self.kernel_size = kernel_size

    def forward(self, x,meta,idx, weights):
        b_o = self.act(torch.nn.functional.conv3d(x, weights, padding=self.kernel_size // 2))
        gating_input = torch.mean(x+meta.unsqueeze(1), dim=[2, 3], keepdim=True)
        gate = self.act(self.boson_gate_0(gating_input))
        #gate = torch.where(gate > self.threshold[idx], gate, torch.zeros_like(gate))
        gate = self.act(self.boson_gate_1(gate))
        gated_output = b_o * gate
        return gated_output