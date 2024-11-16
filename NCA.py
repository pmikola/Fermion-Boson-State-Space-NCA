import time

import torch
from linformer import Linformer
from matplotlib import pyplot as plt
from torch import nn
from WaveletModel import WaveletModel
from torch.nn.utils import spectral_norm as sn

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
        self.wavelet_scales = 10
        self.max_wavelet_scale = 20
        self.min_scale_value = self.max_wavelet_scale*0.01
        self.fermion_kernels_size = torch.arange(1, self.kernel_size + 1, 2)
        self.boson_kernels_size = torch.arange(1, self.kernel_size + 1, 2)

        self.nca_layers_odd = nn.ModuleList([
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1, padding=k // 2)
            for k in range(1, self.patch_size_x+2, 2 )
        ])

        self.nca_fusion = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0)
        self.fermionic_NCA = FermionConvLayer(channels=self.channels,propagation_steps = num_steps, kernel_size=self.kernel_size)
        self.bosonic_NCA = BosonConvLayer(channels=self.channels,propagation_steps = num_steps, kernel_size=self.kernel_size)

        self.wvl = WaveletModel(num_scales=self.wavelet_scales,min_scale_value=self.min_scale_value, max_scale_value=self.max_wavelet_scale,
                         batch_size=self.batch_size, channels=self.channels,num_steps=self.num_steps, height=self.patch_size_x, width=self.patch_size_y,device=self.device).to('cuda')

        self.fermion_features = Linformer(
            dim=self.channels*self.fermion_number*self.wavelet_scales,
            seq_len= self.patch_size_x * self.patch_size_y,
            depth=2,
            heads=self.fermion_number,
            dim_head =self.fermion_number,
            one_kv_head = False,
            share_kv = False,
            reversible = True,
            dropout = 0.00,
            k=self.patch_size_x * self.patch_size_y
        )

        self.boson_features = Linformer(
            dim=self.channels*self.boson_number*self.wavelet_scales,
            seq_len= self.patch_size_x * self.patch_size_y,
            depth=2,
            heads=self.boson_number,
            dim_head=self.boson_number,
            one_kv_head=False,
            share_kv=False,
            reversible=True,
            dropout=0.00,
            k=self.patch_size_x * self.patch_size_y
        )
        self.project_fermions_seq = nn.Conv1d( self.patch_size_x * self.patch_size_y, self.fermion_number  * torch.sum(self.fermion_kernels_size**3)*self.channels,kernel_size=1)
        self.project_bosons_seq = nn.Conv1d( self.patch_size_x * self.patch_size_y, self.boson_number  * torch.sum(self.boson_kernels_size**3)*self.channels,kernel_size=1)

        self.project_fermions_feature = nn.Conv1d(self.channels*self.fermion_number*self.wavelet_scales, self.channels,kernel_size=1)
        self.project_bosons_feature = nn.Conv1d(self.channels*self.boson_number*self.wavelet_scales,self.channels, kernel_size=1)

        self.common_nca_pool_layer_norm = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y]).to(self.device)
        self.lnorm_fermion = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y])
        self.lnorm_boson = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y])
        self.wvl_layer_norm = nn.LayerNorm(self.channels*self.particle_number*self.wavelet_scales).to(self.device)

        self.learned_fermion_kernels = nn.ParameterList([
            nn.ParameterList([nn.Parameter(torch.randn(self.fermion_number, self.channels, k, k, k), requires_grad=False)
                for k in range(1,self.kernel_size+1,2)])
            for _ in range(num_steps)])

        self.learned_boson_kernels = nn.ParameterList([
            nn.ParameterList([nn.Parameter(torch.randn(self.boson_number, self.channels, k, k, k), requires_grad=False)
                for k in range(1,self.kernel_size+1,2)])
            for _ in range(num_steps)])
        self.act = nn.ELU(alpha=2.)
        # self.act = nn.GELU()
        self.step_param = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.rand( self.num_steps), requires_grad=True)

    def nca_update(self, x,meta_embeddings,spiking_probabilities,hf_data,batch_size):
        self.batch_size = batch_size
        nca_var = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
        log_det_j_loss = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
        freq_loss = torch.zeros((self.fermion_kernels_size.shape[0],self.num_steps), requires_grad=True).to(self.device)

        nca_out_odd = [self.act(layer(x)) for layer in self.nca_layers_odd]
        # if self.training:
        #     freq_loss_nca = torch.mean(torch.stack([self.fft_high_frequency_loss(layer.weight,hf_data) for layer in self.nca_layers_odd],dim=0))
        # else:
        #     freq_loss_nca = None
        x = torch.sum(torch.stack(nca_out_odd), dim=0)
        x = self.common_nca_pool_layer_norm(x)
        energy_spectrum = self.act(self.nca_fusion(x))
        #energy_spectrum = dct_3d(x)
        for i in range(self.num_steps):
            if self.training:
                #reshaped_energy_spectrum = energy_spectrum.view(energy_spectrum.shape[0], self.patch_size_x * self.patch_size_y*self.channels, energy_spectrum.shape[1])
                reshaped_energy_spectrum = energy_spectrum.view(energy_spectrum.shape[0],self.channels**2, self.patch_size_x , self.patch_size_y)
                cwt_wvl = self.act(self.wvl(reshaped_energy_spectrum,i).permute(0,2,1))
                wavelet_space = self.wvl_layer_norm(cwt_wvl)

                fermion_kernels = self.act(self.fermion_features(wavelet_space))
                ortho_mean,ortho_max = self.validate_channel_orthogonality(fermion_kernels)

                boson_kernels = self.act(self.boson_features(wavelet_space))
                #ortho_mean_b, ortho_max_b = self.validate_channel_orthogonality(boson_kernels)
                #ortho_mean, ortho_max = ortho_mean_f+ortho_mean_b,ortho_max_f+ortho_max_b


                fermion_kernels = self.act(self.project_fermions_seq(fermion_kernels)).permute(0, 2, 1)
                boson_kernels = self.act(self.project_bosons_seq(boson_kernels)).permute(0, 2, 1)
                fermion_kernels = self.act(self.project_fermions_feature(fermion_kernels))
                boson_kernels = self.act(self.project_bosons_feature(boson_kernels))

                fermion_kernels = fermion_kernels.mean(dim=0)
                boson_kernels = boson_kernels.mean(dim=0)
                l_idx = 0
                f_kernels = []
                k_idx = 0
                for k_size in self.fermion_kernels_size:
                    l_idx = l_idx
                    h_idx = l_idx+(k_size**3).int()*self.channels
                    f_k = fermion_kernels[:, l_idx:h_idx].reshape(self.particle_number, self.channels, k_size.int(),
                                                            k_size.int(), k_size.int())
                    f_kernels.append(f_k)
                    freq_loss[k_idx,i] = self.fft_high_frequency_loss(f_k,hf_data) / f_k.numel()
                    k_idx +=1
                    l_idx = h_idx

                l_idx = 0
                b_kernels = []
                k_idx = 0
                for k_size in self.boson_kernels_size:
                    l_idx = l_idx
                    h_idx = l_idx+(k_size ** 3).int()*self.channels
                    b_k = boson_kernels[:, l_idx:h_idx].reshape(self.particle_number, self.channels, k_size.int(),
                                                          k_size.int(), k_size.int())
                    b_kernels.append(b_k)
                    freq_loss[k_idx, i] += self.fft_high_frequency_loss(b_k,hf_data) / b_k.numel()
                    k_idx += 1
                    l_idx = h_idx

                fermionic_response,f_log_det_jacobian = self.fermionic_NCA(energy_spectrum,meta_embeddings,i, weights=f_kernels)
                fermion_energy_states = self.act(self.lnorm_fermion(fermionic_response))
                bosonic_response,b_log_det_jacobian = self.bosonic_NCA(fermion_energy_states,meta_embeddings,i, weights=b_kernels)
                bosonic_energy_states = self.act(self.lnorm_boson(bosonic_response))
                energy_spectrum = energy_spectrum + (bosonic_energy_states * self.step_param[i] +
                                                     torch.rand_like(bosonic_energy_states) * spiking_probabilities[i] *
                                                     self.spike_scale[i])  # Note: Progressing NCA dynamics by dx
                nca_var[:, i] = torch.var(energy_spectrum, dim=[1, 2, 3, 4],unbiased=False)

                f_log_det_loss = self.fermionic_NCA.normalizing_flow_loss(fermionic_response,f_log_det_jacobian)
                b_log_det_loss = self.fermionic_NCA.normalizing_flow_loss(bosonic_response, b_log_det_jacobian)
                log_det_j_loss[:,i] = f_log_det_loss + b_log_det_loss
                #energy_spectrum = torch.clamp(energy_spectrum, min=self.clamp_low, max=self.clamp_high)
                with torch.no_grad():
                    for j in range(0,len(self.learned_fermion_kernels[i])):
                        self.learned_fermion_kernels[i][j].copy_(f_kernels[j])
                    for j in range(0, len(self.learned_boson_kernels[i])):
                        self.learned_boson_kernels[i][j].copy_(b_kernels[j])
            else:
                f_kernels = self.learned_fermion_kernels[i]
                b_kernels = self.learned_boson_kernels[i]
                fermionic_response,_ = self.fermionic_NCA(energy_spectrum,meta_embeddings,i,weights=f_kernels)
                fermion_energy_states = self.act(self.lnorm_fermion(fermionic_response))
                bosonic_response,_ = self.bosonic_NCA(fermion_energy_states,meta_embeddings,i, weights=b_kernels)
                bosonic_energy_states = self.act(self.lnorm_boson(bosonic_response))
                energy_spectrum = energy_spectrum + (bosonic_energy_states * self.step_param[i]) # Note: Progressing NCA dynamics by dx

                nca_var, ortho_mean, ortho_max,log_det_j_loss ,freq_loss= None,None,None,None,None
                #energy_spectrum = dct.idct_3d(energy_spectrum)
        return energy_spectrum,nca_var,ortho_mean,ortho_max,log_det_j_loss,freq_loss

    def forward(self, x,meta_embeddings,spiking_probabilities,hf_data,batch_size):
        x,nca_var,ortho_mean,ortho_max,log_det_jacobian,freq_loss = self.nca_update(x,meta_embeddings,spiking_probabilities,hf_data,batch_size)
        return x,nca_var,ortho_mean,ortho_max,log_det_jacobian,freq_loss

    def fft_high_frequency_loss(self,kernels,hf_data, cutoff_ratio=0.7):
        C,HDC, H, W, D = kernels.shape
        fft_k = torch.fft.fftn(kernels)
        fft_k_shifted = torch.fft.fftshift(fft_k)
        mask_lf = torch.zeros((C, HDC, H, W, D), device=self.device)
        center_ch,center_hdc,center_x, center_y, center_z = (C-1)//2, (HDC-1)// 2, (H-1) // 2, (W -1)// 2,( D -1)// 2
        cutoff_ch = int(cutoff_ratio * center_ch)
        cutoff_hdc = int(cutoff_ratio * center_hdc)
        cutoff_x = int(cutoff_ratio * center_x)
        cutoff_y = int(cutoff_ratio * center_y)
        cutoff_z = int(cutoff_ratio * center_z)
        mask_lf[center_ch - cutoff_ch:center_ch + cutoff_ch,center_hdc - cutoff_hdc:center_hdc + cutoff_hdc,center_x - cutoff_x:center_x + cutoff_x, center_y - cutoff_y:center_y + cutoff_y,center_z - cutoff_z:center_z + cutoff_z] = 1.
        high_freq_k = fft_k_shifted * ( 1- mask_lf)
        low_freq_k = fft_k_shifted * mask_lf
        hf_mean = torch.abs(high_freq_k).mean()
        lf_mean = torch.abs(low_freq_k).mean()
        # loss = (hf_mean / (1+lf_mean+hf_data))
        loss = (hf_mean - hf_data) ** 2 + torch.relu(3 * lf_mean - hf_mean)
        return loss

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
        self.fermion_gate_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.fermion_gate_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        # Note: Normalising flow
        self.scale_net = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.shift_net = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        #self.threshold = nn.Parameter(torch.rand(propagation_steps))
        self.act = nn.ELU(alpha=2.)
        self.kernel_size= kernel_size

    def normalizing_flow_loss(self,z, log_det_jacobian):
        log_p_z = -0.5 * torch.sum(z ** 2, dim=[1, 2, 3, 4])
        log_p_z -= 0.5 * z[0].numel() * torch.log(torch.tensor(2 * torch.pi))
        loss = -log_p_z + log_det_jacobian
        return loss

    def forward(self, x,meta,idx, weights):
        w_len = len(weights)
        scale = torch.sigmoid(self.scale_net(x))
        shift = self.shift_net(x)
        transformed_x = scale * x + shift
        log_det_jacobian = torch.sum(torch.log(scale), dim=[1, 2, 3, 4])
        f_o = [self.act(torch.nn.functional.conv3d(transformed_x, w, padding=w.shape[-1]//2 )) for w in weights]
        f_o = torch.sum(torch.stack(f_o), dim=0)*(1 / w_len)
        #f_o = self.act(torch.nn.functional.conv3d(x, weights, padding=self.kernel_size//2 ))
        gating_input = torch.mean(x+meta.unsqueeze(1), dim=[2, 3], keepdim=True)
        gate = self.act(self.fermion_gate_0(gating_input))
        #gate = torch.where(gate > self.threshold[idx], gate, torch.zeros_like(gate))
        gate = torch.sigmoid(self.fermion_gate_1(gate)) # Note: Inhibit
        gated_output = f_o * gate
        return gated_output,log_det_jacobian


class BosonConvLayer(nn.Module):
    def __init__(self, channels,propagation_steps, kernel_size=3):
        super(BosonConvLayer, self).__init__()
        self.boson_gate_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.boson_gate_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.scale_net = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.shift_net = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        #self.threshold = nn.Parameter(torch.rand(propagation_steps))
        self.act = nn.ELU(alpha=2.)
        self.kernel_size = kernel_size

    def normalizing_flow_loss(self,z, log_det_jacobian):
        log_p_z = -0.5 * torch.sum(z ** 2, dim=[1, 2, 3, 4])
        log_p_z -= 0.5 * z[0].numel() * torch.log(torch.tensor(2 * torch.pi))
        loss = -log_p_z + log_det_jacobian
        return loss

    def forward(self, x,meta,idx, weights):
        w_len = len(weights)
        scale = torch.sigmoid(self.scale_net(x))
        shift = self.shift_net(x)
        transformed_x = scale * x + shift
        log_det_jacobian = torch.sum(torch.log(scale), dim=[1, 2, 3, 4])
        b_o = [self.act(torch.nn.functional.conv3d(transformed_x, w, padding=w.shape[-1] // 2)) for w in weights]
        b_o = torch.sum(torch.stack(b_o), dim=0)*(1 / w_len)
        gating_input = torch.mean(x+meta.unsqueeze(1), dim=[2, 3], keepdim=True)
        gate = self.act(self.boson_gate_0(gating_input))
        #gate = torch.where(gate > self.threshold[idx], gate, torch.zeros_like(gate))
        gate = self.act(self.boson_gate_1(gate))
        gated_output = b_o * gate
        return gated_output,log_det_jacobian

