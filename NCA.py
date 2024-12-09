import time

import matplotlib
import numpy as np
import torch
from linformer import Linformer
from matplotlib import pyplot as plt, animation
from scipy.constants import fermi
from torch import nn
from WaveletModel import WaveletModel
from torch.nn.utils import spectral_norm as sn
matplotlib.use('TkAgg')
# print(matplotlib.get_backend())
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
        self.wavelet_scales = 7
        self.max_wavelet_scale = 20
        self.min_scale_value = self.max_wavelet_scale*0.01
        self.fermion_kernels_size = torch.arange(1, self.kernel_size + 1, 2)
        self.boson_kernels_size = torch.arange(1, self.kernel_size + 1, 2)
        self.act = nn.ELU(alpha=1.)

        # self.fig3d = plt.figure(figsize=(6, 6))
        # self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.fig2d, self.axs2d = plt.subplots(6, self.num_steps*2, figsize=(12, 7))
        self.fig2d.subplots_adjust(
            left=0.01,
            right=0.99,
            top=0.99,
            bottom=0.01,
            wspace=0.02,
            hspace=0.02
        )
        self.fig2d.show()
        # self.fig2d.tight_layout()
        self.ims = []
        self.cbar = None

        self.nca_layers_odd = nn.ModuleList([
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1, padding=k // 2)
            for k in range(1, self.patch_size_x+2, 2)
        ])

        self.NSC_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5, 12),
                nn.ELU(alpha=1.),
                nn.Linear(12, 12),
                nn.ELU(alpha=1.),
                nn.Linear(12, 6),
                nn.ELU(alpha=1.),
                nn.Linear(6, 1),
                #nn.ELU(alpha=1.),
            )
            for _ in range(self.num_steps)
        ])

        self.nca_fusion = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0)
        self.fermionic_NCA = FermionConvLayer(channels=self.channels,propagation_steps = num_steps, kernel_size=self.kernel_size)
        self.bosonic_NCA = BosonConvLayer(channels=self.channels,propagation_steps = num_steps, kernel_size=self.kernel_size)

        self.wvl = WaveletModel(num_scales=self.wavelet_scales,min_scale_value=self.min_scale_value, max_scale_value=self.max_wavelet_scale,
                         batch_size=self.batch_size, channels=self.channels,num_steps=self.num_steps, height=self.patch_size_x, width=self.patch_size_y,device=self.device).to('cuda')

        self.fermion_features = Linformer(
            dim=self.channels*self.fermion_number*self.wavelet_scales,
            seq_len= self.patch_size_x * self.patch_size_y,
            depth=1,
            heads=self.fermion_number,
            dim_head =self.fermion_number,
            one_kv_head = False,
            share_kv = False,
            reversible = True,
            dropout = 0.00,
            k=self.channels*self.fermion_number*self.wavelet_scales
        )

        self.boson_features = Linformer(
            dim=self.channels*self.boson_number*self.wavelet_scales,
            seq_len= self.patch_size_x * self.patch_size_y,
            depth=1,
            heads=self.boson_number,
            dim_head=self.boson_number,
            one_kv_head=False,
            share_kv=False,
            reversible=True,
            dropout=0.00,
            k=self.channels*self.boson_number*self.wavelet_scales
        )

        self.project_fermions_seq = nn.Conv1d( self.patch_size_x * self.patch_size_y, self.fermion_number  * torch.sum(self.fermion_kernels_size**3)*self.channels,kernel_size=3,padding=1)
        self.project_bosons_seq = nn.Conv1d( self.patch_size_x * self.patch_size_y, self.boson_number  * torch.sum(self.boson_kernels_size**3)*self.channels,kernel_size=3,padding=1)

        self.project_fermions_feature = nn.Conv1d(self.channels*self.fermion_number*self.wavelet_scales, self.channels,kernel_size=3,padding=1)
        self.project_bosons_feature = nn.Conv1d(self.channels*self.boson_number*self.wavelet_scales,self.channels,kernel_size=3,padding=1)

        self.common_nca_pool_layer_norm = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y]).to(self.device)
        self.lnorm_fermion = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y])
        self.lnorm_boson = nn.LayerNorm([self.channels,self.channels, self.patch_size_x, self.patch_size_y])
        self.wvl_layer_norm = nn.LayerNorm(self.channels*self.particle_number*self.wavelet_scales).to(self.device)

        self.learned_fermion_kernels = nn.ParameterList([
            nn.ParameterList([self._init_orthogonal_kernel(nn.Parameter(torch.empty(self.fermion_number, self.channels, k, k, k), requires_grad=False))
                for k in range(1,self.kernel_size+1,2)])
            for _ in range(num_steps)])

        self.learned_boson_kernels = nn.ParameterList([
            nn.ParameterList([self._init_orthogonal_kernel(nn.Parameter(torch.empty(self.boson_number, self.channels, k, k, k), requires_grad=False))
                for k in range(1,self.kernel_size+1,2)])
            for _ in range(num_steps)])
        # self.act = nn.GELU()
        self.step_param = nn.Parameter(torch.ones( self.num_steps), requires_grad=True)
        self.spike_scale = nn.Parameter(torch.ones( self.num_steps), requires_grad=True)

    @staticmethod
    def _init_orthogonal_kernel( param):
        nn.init.orthogonal_(param)
        return param

    def nca_update(self, x,meta_embeddings,spiking_probabilities,hf_data,batch_size):
        self.batch_size = batch_size
        nca_var = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
        log_det_j_loss = torch.zeros((x.shape[0],self.num_steps), requires_grad=True).to(self.device)
        freq_loss = torch.zeros((self.fermion_kernels_size.shape[0],self.num_steps), requires_grad=True).to(self.device)
        past_fermion_kernels = []
        past_boson_kernels = []
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
                # (self.boson_number, self.channels, k, k, k

                cwt_wvl = self.act(self.wvl(reshaped_energy_spectrum,i).permute(0,2,1))
                wavelet_space = self.wvl_layer_norm(cwt_wvl)

                fermion_kernels = self.act(self.fermion_features(wavelet_space))
                ortho_mean,ortho_max = self.validate_channel_orthogonality(fermion_kernels)
                boson_kernels = self.act(self.boson_features(wavelet_space))

                fermion_kernels = self.act(self.project_fermions_seq(fermion_kernels)).permute(0, 2, 1)
                boson_kernels = self.act(self.project_bosons_seq(boson_kernels)).permute(0, 2, 1)
                fermion_kernels = self.act(self.project_fermions_feature(fermion_kernels))
                past_fermion_kernels.append(fermion_kernels)
                boson_kernels = self.act(self.project_bosons_feature(boson_kernels))
                past_boson_kernels.append(boson_kernels)

                fermion_quality = self.evaluate_kernel_quality(fermion_kernels,boson_kernels,past_fermion_kernels,i)
                boson_quality = self.evaluate_kernel_quality(boson_kernels,fermion_kernels,past_boson_kernels,i)

                fermion_kernels = fermion_kernels * fermion_quality
                boson_kernels = boson_kernels * boson_quality
                boson_kernels= torch.mean(boson_kernels,dim=0)
                fermion_kernels = torch.mean(fermion_kernels, dim=0)

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
                energy_spectrum =  (bosonic_energy_states * self.step_param[i] +
                                                     torch.rand_like(bosonic_energy_states) * spiking_probabilities[i] *
                                                     self.spike_scale[i])+energy_spectrum  # Note: Progressing NCA dynamics by dx
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
                energy_spectrum = (bosonic_energy_states * self.step_param[i])+energy_spectrum # Note: Progressing NCA dynamics by dx

                nca_var, ortho_mean, ortho_max,log_det_j_loss ,freq_loss= None,None,None,None,None
                #energy_spectrum = dct.idct_3d(energy_spectrum)
        if self.training:
            self.draw_neural_space_in_2d(self.learned_fermion_kernels,self.learned_boson_kernels,fermionic_response*fermion_quality.unsqueeze(1).unsqueeze(2),bosonic_response*boson_quality.unsqueeze(1).unsqueeze(2))

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
        loss = torch.abs(hf_mean - hf_data) ** 2 + torch.abs(lf_mean - hf_mean)** 2
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

    def evaluate_kernel_quality(self, kernels,comm_kernels,past_kernels,i):
        kernels = kernels.detach()
        comm_kernels = comm_kernels.detach()

        sparsity_score = torch.mean(torch.abs(kernels), dim=[1, 2]) / (torch.norm(kernels, dim=[1, 2], p=1) + 1e-6)
        variance_score = torch.var(kernels, dim=[1, 2])
        #dif = self.directional_information_flow(kernels,comm_kernels)
        E_c = self.energy_conservation(kernels,comm_kernels)
        #f_div = self.frequency_diversity(kernels)
        spatial_coherence = self.spatial_coherence(kernels)
        if len(past_kernels) < 2:
            #mutual_info_score = torch.full_like(variance_score,0.,requires_grad=False)
            temp_coherence = torch.full_like(variance_score,0.,requires_grad=False)

        else:
            #mutual_info_score = self.mutual_information_score(kernels,past_kernels[-2].detach())
            temp_coherence = self.temporal_coherence(kernels, past_kernels[-2].detach())
        quality_score = torch.cat([
                0.5*variance_score.unsqueeze(1),
                0.5*temp_coherence.unsqueeze(1),
                #0.5*spatial_coherence.unsqueeze(1),
                #f_div.unsqueeze(1),
                0.25*E_c.unsqueeze(1),
                #-0.5*mutual_info_score.unsqueeze(1),
                #dif.unsqueeze(1),
                -0.1*sparsity_score.unsqueeze(1)
        ],dim=1)
        #temperature = self.NSC_layers[i](quality_score)
        # print(quality_score.shape)
        quality_score = torch.mean(torch.softmax(quality_score, dim=0),dim=1).unsqueeze(1).unsqueeze(2)
        # print(quality_score.shape)
        return quality_score

    def temporal_coherence(self,f_kernels,s_kernels):
        dot_product = torch.sum(f_kernels*s_kernels,dim=[1,2])
        norm1 = torch.sqrt(torch.sum(f_kernels ** 2,dim=[1,2]))
        norm2 = torch.sqrt(torch.sum(s_kernels ** 2,dim=[1,2]))
        temporal_coherence = dot_product / (norm1 * norm2 + 1e-8)
        return temporal_coherence

    def spatial_coherence(self,kernels):
        mean_per_channel = torch.mean(kernels, dim=[1,2], keepdim=True)
        deviation = kernels - mean_per_channel
        variance = torch.mean(deviation ** 2, dim=[1,2])
        spatial_coherence = 1 / (variance + 1e-8)
        return spatial_coherence

    def mutual_information_score(self, input_tensor, output_tensor, num_bins=128):
        batch_size = input_tensor.size(0)
        input_flat = input_tensor.view(batch_size, -1)
        output_flat = output_tensor.view(batch_size, -1)

        min_val = torch.min(torch.cat([input_flat, output_flat], dim=1), dim=1).values
        max_val = torch.max(torch.cat([input_flat, output_flat], dim=1), dim=1).values
        bin_edges = torch.stack([
            torch.linspace(min_val[i], max_val[i], steps=num_bins, device=input_tensor.device)
            for i in range(batch_size)
        ])

        bin_widths = bin_edges[:, 1] - bin_edges[:, 0]

        input_probs = torch.stack([
            torch.exp(-0.5 * ((input_flat[i][:, None] - bin_edges[i]) / bin_widths[i]) ** 2)
            for i in range(batch_size)
        ])

        output_probs = torch.stack([
            torch.exp(-0.5 * ((output_flat[i][:, None] - bin_edges[i]) / bin_widths[i]) ** 2)
            for i in range(batch_size)
        ])

        input_probs = input_probs / input_probs.sum(dim=2, keepdim=True)
        output_probs = output_probs / output_probs.sum(dim=2, keepdim=True)

        joint_probs = torch.stack([
            torch.einsum('ij,ik->jk', input_probs[i], output_probs[i])
            for i in range(batch_size)
        ])
        joint_probs /= joint_probs.sum(dim=(1, 2), keepdim=True)

        input_marginal = joint_probs.sum(dim=2)
        output_marginal = joint_probs.sum(dim=1)

        joint_probs = joint_probs + 1e-10
        input_marginal = input_marginal + 1e-10
        output_marginal = output_marginal + 1e-10

        mutual_info = torch.stack([
            torch.sum(
                joint_probs[i] * torch.log(joint_probs[i] / (input_marginal[i][:, None] * output_marginal[i][None, :]))
            )
            for i in range(batch_size)
        ])

        return mutual_info

    def directional_information_flow(self, f_kernels, s_kernels):
        f_energy = torch.norm(f_kernels, p=2, dim=[1,2])
        s_energy = torch.norm(s_kernels, p=2, dim=[1,2])
        similarity = f_energy * s_energy / (
                torch.norm(f_energy) * torch.norm(s_energy) + 1e-6
        )
        return -similarity

    def energy_conservation(self, initial_energy, transformed_energy):
        energy_difference = torch.abs(torch.sum(transformed_energy, dim=[1,2]) - torch.sum(initial_energy, dim=[1,2]))
        return -torch.log(1 + energy_difference)

    def frequency_diversity(self, kernel):
        fft_k = torch.fft.fftn(kernel, dim=[1,2])
        spectrum = torch.abs(fft_k)
        low_freq_energy = torch.mean(spectrum[..., :kernel.shape[-1] // 3], dim=[1,2])
        mid_freq_energy = torch.mean(spectrum[..., kernel.shape[-1] // 3: 2 * kernel.shape[-1] // 3],dim=[1,2])
        high_freq_energy = torch.mean(spectrum[..., 2 * kernel.shape[-1] // 3:],dim=[1,2])

        diversity_score = 1.0 / (torch.abs(low_freq_energy - mid_freq_energy) +
                                 torch.abs(mid_freq_energy - high_freq_energy) + 1e-6)
        return diversity_score

    def temporal_stability(self, states):
        temporal_differences = [torch.norm(states[i] - states[i - 1], p=2) for i in range(1, len(states))]
        stability_score = torch.mean(torch.stack(temporal_differences))
        return -stability_score


    def draw_neural_space_in_2d(self,f_kernels,b_kernels,e1,e2):
        for ax_row in self.axs2d:
            for ax in ax_row:
                ax.clear()
        cmap_diff = 'binary'
        cmap_org = 'plasma'
        e1 = e1.mean(dim=0).reshape(self.num_steps,25,45).cpu().detach().numpy()
        e2 = e2.mean(dim=0).reshape(self.num_steps,25,45).cpu().detach().numpy()
        k_1 = []
        k_3 = []
        for i in range(0, len(f_kernels)):
            for j in range(0, len(f_kernels[i])):
                if f_kernels[i][j].shape[-1] == 1:
                    k_1.append(f_kernels[i][j])
                else:
                    k_3.append(f_kernels[i][j])
        k_1 = torch.stack(k_1)
        k_3 = torch.stack(k_3)
        k_1_nparray = k_1.reshape(self.num_steps, 5, 5).cpu().detach().numpy()
        k_3_nparray = k_3.reshape(self.num_steps, 25, 27).cpu().detach().numpy()
        norm_data_k1 = (k_1_nparray - np.min(k_1_nparray)) / (np.max(k_1_nparray) - np.min(k_1_nparray))
        norm_data_k3 = (k_3_nparray - np.min(k_3_nparray)) / (np.max(k_3_nparray) - np.min(k_3_nparray))
        k = 0
        for step in range(0,self.num_steps*2,2):
            ax = self.axs2d[0, step]
            ax.imshow(e1[k], cmap=cmap_org, origin="lower")
            #ax.set_title(f"Step {step}")
            k+=1
        k=0
        for step in range(1,self.num_steps*2,2):
            ax = self.axs2d[0,step]
            if step == 0:
                ax.imshow(e1[k], cmap=cmap_org, origin="lower")
                #ax.set_title(f"Step {step}")
            else:
                diff = np.abs(e1[k] - e1[k - 1])
                diff = 1 - (diff / np.max(diff))
                ax.imshow(diff, cmap=cmap_diff, origin="lower")
            k += 1
        k = 0

        for step in range(0,self.num_steps*2,2):
            ax = self.axs2d[1, step]
            ax.imshow(norm_data_k1[k], cmap=cmap_org, origin="lower")
            #ax.set_title(f"Step {step}")
            k+=1
        k=0
        for step in range(1,self.num_steps*2,2):
            ax = self.axs2d[1,step]
            if step == 0:
                ax.imshow(norm_data_k1[k], cmap=cmap_org, origin="lower")
                #ax.set_title(f"Step {step}")
            else:
                diff = np.abs(norm_data_k1[k] - norm_data_k1[k - 1])
                diff = 1 - (diff / np.max(diff))
                ax.imshow(diff, cmap=cmap_diff, origin="lower")
            k += 1
        k = 0
        for step in range(0,self.num_steps*2,2):
            ax = self.axs2d[2, step]
            ax.imshow(norm_data_k3[k], cmap=cmap_org, origin="lower")
            #ax.set_title(f"Original Step {step}")
            k += 1
        k = 0
        for step in range(1,self.num_steps*2,2):
            ax = self.axs2d[2,step]
            if step == 0:
                ax.imshow(norm_data_k3[k], cmap=cmap_org, origin="lower")
                #ax.set_title(f"Step {step}")
            else:
                diff  = np.abs(norm_data_k3[k] - norm_data_k3[k - 1])
                diff = 1 - (diff / np.max(diff))
                ax.imshow(diff, cmap=cmap_diff, origin="lower")
            k += 1

        for ax_row in self.axs2d:
            for a in ax_row:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_aspect('equal')

        k_1 = []
        k_3 = []
        for i in range(0, len(b_kernels)):
            for j in range(0, len(b_kernels[i])):
                if b_kernels[i][j].shape[-1] == 1:
                    k_1.append(b_kernels[i][j])
                else:
                    k_3.append(b_kernels[i][j])
        k_1 = torch.stack(k_1)
        k_3 = torch.stack(k_3)
        k_1_nparray = k_1.reshape(self.num_steps, 5, 5).cpu().detach().numpy()
        k_3_nparray = k_3.reshape(self.num_steps, 25, 27).cpu().detach().numpy()
        norm_data_k1 = (k_1_nparray - np.min(k_1_nparray)) / (np.max(k_1_nparray) - np.min(k_1_nparray))
        norm_data_k3 = (k_3_nparray - np.min(k_3_nparray)) / (np.max(k_3_nparray) - np.min(k_3_nparray))

        k=0
        for step in range(0,self.num_steps*2,2):
            ax = self.axs2d[3, step]
            ax.imshow(e2[k], cmap=cmap_org, origin="lower")
            #ax.set_title(f"Step {step}")
            k+=1
        k=0
        for step in range(1,self.num_steps*2,2):
            ax = self.axs2d[3,step]
            if step == 0:
                ax.imshow(e2[k], cmap=cmap_org, origin="lower")
                #ax.set_title(f"Step {step}")
            else:
                diff = np.abs(e2[k] - e2[k - 1])
                diff = 1 - (diff / np.max(diff))
                ax.imshow(diff, cmap=cmap_diff, origin="lower")
            k += 1
        k = 0
        for step in range(0,self.num_steps*2,2):
            ax = self.axs2d[4, step]
            ax.imshow(norm_data_k1[k], cmap=cmap_org, origin="lower")
            #ax.set_title(f"Step {step}")
            k += 1
        k = 0
        for step in range(1,self.num_steps*2,2):
            ax = self.axs2d[4, step]
            if step == 0:
                ax.imshow(norm_data_k1[k], cmap=cmap_org, origin="lower")
                # ax.set_title(f"Step {step}")
            else:
                diff = np.abs(norm_data_k1[k] - norm_data_k1[k - 1])
                diff = 1 - (diff / np.max(diff))
                ax.imshow(diff, cmap=cmap_diff, origin="lower")
            k += 1
        k = 0
        for step in range(0,self.num_steps*2,2):
            ax = self.axs2d[5, step]
            ax.imshow(norm_data_k3[k], cmap=cmap_org, origin="lower")
            # ax.set_title(f"Original Step {step}")
            k += 1
        k = 0
        for step in range(1,self.num_steps*2,2):
            ax = self.axs2d[5, step]
            if step == 0:
                ax.imshow(norm_data_k3[k], cmap=cmap_org, origin="lower")
                # ax.set_title(f"Step {step}")
            else:
                diff = np.abs(norm_data_k3[k] - norm_data_k3[k - 1])
                diff = 1 - (diff / np.max(diff))
                ax.imshow(diff, cmap=cmap_diff, origin="lower")
            k += 1

        for ax_row in self.axs2d:
            for a in ax_row:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_aspect('equal')

        self.fig2d.canvas.draw_idle()
        self.fig2d.canvas.start_event_loop(0.01)
        #self.fig2d.canvas.draw()
        #plt.pause(0.01)
       # time.sleep(1000)

    def draw_neural_space_in_3d(self,kernels):
        self.ax3d.clear()
        k_1 = []
        k_3 = []
        for i in range(0, len(kernels)):
            for j in range(0, len(kernels[i])):
                if kernels[i][j].shape[-1] == 1:
                    k_1.append(kernels[i][j])
                else:
                    k_3.append(kernels[i][j])
        #k_1 = torch.stack(k_1)
        k_3 = torch.stack(k_3)
       # print(k_3.shape)
        #print(k_1.reshape(self.num_steps, 5, 5))
        #k_1_nparray = k_1.reshape(self.num_steps, 5, 5).cpu().detach().numpy()
        k_3_nparray = k_3.reshape(self.num_steps, 25, 27).cpu().detach().numpy()
        x, y, z = np.meshgrid(
            np.arange(k_3_nparray.shape[0]),
            np.arange(k_3_nparray.shape[1]),
            np.arange(k_3_nparray.shape[2]),
            indexing="ij"
        )
        dx = dy = dz = 1.
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        norm_data = (k_3_nparray - np.min(k_3_nparray)) / (np.max(k_3_nparray) - np.min(k_3_nparray))
        colors = plt.cm.viridis(norm_data.flatten())
        colors[:, -1] = 0.01

        if hasattr(self, 'bars'):
            for bar in self.bars:
                bar.remove()
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.remove()

        # if self.cbar is not None:
        #     self.cbar.remove()
        # self.ax.bar3d(
        #     x_flat - dx / 2,
        #     y_flat - dy / 2,
        #     z_flat - dz / 2,
        #     dx, dy, dz,
        #     color=colors,
        #     shade=True
        # )
        scatter = self.ax3d.scatter(
            x_flat,
            y_flat,
            z_flat,
            c=norm_data.flatten(),
            cmap="plasma",
            s=25,
            alpha=0.7
        )
        self.fig3d.canvas.draw()
        plt.pause(0.01)




class FermionConvLayer(nn.Module):
    def __init__(self, channels,propagation_steps, kernel_size=3):
        super(FermionConvLayer, self).__init__()
        self.fermion_gate_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.fermion_gate_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        # Note: Normalising flow
        self.scale_net = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.shift_net = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        #self.threshold = nn.Parameter(torch.rand(propagation_steps))
        self.act = nn.ELU(alpha=1.)
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
        self.act = nn.ELU(alpha=1.)
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

