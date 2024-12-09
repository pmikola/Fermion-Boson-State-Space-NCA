import math
import time
import torch.nn.functional as f

import torch
import torch.nn as nn
from linformer import Linformer
from torch.nn.utils import spectral_norm as sn

from NCA import NCA


class Fermionic_Bosonic_Space_State_NCA(nn.Module):
    def __init__(self, batch_size,no_frame_samples, input_window_size,hdc_dim,rbf_dim,nca_steps, device):
        super(Fermionic_Bosonic_Space_State_NCA, self).__init__()
        self.last_frame = None
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.hdc_dim = hdc_dim
        self.rbf_dim = rbf_dim
        self.in_scale = (1 + self.input_window_size * 2)
        self.loss_weights = nn.Parameter(torch.ones(16))
        self.modes = 32
        self.uplift_meta_0 = nn.Linear(200,15*self.modes)
        self.uplift_meta_1 = nn.Linear(15*self.modes, 5 * self.in_scale ** 2)
        self.uplift_meta = nn.Conv2d(in_channels=5, out_channels=self.hdc_dim,kernel_size=1)
        self.uplift_data = nn.Conv2d(in_channels=5, out_channels=self.hdc_dim, kernel_size=1)
        self.cross_correlate_in = nn.Conv3d(in_channels=1, out_channels=self.hdc_dim, kernel_size=3, stride=1, padding=1)
        self.cross_correlate_out = nn.Conv3d(in_channels=self.hdc_dim, out_channels=self.hdc_dim, kernel_size=3, stride=1, padding=1)
        self.nca_steps = nca_steps
        self.act = nn.ELU(alpha=1.0)
        self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H, self.I, self.J, self.K, self.L, self.M, self.N = torch.nn.Parameter(torch.full((14,),1.),requires_grad=True).to(self.device)
        self.channels = 5
        self.xchannels = 4
        self.xxchannels = 3
        # self.act = nn.GELU()
        self.NCA = NCA(self.batch_size,self.hdc_dim, self.nca_steps, self.device)
        self.downlift_data = nn.Conv3d(in_channels=self.hdc_dim,out_channels=self.hdc_dim,kernel_size=1)
        self.rgbas = nn.Conv3d(in_channels=self.hdc_dim,out_channels=self.channels,kernel_size=3, stride=1, padding=1)
        # [1, 3, 5, 7, 9, 11, 13, 15]
        self.r = nn.ModuleList(
            [nn.Conv2d(in_channels=self.channels, out_channels=self.xchannels, kernel_size=k, padding=k // 2) for k in [1, 3, 5, 7, 9, 11, 13, 15]])
        self.g = nn.ModuleList(
            [nn.Conv2d(in_channels=self.channels, out_channels=self.xchannels, kernel_size=k, padding=k // 2) for k in [1, 3, 5, 7, 9, 11, 13, 15]])
        self.b = nn.ModuleList(
            [nn.Conv2d(in_channels=self.channels, out_channels=self.xchannels, kernel_size=k, padding=k // 2) for k in [1, 3, 5, 7, 9, 11, 13, 15]])
        self.a = nn.ModuleList(
            [nn.Conv2d(in_channels=self.channels, out_channels=self.xchannels, kernel_size=k, padding=k // 2) for k in [1, 3, 5, 7, 9, 11, 13, 15]])
        self.s = nn.ModuleList(
            [nn.Conv2d(in_channels=self.channels, out_channels=self.xchannels, kernel_size=k, padding=k // 2) for k in [1, 3, 5, 7, 9, 11, 13, 15]])
        self.r_norm = nn.LayerNorm([self.xchannels, self.in_scale, self.in_scale])
        self.g_norm = nn.LayerNorm([self.xchannels, self.in_scale, self.in_scale])
        self.b_norm = nn.LayerNorm([self.xchannels, self.in_scale, self.in_scale])
        self.a_norm = nn.LayerNorm([self.xchannels, self.in_scale, self.in_scale])
        self.s_norm = nn.LayerNorm([self.xchannels, self.in_scale, self.in_scale])

        self.r_h = nn.Conv2d(in_channels=self.xchannels, out_channels=self.xxchannels, kernel_size=1)
        self.g_h = nn.Conv2d(in_channels=self.xchannels, out_channels=self.xxchannels, kernel_size=1)
        self.b_h = nn.Conv2d(in_channels=self.xchannels, out_channels=self.xxchannels, kernel_size=1)
        self.a_h = nn.Conv2d(in_channels=self.xchannels, out_channels=self.xxchannels, kernel_size=1)
        self.s_h = nn.Conv2d(in_channels=self.xchannels, out_channels=self.xxchannels, kernel_size=1)

        self.r_o = nn.Conv2d(in_channels=self.xxchannels, out_channels=1, kernel_size=1)
        self.g_o = nn.Conv2d(in_channels=self.xxchannels, out_channels=1, kernel_size=1)
        self.b_o = nn.Conv2d(in_channels=self.xxchannels, out_channels=1, kernel_size=1)
        self.a_o = nn.Conv2d(in_channels=self.xxchannels, out_channels=1, kernel_size=1)
        self.s_o = nn.Conv2d(in_channels=self.xxchannels, out_channels=1, kernel_size=1)
        #self.feedback_weights = nn.Parameter(torch.rand(10))
        self.init_weights()

    def init_weights(self, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, Linformer):
                for submodule in m.modules():
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        if submodule.bias is not None:
                            nn.init.constant_(submodule.bias, 0.0)
                    elif isinstance(submodule, nn.Conv2d):
                        nn.init.orthogonal_(submodule.weight)
                        if submodule.bias is not None:
                            nn.init.constant_(submodule.bias, 0.0)

        print("Weight initialization complete")

    def weight_reset(self: nn.Module):
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()
        # NOTE : Making sure that nn.conv2d and nn.linear will be reset
        if isinstance(self, nn.Conv3d) or isinstance(self, nn.Linear):
            self.reset_parameters()

    def forward(self, din,spiking_probabilities=None):
        #torch.cuda.synchronize()
        #t_start = time.perf_counter()
        #spiking_probabilities = F.normalize(spiking_probabilities, p=1, dim=0)

        old_batch_size = self.batch_size
        (data_input, structure_input, meta_input_h1, meta_input_h2, meta_input_h3,
         meta_input_h4, meta_input_h5, noise_var_in_binary, fmot_in_binary, meta_output_h1, meta_output_h2,
         meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out) = din
        if data_input.shape[0] != self.batch_size:
            self.batch_size = data_input.shape[0]
        #################################################################
        # s = torch.flatten(structure_input,start_dim=1)
        # data = torch.flatten(data_input,start_dim=1)
        r_in = data_input[:, 0:self.in_scale, :].unsqueeze(1)
        g_in = data_input[:, self.in_scale:self.in_scale * 2, :].unsqueeze(1)
        b_in = data_input[:, self.in_scale * 2:self.in_scale * 3, :].unsqueeze(1)
        a_in = data_input[:, self.in_scale * 3:self.in_scale * 4, :].unsqueeze(1)
        s_in = structure_input.unsqueeze(1)
        data = torch.cat([r_in,g_in,b_in,a_in,s_in],dim=1)
        if self.training:
            hf_data = self.high_frequency_extraction(data)
        else:
            hf_data = None
        time_in,time_out = meta_input_h2,meta_output_h2
        pos_in,pos_out = meta_input_h3,meta_output_h3

        meta_to_uplift = torch.cat([time_in,time_out,fmot_in_binary,noise_var_in_binary,noise_var_out,pos_in,pos_out],dim=-1)
        #print(meta_to_uplift.shape)
        meta_uplifted = self.act(self.uplift_meta_0(meta_to_uplift))
        meta_uplifted = self.act(self.uplift_meta_1(meta_uplifted))
        meta_uplifted = meta_uplifted.view(self.batch_size,5, self.in_scale,  self.in_scale)
        meta_embeddings =  self.act(self.uplift_meta(meta_uplifted))

        x = self.act(self.uplift_data(data))
        x = x.unsqueeze(1)
        x_i = self.act(self.cross_correlate_in(x))
        x,nca_var,ortho_mean,ortho_max,log_det_jacobian_loss,freq_loss = self.NCA(x_i,meta_embeddings,spiking_probabilities,hf_data,self.batch_size)
        x = self.act(self.cross_correlate_out(x))+x_i
        x = self.act(self.downlift_data(x))
        rgbas = self.act(self.rgbas(x))

        r = torch.sum(torch.stack([self.act(layer(rgbas[:, :,  0, :, :].squeeze(1))) for layer in self.r]), dim=0)
        g = torch.sum(torch.stack([self.act(layer(rgbas[:, :,  1, :, :].squeeze(1))) for layer in self.g]), dim=0)
        b = torch.sum(torch.stack([self.act(layer(rgbas[:, :,  2, :, :].squeeze(1))) for layer in self.b]), dim=0)
        a = torch.sum(torch.stack([self.act(layer(rgbas[:, :,  3, :, :].squeeze(1))) for layer in self.a]), dim=0)
        s = torch.sum(torch.stack([self.act(layer(rgbas[:, :,  4, :, :].squeeze(1))) for layer in self.s]), dim=0)

        r = self.r_norm(r)
        g = self.g_norm(g)
        b = self.b_norm(b)
        a = self.a_norm(a)
        s = self.s_norm(s)

        r = self.act(self.r_h(r))
        g = self.act(self.g_h(g))
        b = self.act(self.b_h(b))
        a = self.act(self.a_h(a))
        s = self.act(self.s_h(s))

        r = self.r_o(r).squeeze(1)
        g = self.g_o(g).squeeze(1)
        b = self.b_o(b).squeeze(1)
        a = self.a_o(a).squeeze(1)
        s = self.s_o(s).squeeze(1)

        deepS = r, g, b, a, s
        #torch.cuda.current_stream().synchronize()
        #t_stop = time.perf_counter()
        # print("model internal time patch : ", ((t_stop - t_start) * 1e3) / self.batch_size, "[ms]")
        # time.sleep(10000)
        if self.training:
            # hf_loss =self.fft_high_frequency_loss(self.r_h.weight,hf_data)
            # hf_loss +=self.fft_high_frequency_loss(self.g_h.weight,hf_data)
            # hf_loss +=self.fft_high_frequency_loss(self.b_h.weight,hf_data)
            # hf_loss +=self.fft_high_frequency_loss(self.a_h.weight,hf_data)
            # hf_loss +=self.fft_high_frequency_loss(self.s_h.weight,hf_data)
            hf_loss = sum(self.fft_high_frequency_loss(layer.weight, hf_data) / layer.weight.numel() for layer in
                          [self.r_h, self.g_h, self.b_h, self.a_h, self.s_h])

            # hf_loss += torch.sum(torch.stack([self.fft_high_frequency_loss(layer.weight,hf_data) for layer in self.r]), dim=0)
            # hf_loss += torch.sum(torch.stack([self.fft_high_frequency_loss(layer.weight,hf_data) for layer in self.g]), dim=0)
            # hf_loss += torch.sum(torch.stack([self.fft_high_frequency_loss(layer.weight,hf_data) for layer in self.b]), dim=0)
            # hf_loss += torch.sum(torch.stack([self.fft_high_frequency_loss(layer.weight,hf_data) for layer in self.a]), dim=0)
            # hf_loss += torch.sum(torch.stack([self.fft_high_frequency_loss(layer.weight,hf_data) for layer in self.s]), dim=0)

            hf_loss += self.fft_high_frequency_loss(self.rgbas.weight,hf_data) / self.rgbas.weight.numel()
            hf_loss += self.fft_high_frequency_loss(self.downlift_data.weight,hf_data) / self.downlift_data.weight.numel()
            hf_loss += self.fft_high_frequency_loss(self.cross_correlate_out.weight,hf_data) / self.cross_correlate_out.weight.numel()
            hf_loss += self.fft_high_frequency_loss(self.cross_correlate_in.weight,hf_data) / self.cross_correlate_in.weight.numel()
            hf_loss += self.fft_high_frequency_loss( self.uplift_data.weight,hf_data) / self.uplift_data.weight.numel()

            freq_loss += hf_loss
        self.batch_size = old_batch_size
        return r, g, b, a, s, deepS,nca_var,ortho_mean,ortho_max,log_det_jacobian_loss,freq_loss,self.loss_weights

    def high_frequency_extraction(self,data, cutoff_ratio=0.7):
        C, H, W, D = data.shape
        mask_lf = torch.zeros((C, H, W, D), device=self.device)
        fft_d = torch.fft.fftn(data)
        fft_d_shifted = torch.fft.fftshift(fft_d)
        center_ch, center_x, center_y, center_z = (C - 1) // 2, (H - 1) // 2, (W - 1) // 2, (D - 1) // 2
        cutoff_ch = int(cutoff_ratio * center_ch)
        cutoff_x = int(cutoff_ratio * center_x)
        cutoff_y = int(cutoff_ratio * center_y)
        cutoff_z = int(cutoff_ratio * center_z)
        mask_lf[center_ch - cutoff_ch:center_ch + cutoff_ch,
        center_x - cutoff_x:center_x + cutoff_x,
        center_y - cutoff_y:center_y + cutoff_y,
        center_z - cutoff_z:center_z + cutoff_z] = 1.
        high_freq_k = fft_d_shifted * (1 - mask_lf)
        hf_mean = torch.abs(high_freq_k).mean()
        return hf_mean

    def fft_high_frequency_loss(self,kernels,hf_data, cutoff_ratio=0.7):
        if len(kernels.shape) == 4:
            C,H, W, D = kernels.shape
            mask_lf = torch.zeros((C, H, W, D), device=self.device)
            fft_k = torch.fft.fftn(kernels)
            fft_k_shifted = torch.fft.fftshift(fft_k)
            center_ch,center_x, center_y, center_z = (C-1)//2, (H-1) // 2, (W -1)// 2,( D -1)// 2
            cutoff_ch = int(cutoff_ratio * center_ch)
            cutoff_x = int(cutoff_ratio * center_x)
            cutoff_y = int(cutoff_ratio * center_y)
            cutoff_z = int(cutoff_ratio * center_z)
            mask_lf[center_ch - cutoff_ch:center_ch + cutoff_ch,
            center_x - cutoff_x:center_x + cutoff_x,
            center_y - cutoff_y:center_y + cutoff_y,
            center_z - cutoff_z:center_z + cutoff_z] = 1.
        else:
            C,HDC, H, W, D = kernels.shape
            mask_lf = torch.zeros((C, HDC, H, W, D), device=self.device)
            fft_k = torch.fft.fftn(kernels)
            fft_k_shifted = torch.fft.fftshift(fft_k)
            center_ch,center_hdc,center_x, center_y, center_z = (C-1)//2, (HDC-1)// 2, (H-1) // 2, (W -1)// 2,( D -1)// 2
            cutoff_ch = int(cutoff_ratio * center_ch)
            cutoff_hdc = int(cutoff_ratio * center_hdc)
            cutoff_x = int(cutoff_ratio * center_x)
            cutoff_y = int(cutoff_ratio * center_y)
            cutoff_z = int(cutoff_ratio * center_z)
            mask_lf[center_ch - cutoff_ch:center_ch + cutoff_ch,
            center_hdc - cutoff_hdc:center_hdc + cutoff_hdc,
            center_x - cutoff_x:center_x + cutoff_x,
            center_y - cutoff_y:center_y + cutoff_y,
            center_z - cutoff_z:center_z + cutoff_z] = 1.
        high_freq_k = fft_k_shifted * (1 - mask_lf)
        low_freq_k = fft_k_shifted * mask_lf                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        hf_mean = torch.abs(high_freq_k).mean()
        lf_mean = torch.abs(low_freq_k).mean()
        # loss = (hf_mean / (1+lf_mean+hf_data))
        loss = torch.abs(hf_mean - hf_data) ** 2 + torch.abs(lf_mean - hf_mean)**2
        return loss

    @staticmethod
    def binary(x, bits):
        mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


    def meta_encoding(self,meta_in,meta_out, dim, max_time_step):
        pos_in = meta_in / max_time_step
        pos_out = meta_out / max_time_step
        pe = torch.zeros(self.batch_size,dim,requires_grad=True).to(self.device)
        for i in range(dim):
            angle_in = pos_in / (10000 ** ((2 * (i // 2)) / dim))
            angle_out = pos_out / (10000 ** ((2 * (i // 2)) / dim))
            angle_in = angle_in.unsqueeze(0)
            angle_out = angle_out.unsqueeze(0)
            if i % 2 == 0:
                pe[:,i] = (torch.sin(angle_in)+torch.cos(angle_out))/2
            else:
                pe[:,i] = (torch.cos(angle_in)+torch.sin(angle_out))/2
        return pe