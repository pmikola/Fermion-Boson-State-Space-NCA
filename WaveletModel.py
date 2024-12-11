import time

import torch
from torch import nn
import torch.nn.functional as F

class WaveletModel(nn.Module):
    def __init__(self, num_scales, min_scale_value,max_scale_value, batch_size, channels,num_steps, height, width,device):
        super(WaveletModel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.num_steps = num_steps
        self.scales = nn.Parameter(torch.linspace(min_scale_value, max_scale_value, self.num_scales)).to(self.device)
        self.f_mod = nn.Parameter(torch.ones(self.num_steps),requires_grad=True).to(self.device)
        self.poly_mod = nn.Parameter(torch.ones(self.num_steps,14),requires_grad=True).to(self.device)
        self.exp_mod = nn.Parameter(torch.ones(self.num_steps, 1),requires_grad=True).to(self.device)
        self.frac_order = nn.Parameter(torch.ones(self.num_steps,12),requires_grad=True).to(self.device)
        self.phase_mod = nn.Parameter(torch.ones(self.num_steps, 3),requires_grad=True).to(self.device)
        self.norm_const = nn.Parameter(torch.ones(self.num_scales,self.num_steps),requires_grad=True).to(self.device)

    def forward(self, x,i):
        cwt_transformed = self.cwt(x,self.scales,i)
        return cwt_transformed

    def wavelet(self,t, f,i):
        # Note : Real Valued Morlet Wavelet
        sigma = (self.f_mod[i] / (2 * torch.pi * f)).clamp(min=1e-6)
        base_real = torch.cos(2 * torch.pi * f * t)
        base_imag = torch.sin(2 * torch.pi * f * t)
        real_part = (
                base_real*self.poly_mod[i,0]
                - (t ** self.frac_order[i,0])*2 / (sigma ** self.frac_order[i,0])*2 * base_real*self.poly_mod[i,1]
                + (t ** self.frac_order[i,1])*3 / (sigma ** self.frac_order[i,1])*3 * base_real*self.poly_mod[i,2]
                - (t ** self.frac_order[i,2])*4 / (sigma ** self.frac_order[i,2])*4 * base_real*self.poly_mod[i,3]
                + (t ** self.frac_order[i,3])*5 / (sigma ** self.frac_order[i,3])*5 * base_real*self.poly_mod[i,4]
                - (t ** self.frac_order[i,4])*6 / (sigma ** self.frac_order[i,4])*6 * base_real*self.poly_mod[i,5]
                + (t ** self.frac_order[i,5])*7 / (sigma ** self.frac_order[i,5])*7 * base_real*self.poly_mod[i,6]
        )

        imag_part = (
                base_imag*self.poly_mod[i,7]
                - (t ** self.frac_order[i,6])*2 / (sigma ** self.frac_order[i,6])*2 * base_imag*self.poly_mod[i,8]
                + (t ** self.frac_order[i,7])*3 / (sigma ** self.frac_order[i,7])*3 * base_imag*self.poly_mod[i,9]
                - (t ** self.frac_order[i,8])*4 / (sigma ** self.frac_order[i,8])*4 * base_imag*self.poly_mod[i,10]
                + (t ** self.frac_order[i,9])*5 / (sigma ** self.frac_order[i,9]*5) * base_imag*self.poly_mod[i,11]
                - (t ** self.frac_order[i,10])*6 / (sigma ** self.frac_order[i,10])*6 * base_imag*self.poly_mod[i,12]
                + (t ** self.frac_order[i,11])*7 / (sigma ** self.frac_order[i,11])*7 * base_imag*self.poly_mod[i,13]
        )

        exp_dec_2 = torch.exp(-t ** 2 / (2 * sigma ** 2))
        exp_dec_3 = torch.exp(-t ** 3 / (2 * sigma ** 3))
        gauss_window =  self.exp_mod[i,0]*exp_dec_2#+exp_dec_3)

        real_wavelet = real_part * gauss_window
        imag_wavelet = imag_part * gauss_window

        amp = real_wavelet + imag_wavelet
        scaled_amp = amp / torch.norm(amp)
        phase = torch.atan2(imag_wavelet, real_wavelet)
        #modulated_phase = phase * torch.exp(-self.phase_mod[i, 0]) + torch.sin(self.phase_mod[i, 1] * phase)
        sin_component = scaled_amp * torch.sin(phase)* self.phase_mod[i, 0]
        cos_component = scaled_amp * torch.cos(phase)* self.phase_mod[i, 1]
        sinc_component = scaled_amp * torch.sinc(phase / torch.pi) * self.phase_mod[i, 2]
        wavelet = sin_component * cos_component * sinc_component
        return wavelet

    def cwt(self,x, scales,i, device="cuda"):
        fdim = x.shape[1]
        x= x.flatten(start_dim=2)
        space_steps = torch.arange(x.shape[-1], device=device)
        wavelets = torch.stack([self.wavelet(space_steps, 1.0 / scale,i).unsqueeze(0) for scale in self.scales]).to(device).repeat(fdim,1,1)
        x = x.repeat(1,len(scales),1)
        cwt_result = F.conv1d(x, wavelets, padding="same", groups=fdim * len(scales))
        return cwt_result

    def icwt(self, cwt_result, i, device="cuda"):
        cwt_result = cwt_result.permute(0, 2, 1)
        B = cwt_result.size(0)
        num_scales = self.num_scales
        channels_per_scale = cwt_result.shape[1] // num_scales
        H, W = self.height, self.width
        cwt_result = cwt_result.view(B, num_scales, channels_per_scale, H * W)
        reconstructed = torch.zeros((B, channels_per_scale, H * W), device=device)
        time_steps = torch.arange(H * W, device=device)
        for s in range(num_scales):
            scale = self.scales[s]
            scale_coeffs = cwt_result[:, s, :, :]
            wavelet = self.wavelet(time_steps, 1.0 / scale, i).to(device)
            wavelet = wavelet.unsqueeze(0).unsqueeze(0)
            wavelet = wavelet.repeat(channels_per_scale, 1, 1)
            wavelet_rev = torch.flip(wavelet, dims=[-1])
            contribution = torch.nn.functional.conv1d(scale_coeffs, wavelet_rev, padding='same',
                                                      groups=channels_per_scale)
            reconstructed += self.norm_const[s,i]*contribution / num_scales # Note: not excat normalisation - should be integrated over scales contribiution in weighted manner
        reconstructed = reconstructed.view(B, self.channels, self.channels, H, W)
        return reconstructed