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
        self.scales = nn.Parameter(torch.linspace(min_scale_value, max_scale_value, num_scales)).to(self.device)
        self.f_mod = nn.Parameter(torch.ones(num_steps),requires_grad=True).to(self.device)
        self.poly_mod = nn.Parameter(torch.ones(num_steps,14),requires_grad=True).to(self.device)
        self.exp_mod = nn.Parameter(torch.ones(num_steps, 1),requires_grad=True).to(self.device)
        self.frac_order = nn.Parameter(torch.ones(num_steps,12),requires_grad=True).to(self.device)
        self.phase_mod = nn.Parameter(torch.ones(num_steps, 3),requires_grad=True).to(self.device)
    def forward(self, x,i):
        cwt_transformed = self.cwt(x,self.scales,i)
        return cwt_transformed

    def wavelet(self,t, f,i):
        # Note : Real Valued Morlet Wavelet
        sigma = self.f_mod[i] / (2 * torch.pi * f)
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
        gauss_window =  self.exp_mod[i,0]*exp_dec_2

        real_wavelet = real_part * gauss_window
        imag_wavelet = imag_part * gauss_window

        amp = real_wavelet + imag_wavelet
        scaled_amp = amp / torch.norm(amp)#3.0
        phase = torch.atan2(imag_wavelet, real_wavelet)
        sin_component = scaled_amp * torch.sin(phase)*self.phase_mod[i,0]
        cos_component = scaled_amp * torch.cos(phase)*self.phase_mod[i,1]
        sinc_component = scaled_amp * torch.sinc(phase / torch.pi) * self.phase_mod[i, 2]
        wavelet = torch.abs(sin_component * cos_component * sinc_component)
        return wavelet

    def cwt(self,x, scales,i, device="cuda"):
        fdim = x.shape[1]
        x= x.flatten(start_dim=2)#.type(torch.complex64)
        space_steps = torch.arange(x.shape[-1], device=device)
        wavelets = torch.stack([self.wavelet(space_steps, 1.0 / scale,i).unsqueeze(0) for scale in self.scales]).to(device).repeat(fdim,1,1)
        x = x.repeat(1,len(scales),1)
        cwt_result = F.conv1d(x, wavelets, padding="same", groups=fdim * len(scales))
        return cwt_result