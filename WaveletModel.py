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
        self.f_mod = nn.Parameter(torch.randn(num_steps)).to(self.device)
        self.poly_mod = nn.Parameter(torch.randn(num_steps,3)).to(self.device)


    def forward(self, x,i):
        cwt_transformed = self.cwt(x,self.scales,i)
        return cwt_transformed


    def wavelet(self,t, f,i):
        # Note: second derivative Gaussian wavelet:real
        sigma = self.f_mod[i] / (2 * torch.pi * f)
        # Note : Only even for stability - not asymmetric oscillations
        poly_first_term = (t ** 2) / sigma ** 2
        poly_second_term = (t ** 4) / sigma ** 4
        poly_third_term = (t ** 6) / sigma ** 6
        polynomial = 1 - self.poly_mod[i,0]*poly_first_term  + self.poly_mod[i,1]*poly_second_term + self.poly_mod[i,2]*poly_third_term
        wavelet = polynomial * torch.exp(-t ** 2 / (2 * sigma ** 2))
        return wavelet

    def cwt(self,x, scales,i, device="cuda"):
        fdim = x.shape[1]
        x= x.flatten(start_dim=2)#.type(torch.complex64)
        space_steps = torch.arange(x.shape[-1], device=device)
        wavelets = torch.stack([self.wavelet(space_steps, 1.0 / scale,i).unsqueeze(0) for scale in self.scales]).to(device).repeat(fdim,1,1)
        x = x.repeat(1,len(scales),1)
        cwt_result = F.conv1d(x, wavelets, padding="same", groups=fdim * len(scales))
        return cwt_result