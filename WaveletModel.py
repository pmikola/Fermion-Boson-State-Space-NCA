import torch
from torch import nn
import torch.nn.functional as F

class WaveletModel(nn.Module):
    def __init__(self, num_scales, max_scale_value, batch_size, channels, height, width,device):
        super(WaveletModel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.scales = nn.Parameter(torch.linspace(1, max_scale_value, num_scales)).to(self.device)

    def forward(self, x):
        cwt_transformed = self.cwt(x,self.scales)
        return cwt_transformed

    def wavelet(self,t, f):
        # Note: morlet wavelet for now
        sigma = 6 / (2 * torch.pi * f)
        return torch.exp(2j * torch.pi * f * t) * torch.exp(-t ** 2 / (2 * sigma ** 2))

    def cwt(self,x, scales, device="cuda"):
        fdim = x.shape[1]
        x= x.flatten(start_dim=2).type(torch.complex64)
        time_steps = torch.arange(x.shape[-1], device=device)
        morlet_wavelets = torch.stack([self.wavelet(time_steps, 1.0 / scale).unsqueeze(0) for scale in self.scales]).to(device).repeat(fdim,1,1)
        x = x.repeat(1,len(scales),1)
        cwt_result = F.conv1d(x, morlet_wavelets, padding="same", groups=fdim * len(scales))
        return cwt_result