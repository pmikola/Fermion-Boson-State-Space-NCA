import math
import time

import torch
import torch.nn as nn
from linformer import Linformer

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
        self.loss_weights = nn.Parameter(torch.ones(12))
        self.modes = 32
        self.uplift_meta_0 = nn.Linear(200,15*self.modes)
        self.uplift_meta_1 = nn.Linear(15*self.modes, 5 * self.in_scale ** 2)
        self.uplift_meta = nn.Conv2d(in_channels=5, out_channels=self.hdc_dim,kernel_size=1)
        self.uplift_data = nn.Conv2d(in_channels=5, out_channels=self.hdc_dim, kernel_size=1)
        self.cross_correlate_in = nn.Conv3d(in_channels=1, out_channels=self.hdc_dim, kernel_size=3, stride=1, padding=1)
        self.cross_correlate_out = nn.Conv3d(in_channels=self.hdc_dim, out_channels=self.hdc_dim, kernel_size=3, stride=1, padding=1)
        self.nca_steps = nca_steps
        self.act = nn.ELU(alpha=2.0)
        # self.act = nn.GELU()
        self.NCA = NCA(self.batch_size,self.hdc_dim, self.nca_steps, self.device)
        self.downlift_data = nn.Conv3d(in_channels=self.hdc_dim,out_channels=self.hdc_dim,kernel_size=1)
        self.rgbas = nn.Conv3d(in_channels=self.hdc_dim,out_channels=self.hdc_dim,kernel_size=1)
        self.r = nn.Conv2d(in_channels=self.hdc_dim, out_channels=1, kernel_size=1)
        self.g = nn.Conv2d(in_channels=self.hdc_dim, out_channels=1, kernel_size=1)
        self.b = nn.Conv2d(in_channels=self.hdc_dim, out_channels=1, kernel_size=1)
        self.a = nn.Conv2d(in_channels=self.hdc_dim, out_channels=1, kernel_size=1)
        self.s = nn.Conv2d(in_channels=self.hdc_dim, out_channels=1, kernel_size=1)
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
        old_batch_size = self.batch_size
        (data_input, structure_input, meta_input_h1, meta_input_h2, meta_input_h3,
         meta_input_h4, meta_input_h5, noise_var_in_binary, fmot_in_binary, meta_output_h1, meta_output_h2,
         meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out) = din
        if data_input.shape[0] != self.batch_size:
            self.batch_size = data_input.shape[0]
        #################################################################
        # s = torch.flatten(structure_input,start_dim=1)
        # data = torch.flatten(data_input,start_dim=1)
        r = data_input[:, 0:self.in_scale, :].unsqueeze(1)
        g = data_input[:, self.in_scale:self.in_scale * 2, :].unsqueeze(1)
        b = data_input[:, self.in_scale * 2:self.in_scale * 3, :].unsqueeze(1)
        a = data_input[:, self.in_scale * 3:self.in_scale * 4, :].unsqueeze(1)
        s = structure_input.unsqueeze(1)
        data = torch.cat([r,g,b,a,s],dim=1)
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
        x = self.act(self.cross_correlate_in(x))
        x,nca_var,ortho_mean,ortho_max = self.NCA(x,meta_embeddings,spiking_probabilities,self.batch_size)

        x = self.act(self.cross_correlate_out(x))
        x = self.act(self.downlift_data(x))
        rgbas = self.act(self.rgbas(x))
        r = self.r(rgbas[:, :,  0, :, :].squeeze(1)).squeeze(1)
        g = self.g(rgbas[:, :,  1, :, :].squeeze(1)).squeeze(1)
        b = self.b(rgbas[:, :,  2, :, :].squeeze(1)).squeeze(1)
        a = self.a(rgbas[:, :,  3, :, :].squeeze(1)).squeeze(1)
        s = self.s(rgbas[:, :,  4, :, :].squeeze(1)).squeeze(1)

        deepS = r, g, b, a, s
        #torch.cuda.current_stream().synchronize()
        #t_stop = time.perf_counter()
        # print("model internal time patch : ", ((t_stop - t_start) * 1e3) / self.batch_size, "[ms]")
        # time.sleep(10000)
        self.batch_size = old_batch_size
        return r, g, b, a, s, deepS,nca_var,ortho_mean,ortho_max,self.loss_weights

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