import math
import time

import torch
import torch.nn as nn

from NCA import NCA


class HyperRadialNeuralFourierCelularAutomata(nn.Module):
    def __init__(self, batch_size,no_frame_samples, input_window_size,hdc_dim,rbf_dim,nca_steps, device):
        super(HyperRadialNeuralFourierCelularAutomata, self).__init__()
        self.last_frame = None
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.hdc_dim = hdc_dim
        self.rbf_dim = rbf_dim
        self.in_scale = (1 + self.input_window_size * 2)
        self.modes = 32
        #self.rbf_probes = nn.Parameter(torch.rand(self.rbf_dim,5,self.in_scale,self.in_scale, self.hdc_dim), requires_grad=True).to(self.device)
        #torch.nn.init.xavier_uniform_(self.rbf_probes)
        #self.hdc_projection_matrix = nn.Parameter(torch.rand(1,5, self.hdc_dim, device=self.device,requires_grad=True)* 2 - 1)
        #torch.nn.init.xavier_uniform_(self.hdc_projection_matrix)
        self.compress_time = nn.Conv2d(in_channels=self.modes, out_channels=5, kernel_size=1)
        #self.gate = nn.Parameter(torch.rand(1), requires_grad=True).to(self.device)
        #self.compress = nn.Conv3d(in_channels=self.rbf_dim*self.hdc_dim,out_channels=self.rbf_dim,kernel_size=1)
        self.uplift = nn.Conv2d(in_channels=5, out_channels=15, kernel_size=1)
        self.nca_steps = nca_steps
        self.act = nn.ELU(alpha=1.0)
        self.NCA = NCA(15,self.nca_steps,self.device)
        self.compress_NCA_out = nn.Conv2d(in_channels=15,out_channels=5,kernel_size=3,stride=1,padding=1)
        self.r = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.g = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.b = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.a = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.s = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.00)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.00)

    def weight_reset(self: nn.Module):
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()
        # NOTE : Making sure that nn.conv2d and nn.linear will be reset
        if isinstance(self, nn.Conv3d) or isinstance(self, nn.Linear):
            self.reset_parameters()

    def forward(self, din,spiking_probabilities):
        #torch.cuda.synchronize()
        #t_start = time.perf_counter()
        old_batch_size = self.batch_size
        (data_input, structure_input, meta_input_h1, meta_input_h2, meta_input_h3,
         meta_input_h4, meta_input_h5, noise_var_in_binary, fmot_in_binary, meta_output_h1, meta_output_h2,
         meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out) = din
        if data_input.shape[0] != self.batch_size:
            self.batch_size = data_input.shape[0] * old_batch_size
            din = [data_input, structure_input, meta_input_h1, meta_input_h2, meta_input_h3,
                   meta_input_h4, meta_input_h5, noise_var_in_binary, fmot_in_binary, meta_output_h1, meta_output_h2,
                   meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out]
            stacked_vars = []
            for var in din:
                permuted_var = var.permute(1, 0, *range(2, var.ndimension()))
                stacked_var = permuted_var.reshape(-1, *var.shape[2:])
                stacked_vars.append(stacked_var)
            (data_input, structure_input, meta_input_h1, meta_input_h2,
             meta_input_h3, meta_input_h4, meta_input_h5, noise_var_in_binary, fmot_in_binary,
             meta_output_h1, meta_output_h2, meta_output_h3, meta_output_h4,
             meta_output_h5, noise_var_out) = stacked_vars

        #################################################################
        # s = torch.flatten(structure_input,start_dim=1)
        # data = torch.flatten(data_input,start_dim=1)
        r = data_input[:, 0:self.in_scale, :].unsqueeze(1)
        g = data_input[:, self.in_scale:self.in_scale * 2, :].unsqueeze(1)
        b = data_input[:, self.in_scale * 2:self.in_scale * 3, :].unsqueeze(1)
        a = data_input[:, self.in_scale * 3:self.in_scale * 4, :].unsqueeze(1)
        s = structure_input.unsqueeze(1)
        data = torch.cat([r,g,b,a,s],dim=1)
        time_in = meta_input_h5
        time_out = meta_output_h5
        time_encoded = self.meta_encoding(time_in,time_out, self.modes, self.last_frame)
        time_encoded = time_encoded.unsqueeze(2).unsqueeze(3).expand(-1, -1, data.shape[-2], data.shape[-1])
        time_encoded =  self.act(self.compress_time(time_encoded))
        data = data * time_encoded
        x = self.act(self.uplift(data))
        x,nca_var = self.NCA(x,spiking_probabilities)
        x = self.act(self.compress_NCA_out(x))
        r = self.r(x).squeeze()
        g = self.g(x).squeeze()
        b = self.b(x).squeeze()
        a = self.a(x).squeeze()
        s = self.s(x).squeeze()
        deepS = r, g, b, a, s
        #torch.cuda.current_stream().synchronize()
        #t_stop = time.perf_counter()
        # print("model internal time patch : ", ((t_stop - t_start) * 1e3) / self.batch_size, "[ms]")
        # time.sleep(10000)
        self.batch_size = old_batch_size
        return r, g, b, a, s, deepS,nca_var

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
