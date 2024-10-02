import math
import time

import torch
import torch.nn as nn

from NCA import NCA


class HyperRadialNeuralFourierCelularAutomata(nn.Module):
    def __init__(self, batch_size,no_frame_samples, input_window_size,hdc_dim,rbf_dim, device):
        super(HyperRadialNeuralFourierCelularAutomata, self).__init__()
        self.last_frame = None
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.hdc_dim = hdc_dim
        self.rbf_dim = rbf_dim
        self.in_scale = (1 + self.input_window_size * 2)
        self.bits = 32
        self.modes = 16
        self.rbf_probes = nn.Parameter(torch.FloatTensor(self.rbf_dim,5*self.in_scale**2+self.modes, self.hdc_dim).uniform_(0., 2.), requires_grad=False).to(self.device)
        self.xor_ste = XorSTE.apply
        self.compress = nn.Conv2d(in_channels=self.rbf_dim,out_channels=5,kernel_size=1)
        self.NCA = NCA(5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.in_scale**2,1))
        self.lin_r= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_g= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_b= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_a= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_s= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Parameter):
                m.data.normal_(mean=0.0, std=1.0)

    def weight_reset(self: nn.Module):
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()
        # NOTE : Making sure that nn.conv2d and nn.linear will be reset
        if isinstance(self, nn.Conv3d) or isinstance(self, nn.Linear):
            self.reset_parameters()

    def forward(self, din):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
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
        print(data.shape)
        #### HDC ENCODING
        input_dim = data.shape[1]
        sparsity=1.
        num_nonzero_elements = int(sparsity * input_dim * self.hdc_dim)
        non_zero_indices = torch.randint(0, input_dim * self.hdc_dim, (num_nonzero_elements,), device=self.device)
        signs = torch.randint(0, 2, (num_nonzero_elements,), device=self.device,dtype=torch.int32) * 2 - 1
        hdc_projection_matrix = torch.zeros(self.batch_size,input_dim+self.modes, self.hdc_dim, device=self.device)
        hdc_projection_matrix.view(-1)[non_zero_indices] = signs.float()
        time_in = meta_input_h5
        time_out = meta_output_h5
        time_encoded = self.meta_encoding(time_in,time_out, self.modes, self.last_frame)
        data = torch.cat([data, time_encoded], dim=1)

        # Note: Check witch is better in the final result - using one hdc_projection matrix or rgbas (5) tensors separetly?
        #### HDC ENCODING -> ~30 us per first frame (so probably much faster)
        #### RBF PROBING HAMMING DISTANCE

        data_projection = torch.einsum('bi,bjd->bjd',data,hdc_projection_matrix)
        rbf_distances = self.rbf_probes.unsqueeze(0) - data_projection.unsqueeze(1)
        rbf_distances = rbf_distances ** 2

        #### RBF PROBING HAMMING DISTANCE
        print(rbf_distances.shape)
        x = torch.tanh(self.compress(rbf_distances))
        print(x.shape)
        x = torch.tanh(self.NCA(x,20))
        print(x.shape)
        x = self.adaptive_pool(x).squeeze(-1)
        print(x.shape)
        r = x[: , 0 ,:].view(self.batch_size,self.in_scale,self.in_scale)
        g = x[: , 1, :].view(self.batch_size,self.in_scale,self.in_scale)
        b = x[: , 2, :].view(self.batch_size,self.in_scale,self.in_scale)
        a = x[: , 3, :].view(self.batch_size,self.in_scale,self.in_scale)
        s = x[: , 4, :].view(self.batch_size,self.in_scale,self.in_scale)
        deepS = r, g, b, a, s
        torch.cuda.current_stream().synchronize()
        t_stop = time.perf_counter()
        print("model internal time patch : ", ((t_stop - t_start) * 1e3) / self.batch_size, "[ms]")
        time.sleep(10000)
        self.batch_size = old_batch_size
        return r, g, b, a, s, deepS

    @staticmethod
    def binary(x, bits):
        mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


    def meta_encoding(self,meta_in,meta_out, dim, max_time_step):
        pos_in = meta_in / max_time_step
        pos_out = meta_out / max_time_step
        pe = torch.zeros(self.batch_size,dim).to(self.device)
        for i in range(dim):
            angle_in = pos_in / (10000 ** ((2 * (i // 2)) / dim))
            angle_out = pos_out / (10000 ** ((2 * (i // 2)) / dim))
            angle_in = angle_in.unsqueeze(0)
            angle_out = angle_out.unsqueeze(0)
            if i % 2 == 0:
                pe[:,i] = (torch.sin(angle_in)+torch.cos(angle_in))/2
            else:
                pe[:,i] = (torch.cos(angle_in)+torch.sin(angle_out))/2
        return pe

class XorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rbf, hdc_proj):
        rbf = torch.nn.functional.hardtanh(rbf,0,2)
        return torch.bitwise_xor(rbf.int(), hdc_proj.int()).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

