import time

import torch
import torch.nn as nn

class HyperRadialNeuralFourierCelularAutomata(nn.Module):
    def __init__(self, batch_size,no_frame_samples, input_window_size,hdc_dim, device):
        super(HyperRadialNeuralFourierCelularAutomata, self).__init__()
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.hdc_dim = hdc_dim
        self.in_scale = (1 + self.input_window_size * 2)
        self.parameters_temp = nn.Parameter(torch.rand((self.in_scale, self.in_scale), dtype=torch.float))
        self.bits = 32
        self.hdc_projection_matrix = torch.zeros(self.batch_size,self.in_scale**2,self.bits, self.hdc_dim, dtype=torch.int32).to(self.device)

    def init_weights(self):
        if isinstance(self, nn.Linear):
            # torch.nn.init.xavier_uniform(self.weight)
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Conv1d):
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Parameter):
            self.data.normal_(mean=0.0, std=1.0)

    def weight_reset(self: nn.Module):
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()
        # NOTE : Making sure that nn.conv2d and nn.linear will be reset
        if isinstance(self, nn.Conv2d) or isinstance(self, nn.Linear):
            self.reset_parameters()

    def forward(self, din):
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
        r = data_input[:, 0:self.in_scale, :]
        g = data_input[:, self.in_scale:self.in_scale * 2, :]
        b = data_input[:, self.in_scale * 2:self.in_scale * 3, :]
        a = data_input[:, self.in_scale * 3:self.in_scale * 4, :]
        s = structure_input*self.parameters_temp
        deepS = r, g, b, a, s

        #### HDC ENCODING
        r = torch.flatten(r,start_dim=1)
        g = torch.flatten(g, start_dim=1)
        b = torch.flatten(b, start_dim=1)
        a = torch.flatten(a, start_dim=1)
        s = torch.flatten(s, start_dim=1)
        input_dim = r.shape[1]
        sparsity=0.5
        num_nonzero_elements = int(sparsity * input_dim *  self.hdc_dim)
        non_zero_indices = torch.randint(0, input_dim * self.hdc_dim, (num_nonzero_elements,)).to(self.device)
        self.hdc_projection_matrix.view(-1)[non_zero_indices] = 1
        self.hdc_projection_matrix.view(-1)[~non_zero_indices] = 0

        r_bin = self.binary(r.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        g_bin = self.binary(g.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        b_bin = self.binary(b.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        a_bin = self.binary(a.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        s_bin = self.binary(s.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)


        r_bin = r_bin.repeat(1,1,1, self.hdc_dim)
        g_bin = g_bin.repeat(1,1,1, self.hdc_dim)
        b_bin = b_bin.repeat(1,1,1, self.hdc_dim)
        a_bin = a_bin.repeat(1,1,1, self.hdc_dim)
        s_bin = s_bin.repeat(1,1,1, self.hdc_dim)

        r_bin.bitwise_xor_(self.hdc_projection_matrix)
        g_bin.bitwise_xor_(self.hdc_projection_matrix)
        b_bin.bitwise_xor_(self.hdc_projection_matrix)
        a_bin.bitwise_xor_(self.hdc_projection_matrix)
        s_bin.bitwise_xor_(self.hdc_projection_matrix)
        #### HDC ENCODING -> ~30 us per first frame (so probably much faster)
        #### RBF PROBING

        t_stop = time.perf_counter()
        print(r_bin.shape)
        print("model internal time : ",((t_stop-t_start)*1e6)/self.batch_size, "[us]")
        time.sleep(10000)
        self.batch_size = old_batch_size
        return r, g, b, a, s, deepS

    def binary(self,x, bits):
        mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


