import time

import torch
import torch.nn as nn

class HyperRadialNeuralFourierCelularAutomata(nn.Module):
    def __init__(self, batch_size,no_frame_samples, input_window_size,hdc_dim,rbf_dim, device):
        super(HyperRadialNeuralFourierCelularAutomata, self).__init__()
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.hdc_dim = hdc_dim
        self.rbf_dim = rbf_dim
        self.in_scale = (1 + self.input_window_size * 2)
        self.bits = 32
        self.rbf_probes = nn.Parameter(torch.FloatTensor(self.rbf_dim,self.in_scale**2,self.bits, self.hdc_dim).uniform_(0., 2.),requires_grad=True).to(self.device)
        self.xor_ste = XorSTE.apply
        self.NCA = nn.Conv3d(in_channels=self.rbf_dim, out_channels=self.rbf_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv0 = nn.Conv3d(in_channels=self.rbf_dim, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.lin_compress = nn.Linear(in_features=self.in_scale**2*self.bits*self.hdc_dim,out_features=self.in_scale**2)
        self.lin_r= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_g= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_b= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_a= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)
        self.lin_s= nn.Linear(in_features=self.in_scale**2,out_features=self.in_scale**2)

        self.init_weights()
    def init_weights(self):
        if isinstance(self, nn.Linear):
            # torch.nn.init.xavier_uniform(self.weight)
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Conv3d):
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Parameter):
            self.data.normal_(mean=0.0, std=1.0)

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
        r = data_input[:, 0:self.in_scale, :]
        g = data_input[:, self.in_scale:self.in_scale * 2, :]
        b = data_input[:, self.in_scale * 2:self.in_scale * 3, :]
        a = data_input[:, self.in_scale * 3:self.in_scale * 4, :]
        s = structure_input


        #### HDC ENCODING
        time_in = meta_input_h2.unsqueeze(1).unsqueeze(3)
        time_out = meta_output_h2.unsqueeze(1).unsqueeze(3)
        r = torch.flatten(r,start_dim=1)
        g = torch.flatten(g, start_dim=1)
        b = torch.flatten(b, start_dim=1)
        a = torch.flatten(a, start_dim=1)
        s = torch.flatten(s, start_dim=1)
        input_dim = r.shape[1]
        sparsity=1.
        num_nonzero_elements = int(sparsity * input_dim *  self.hdc_dim)
        non_zero_indices = torch.randint(0, input_dim * self.hdc_dim, (num_nonzero_elements,)).to(self.device)
        hdc_projection_matrix = torch.zeros(self.batch_size,self.in_scale**2,self.bits, self.hdc_dim, dtype=torch.int32).to(self.device)
        hdc_projection_matrix.view(-1)[non_zero_indices] = 1
        hdc_projection_matrix.view(-1)[~non_zero_indices] = 0

        r_bin = self.binary(r.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        g_bin = self.binary(g.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        b_bin = self.binary(b.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        a_bin = self.binary(a.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)
        s_bin = self.binary(s.view(dtype=torch.int32)[:,:],self.bits).unsqueeze(3)

        hdc_projection_matrix.bitwise_xor_(r_bin)
        hdc_projection_matrix.bitwise_xor_(g_bin)
        hdc_projection_matrix.bitwise_xor_(b_bin)
        hdc_projection_matrix.bitwise_xor_(a_bin)
        hdc_projection_matrix.bitwise_xor_(s_bin)
        hdc_projection_matrix.bitwise_xor_(time_in)
        hdc_projection_matrix.bitwise_xor_(time_out)
        # Note: Check witch is better in the final result - using one hdc_projection matrix or rgbas (5) tensors separetly?
        #### HDC ENCODING -> ~30 us per first frame (so probably much faster)
        #### RBF PROBING HAMMING DISTANCE
        rbf_distances = self.xor_ste(self.rbf_probes.unsqueeze(0),hdc_projection_matrix.unsqueeze(1))
        #### RBF PROBING HAMMING DISTANCE
        dx = torch.tanh(self.NCA(rbf_distances))
        x = rbf_distances + dx
        # x = self.conv0(x)
        # x = torch.tanh(x).flatten(start_dim=1)
        # x = self.lin_compress(x)
        # x = torch.tanh(x)
        #
        # r = self.lin_r(x).view(self.batch_size, self.in_scale, self.in_scale)
        # g = self.lin_g(x).view(self.batch_size, self.in_scale, self.in_scale)
        # b = self.lin_b(x).view(self.batch_size, self.in_scale, self.in_scale)
        # a = self.lin_a(x).view(self.batch_size, self.in_scale, self.in_scale)
        # s = self.lin_s(x).view(self.batch_size, self.in_scale, self.in_scale)
        # deepS = r, g, b, a, s
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

class XorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rbf, hdc_proj):
        rbf = torch.nn.functional.hardtanh(rbf,0,2)
        return torch.bitwise_xor(rbf.int(), hdc_proj.int()).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

