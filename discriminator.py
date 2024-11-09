import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as f


class discriminator(nn.Module):
    def __init__(self, no_frame_samples, batch_size, input_window_size, device):
        super(discriminator, self).__init__()
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.in_scale = (1 + self.input_window_size * 2)
        self.no_subslice_in_tensors = 4
        self.in_data = 20
        self.activation_weight = nn.Parameter(torch.rand(1, dtype=torch.float))
        self.act = nn.ELU(alpha=2.0)
        self.cutoff_ratio = torch.tensor([0.2])
        # Definition of the weights for fftfeature
        self.weights_data_0 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.float))
        self.conv0_1x1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        self.conv0_3x3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.conv1_1x1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        self.conv1_3x3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.conv2_1x1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        self.conv2_3x3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.conv3_1x1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        self.conv3_3x3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)

        # Definition of Walsh-Hadamard rescale layers
        self.Walsh_Hadamard_rescaler_l0wh = nn.Linear(in_features=256, out_features=(5 * self.in_scale ** 2))
        self.Walsh_Hadamard_rescaler_l1wh = nn.Linear(in_features=(5 * self.in_scale ** 2),
                                                      out_features=(5 * self.in_scale ** 2))

        self.lin_fusion = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2) * 5)
        self.r = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.g = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.b = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.a = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.s = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))

        self.disc_output =nn.utils.spectral_norm(nn.Linear(self.in_scale ** 2 * 5, 1, bias=True))
        self.init_weights()

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

    def forward(self, disc_data, g_model_data, shuffle_idx):
        (_, _, meta_input_h1, meta_input_h2, meta_input_h3,
         _, _, noise_var_in, fmot_in, meta_output_h1, meta_output_h2,
         meta_output_h3, _, _, noise_var_out) = g_model_data

        meta_input_h2 = torch.cat([meta_input_h2, meta_input_h2], dim=0)[shuffle_idx]
        meta_input_h3 = torch.cat([meta_input_h3, meta_input_h3], dim=0)[shuffle_idx]
        noise_var_in = torch.cat([noise_var_in, fmot_in, noise_var_in], dim=0)[shuffle_idx]
        meta_output_h2 = torch.cat([meta_output_h2, meta_output_h2], dim=0)[shuffle_idx]
        meta_output_h3 = torch.cat([meta_output_h3, meta_output_h3], dim=0)[shuffle_idx]
        noise_var_out = torch.cat([noise_var_out, noise_var_out], dim=0)[shuffle_idx]

        meta_central_points = torch.cat([meta_input_h3.float(), meta_output_h3.float()], dim=1)
        noise_var = torch.cat([noise_var_in, noise_var_out], dim=1)
        meta_step = torch.cat([meta_input_h2.float(), meta_output_h2.float()], dim=1)
        noise_variance = 0.2
        x = disc_data[shuffle_idx] + torch.nan_to_num(noise_variance * torch.rand_like(disc_data[shuffle_idx]),nan=0.0)
        space_time = self.WalshHadamardSpaceTimeFeature(meta_central_points, meta_step, noise_var)

        # B, C, H, W = disc_data.shape
        # fft_data = torch.fft.fft2(disc_data, dim=(-2, -1))
        # fft_data_shifted = torch.fft.fftshift(fft_data, dim=(-2, -1))
        # mask = torch.ones((H, W), device=self.device)
        # center_x, center_y = H // 2, W // 2
        # cutoff_x = int(self.cutoff_ratio * H // 2)
        # cutoff_y = int(self.cutoff_ratio * W // 2)
        # mask[center_x - cutoff_x:center_x + cutoff_x, center_y - cutoff_y:center_y + cutoff_y] = 0.0
        # mask = mask.unsqueeze(0).unsqueeze(0)
        # fft_data_filtered = fft_data_shifted * mask
        # fft_data_original = torch.fft.ifftshift(fft_data_filtered, dim=(-2, -1))
        # x = torch.fft.ifft2(fft_data_original, dim=(-2, -1))
        # amp = x.real + x.imag
        # scaled_amp = amp / 3.0
        # phase = torch.atan2(x.real, x.imag)
        # sin_component = scaled_amp * torch.sin(phase)
        # cos_component = scaled_amp * torch.cos(phase)
        # sinc_component = scaled_amp * torch.sinc(phase / torch.pi)
        # x = sin_component + cos_component + sinc_component

        x_1 =  self.activate(self.conv0_1x1(x))
        x_3 =  self.activate(self.conv0_3x3(x))
        xa = x_1+x_3
        x_1 = self.activate(self.conv1_1x1(xa))+x
        x_3 = self.activate(self.conv1_3x3(xa))+x
        xb = x_1 + x_3
        x_1 = self.activate(self.conv2_1x1(xb))+xa
        x_3 = self.activate(self.conv2_3x3(xb))+xa
        xc = x_1 + x_3
        x_1 = self.activate(self.conv3_1x1(xc))+xb
        x_3 = self.activate(self.conv3_3x3(xc))+xb
        xd = x_1 + x_3
        x = torch.flatten(xd, start_dim=1)
        x = self.activate(self.lin_fusion(x))+space_time
        r = self.activate(self.r(x))
        g = self.activate(self.g(x))
        b = self.activate(self.b(x))
        a = self.activate(self.a(x))
        s = self.activate(self.s(x))
        x = torch.cat([r, g, b, a, s], dim=1)
        out = torch.sigmoid(self.disc_output(x))
        return out

    def WalshHadamardSpaceTimeFeature(self, meta_central_points, meta_step, noise_var):
        # NOTE: Walsh-Hadamard transform for space and time coding
        space_time = torch.cat([meta_central_points, meta_step, noise_var], dim=1)
        bit_padding = torch.zeros((self.batch_size, 256 - space_time.shape[1])).to(self.device)
        space_time = torch.cat([space_time, bit_padding], dim=1)
        length = space_time.shape[1]
        assert (length & (length - 1)) == 0, "Length must be a power of 2"
        bit = length
        len_tens = torch.tensor(length)
        stages = torch.log2(len_tens)
        stages = torch.arange(0, stages.int())
        for _ in stages:
            bit >>= 1
            indices = torch.arange(length).view(1, -1)
            indices_i = (indices & ~bit).flatten()
            indices_j = (indices | bit).flatten()
            result_i = space_time[:, indices_i]
            result_j = space_time[:, indices_j]
            space_time[:, indices_i] = result_i + result_j
            space_time[:, indices_j] = result_i - result_j

        space_time /= len_tens  # normalize
        space_time = self.activate(self.Walsh_Hadamard_rescaler_l0wh(space_time))
        space_time = self.activate(self.Walsh_Hadamard_rescaler_l1wh(space_time))
        return space_time.real

    def activate(self, x):
        return self.act(x)