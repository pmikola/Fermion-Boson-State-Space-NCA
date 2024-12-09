import copy
import itertools
import os.path
import random
import struct
import time
from statistics import mean

import matplotlib

from piq import VSILoss
from torch import nn
import gpustat
from geomloss import SamplesLoss
# import ssim
from pytorch_msssim import SSIM, MS_SSIM
import WinTmp
import kornia
import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from torch.autograd import grad
import torch.nn.utils as nn_utils
import torch.nn.functional as f
from torchvision.transforms import transforms
from torchvision.transforms.v2.functional import adjust_hue_image
from VSI_Metric import VS_ESSIM


class teacher(nn.Module):
    def __init__(self, model,discriminator, device):
        super(teacher, self).__init__()
        #self.t = None
        self.mpl_backend = plt.get_backend()
        self.validation_dataset = None
        self.max_seed = int(1e2)
        self.model = model
        self.discriminator = discriminator
        self.device = device
        self.fsim = None
        self.period = 1
        self.no_of_periods = 1
        self.data_tensor = None
        self.meta_tensor = None
        self.meta_binary = None
        self.field_names = None
        self.no_frame_samples, self.first_frame, self.last_frame, self.frame_skip = None, None, None, None
       # self.ssim_loss = SSIM(data_range=1., channel=5,nonnegative_ssim=True,K=(0.01, 0.1),win_size=7).to(self.device)
        self.vsi = VS_ESSIM().to(self.device)
        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)

        self.data_input = None
        self.structure_input = None  #torch.zeros((self.model.batch_size,self.model.in_scale,self.model.in_scale),requires_grad=True)
        self.meta_input_h1 = None
        self.meta_input_h2 = None
        self.meta_input_h3 = None
        self.meta_input_h4 = None
        self.meta_input_h5 = None
        self.noise_diff_in = None
        self.fmot_in = None
        self.fmot_in_binary = None
        self.data_output = None
        self.structure_output = None  #torch.zeros((self.model.batch_size,self.model.in_scale,self.model.in_scale),requires_grad=True)
        self.meta_output_h1 = None
        self.meta_output_h2 = None
        self.meta_output_h3 = None
        self.meta_output_h4 = None
        self.meta_output_h5 = None
        self.noise_diff_out = None

        self.data_input_val = None
        self.data_output_val = None
        self.structure_input_val = None
        self.structure_output_val = None
        self.meta_input_h1_val = None
        self.meta_input_h2_val = None
        self.meta_input_h3_val = None
        self.meta_input_h4_val = None
        self.meta_input_h5_val = None
        self.noise_var_in_val = None
        self.fmot_in_val = None
        self.fmot_in_binary_val = None
        self.meta_output_h1_val = None
        self.meta_output_h2_val = None
        self.meta_output_h3_val = None
        self.meta_output_h4_val = None
        self.meta_output_h5_val = None
        self.noise_var_out_val = None
        self.mask = None
        self.epoch = 0
        self.num_of_epochs = 0
        self.train_loss = []
        self.disc_loss = []
        self.val_loss = []
        self.vsi_loss = []
        self.entropy_loss = []
        self.grad_penalty = []
        self.kl_loss = []
        self.sink_loss = []
        self.critical_loss = []
        self.hf_e_loss = []
        self.b_loss = []
        self.value_loss = []
        self.diff_fft_loss = []
        self.fft_loss = []
        self.diff_loss = []
        self.log_det_jacobian_loss =[]
        self.freq_loss = []
        self.val_vsi_loss = []
        self.val_entropy_loss = []
        self.val_grad_penalty = []
        self.val_kl_loss = []
        self.val_sink_loss = []
        self.val_critical_loss = []
        self.val_hf_e_loss = []
        self.val_b_loss = []
        self.val_value_loss = []
        self.val_diff_fft_loss = []
        self.val_fft_loss = []
        self.val_diff_loss = []
        self.val_log_det_jacobian_loss = []
        self.val_freq_loss = []
        self.cpu_temp = []
        self.gpu_temp = []
        self.h = None
        self.w = None
        self.n_frames = None
        self.fuel_slices = None
        self.r_slices = None
        self.g_slices = None
        self.b_slices = None
        self.alpha_slices = None
        self.meta_binary_slices = None
        self.loss_coeffs = torch.ones(5,requires_grad=False).to(self.device)
        self.preprocess = transforms.Compose([
                        AddGaussianNoise(self.device, 0, 1),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        # self.validator = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
        #                        regressor_dataset="kadid10k")
        # self.validator.eval().to(device)


    def generate_structure(self):
        no_structure = random.randint(0, self.fsim.grid_size_y - self.fsim.N_boundary)
        self.fsim.idx = torch.randint(low=self.fsim.N_boundary, high=self.fsim.grid_size_x - self.fsim.N_boundary,
                                      size=(no_structure,))
        self.fsim.idy = torch.randint(low=self.fsim.N_boundary, high=self.fsim.grid_size_y - self.fsim.N_boundary,
                                      size=(no_structure,))
        # self.fsim.idx = random.sample(range(self.fsim.N_boundary, self.fsim.grid_size_x - self.fsim.N_boundary), no_structure)
        # self.fsim.idy = random.sample(range(self.fsim.N_boundary, self.fsim.grid_size_y - self.fsim.N_boundary),no_structure)
        self.fsim.idx_u = self.fsim.idx
        self.fsim.idy_u = self.fsim.idy
        self.fsim.idx_v = self.fsim.idx
        self.fsim.idy_v = self.fsim.idy

    def generate_sim_params(self):
        pass

    def data_preparation(self, create_val_dataset=0):
        if self.data_tensor is None:
            folder_names = ['v', 'u', 'velocity_magnitude', 'fuel_density', 'oxidizer_density',
                            'product_density', 'pressure', 'temperature', 'rgb', 'alpha']
            data_tensor = []
            meta_tensor = []
            meta_binary = []
            field_names = []

            for name in folder_names:
                if os.path.exists(name):
                    for i in range(self.first_frame, self.last_frame, self.frame_skip):
                        if name == 'rgb':
                            ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                            for j in range(0, 3):
                                data_tensor.append(ptfile['data'][:, :, j] / 255.)
                                meta_tensor.append(ptfile['metadata'])
                                field_names.append(ptfile['name'])
                        else:
                            ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                            data_tensor.append(ptfile['data'])
                            meta_tensor.append(ptfile['metadata'])
                            field_names.append(ptfile['name'])

            self.data_tensor = torch.stack(data_tensor, dim=0)
            self.meta_tensor = torch.stack(meta_tensor, dim=0)

            for i in range(self.meta_tensor.shape[0]):
                meta_temp = []
                for j in range(self.meta_tensor.shape[1]):
                    binary_var = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', self.meta_tensor[i, j]))
                    # Note : '!f' The '!' ensures that
                    #     it's in network byte order (big-endian) and the 'f' says that it should be
                    #     packed as a float. Use d for double precision
                    binary_var = np.frombuffer(binary_var.encode("ascii"), dtype='u1') - 48
                    # binary_var = torch.tensor([int(bit) for bit in binary_var], dtype=torch.uint8) - 48
                    meta_temp.append(binary_var)
                meta_binary.append(meta_temp)
            self.meta_binary = torch.from_numpy(np.array(meta_binary))
            self.field_names = field_names
            fdens_idx = np.array([i for i, x in enumerate(self.field_names) if x == "fuel_density"])
            # frame_samples = random.sample(list(set(fdens_idx)), k=self.no_frame_samples)
            f_dens_pos = len(fdens_idx)

            # fdens_idx = fdens_idx[frame_samples]
            rgb_idx = np.array([i for i, x in enumerate(self.field_names) if x == "rgb"])
            r_idx = rgb_idx[::3]  # [frame_samples]
            g_idx = rgb_idx[::3] + 1  # [frame_samples]
            b_idx = rgb_idx[::3] + 2  # [frame_samples]
            alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])  # [frame_samples]
            fuel_slices = self.data_tensor[fdens_idx]
            self.w = fuel_slices.shape[1]
            self.h = fuel_slices.shape[2]
            self.n_frames = fuel_slices.shape[0]
            min_val = fuel_slices.min()
            max_val = fuel_slices.max()
            self.fuel_slices = (fuel_slices - min_val) / ((max_val - min_val) + 1e-12)
            self.r_slices = self.data_tensor[r_idx]
            self.g_slices = self.data_tensor[g_idx]
            self.b_slices = self.data_tensor[b_idx]
            self.alpha_slices = self.data_tensor[alpha_idx]
            self.meta_binary_slices = self.meta_binary[fdens_idx]
        else:
            pass

        # gt = np.stack((r_slices[0].cpu().numpy(), g_slices[0].cpu().numpy(), b_slices[0].cpu().numpy()), axis=2)
        # print(gt.shape)
        # plt.imshow(gt.astype(np.uint8) , alpha=alpha_slices[0].cpu().numpy())
        # plt.show()
        x_range = range(self.fsim.N_boundary + self.input_window_size,
                        self.w - self.fsim.N_boundary - self.input_window_size)
        y_range = range(self.fsim.N_boundary + self.input_window_size,
                        self.h - self.fsim.N_boundary - self.input_window_size)
        data_input = []
        structure_input = []
        meta_input_h1 = []
        meta_input_h2 = []
        meta_input_h3 = []
        meta_input_h4 = []
        meta_input_h5 = []
        noise_var_in = []
        fmot_in = []
        fmot_in_binary = []
        data_output = []
        structure_output = []
        meta_output_h1 = []
        meta_output_h2 = []
        meta_output_h3 = []
        meta_output_h4 = []
        meta_output_h5 = []
        noise_var_out = []
        frame = 0

        while not frame >= self.batch_size * 2:
            choose_diffrent_frame = 0
            noise_flag = torch.randint(low=0, high=10, size=(1,))
            # Note: Below is the pseudo diffusion process
            if create_val_dataset == 1:
                #noise_mod = 0.
                noise_power_in = torch.zeros(1)  # torch.rand(size=(1,))
                noise_power_out = torch.zeros(1)
            else:
                #noise_mod = 1.
                noise_power_in =  torch.randint(1, 20, size=(1,))#torch.rand(size=(1,))
                noise_power_out = noise_power_in-1#torch.randint(0, 20, size=(1,))
            if noise_flag < 5:
                noise_variance_in = torch.tensor(0.).to(self.device)
                noise_variance_out = torch.tensor(0.).to(self.device)
                noise_variance_in_binary = torch.zeros(32).to(self.device)
                noise_variance_out_binary = torch.zeros(32).to(self.device)
            elif 5 < noise_flag < 8:
                noise_variance_in = noise_power_in #* noise_mod
                noise_variance_in_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_in).tobytes())
                noise_variance_in = noise_variance_in.to(self.device)
                noise_variance_in_binary = [int(noise_variance_in_binary[i], 2) for i in
                                            range(0, len(noise_variance_in_binary), 1)]
                noise_variance_in_binary = torch.tensor(np.array(noise_variance_in_binary)).to(self.device)
                noise_variance_out = torch.tensor(0.).to(self.device)
                noise_variance_out_binary = torch.zeros(32).to(self.device)
            elif 8 < noise_flag < 10:
                noise_variance_out = noise_power_out #* noise_mod
                noise_variance_out_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_out).tobytes())
                noise_variance_out = noise_variance_out.to(self.device)
                noise_variance_out_binary = [int(noise_variance_out_binary[i], 2) for i in
                                             range(0, len(noise_variance_out_binary), 1)]
                noise_variance_out_binary = torch.tensor(np.array(noise_variance_out_binary)).to(self.device)
                noise_variance_in = torch.tensor(0.).to(self.device)
                noise_variance_in_binary = torch.zeros(32).to(self.device)
            else:
                noise_variance_in = noise_power_in #* noise_mod
                noise_variance_in_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_in).tobytes())
                noise_variance_in = noise_variance_in.to(self.device)
                noise_variance_in_binary = [int(noise_variance_in_binary[i], 2) for i in
                                            range(0, len(noise_variance_in_binary), 1)]
                noise_variance_in_binary = torch.tensor(np.array(noise_variance_in_binary)).to(self.device)
                noise_variance_out = noise_power_out #* noise_mod
                noise_variance_out_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_out).tobytes())
                noise_variance_out = noise_variance_out.to(self.device)
                noise_variance_out_binary = [int(noise_variance_out_binary[i], 2) for i in
                                             range(0, len(noise_variance_out_binary), 1)]
                noise_variance_out_binary = torch.tensor(np.array(noise_variance_out_binary)).to(self.device)

            # Note: Flow Matching OT Noise gen
            fmot_coef = torch.rand(size=(1,))#torch.ones(size=(1,))
            fmot_coef_binary = ''.join(f'{c:08b}' for c in np.float32(fmot_coef).tobytes())
            fmot_coef_binary = [int(fmot_coef_binary[i], 2) for i in range(0, len(fmot_coef_binary), 1)]
            fmot_coef_binary = torch.tensor(np.array(fmot_coef_binary)).to(self.device)

            idx_input = random.choice(range(0, self.n_frames-1))
            central_point_x_in = random.sample(x_range, 1)[0]
            central_point_y_in = random.sample(y_range, 1)[0]
            window_x_in = np.array(
                range(central_point_x_in - self.input_window_size, central_point_x_in + self.input_window_size + 1))
            window_y_in = np.array(
                range(central_point_y_in - self.input_window_size, central_point_y_in + self.input_window_size + 1))
            central_point_x_binary_in = "{0:010b}".format(central_point_x_in)
            central_point_x_binary_in = torch.tensor(np.array([int(d) for d in central_point_x_binary_in]))
            central_point_y_binary_in = "{0:010b}".format(central_point_y_in)
            central_point_y_binary_in = torch.tensor(np.array([int(d) for d in central_point_y_binary_in]))
            slice_x_in = slice(window_x_in[0], window_x_in[-1] + 1)
            slice_y_in = slice(window_y_in[0], window_y_in[-1] + 1)

            idx_output =idx_input+1 #random.choice(range(0, self.n_frames))
            spatial_offset_x = random.randint(int(-self.input_window_size), int(self.input_window_size))
            spatial_offset_y = random.randint(int(-self.input_window_size), int(self.input_window_size))
            central_point_x_out = central_point_x_in  #+ spatial_offset_x
            central_point_y_out = central_point_y_in  #+ spatial_offset_y

            window_x_out = np.array(
                range(central_point_x_out - self.input_window_size, central_point_x_out + self.input_window_size + 1))
            window_y_out = np.array(
                range(central_point_y_out - self.input_window_size, central_point_y_out + self.input_window_size + 1))
            central_point_x_binary_out = "{0:010b}".format(central_point_x_out)
            central_point_x_binary_out = torch.tensor(np.array([int(d) for d in central_point_x_binary_out]))
            central_point_y_binary_out = "{0:010b}".format(central_point_y_out)
            central_point_y_binary_out = torch.tensor(np.array([int(d) for d in central_point_y_binary_out]))
            slice_x_out = slice(window_x_out[0], window_x_out[-1] + 1)
            slice_y_out = slice(window_y_out[0], window_y_out[-1] + 1)

            # Note : Input data
            fuel_subslice_in = self.fft_data(self.fuel_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                 self.fft_noise(noise_variance_in,self.fuel_slices[idx_input, slice_x_in, slice_y_in]).to(
                    self.device),
                nan=0.0))
            r_subslice_in = self.fft_data(self.r_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                 self.fft_noise(noise_variance_in,self.r_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0))
            g_subslice_in = self.fft_data(self.g_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                self.fft_noise(noise_variance_in,self.g_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0))
            b_subslice_in = self.fft_data(self.b_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                self.fft_noise(noise_variance_in,self.b_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0))
            alpha_subslice_in = self.fft_data(self.alpha_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                self.fft_noise(noise_variance_in,self.alpha_slices[idx_input, slice_x_in, slice_y_in]).to(
                    self.device),
                nan=0.0))
            data_input_subslice = torch.cat([r_subslice_in, g_subslice_in, b_subslice_in, alpha_subslice_in], dim=0)

            meta_step_in = self.meta_binary_slices[idx_input][0]
            meta_step_in_numeric = self.meta_tensor[idx_input][0]

            meta_fuel_initial_speed_in = self.meta_binary_slices[idx_input][1]
            meta_fuel_cut_off_time_in = self.meta_binary_slices[idx_input][2]
            meta_igni_time_in = self.meta_binary_slices[idx_input][3]
            meta_ignition_temp_in = self.meta_binary_slices[idx_input][4]
            meta_viscosity_in = self.meta_binary_slices[idx_input][14]
            meta_diff_in = self.meta_binary_slices[idx_input][15]
            meta_input_subslice = torch.cat([meta_step_in, meta_fuel_initial_speed_in,
                                             meta_fuel_cut_off_time_in, meta_igni_time_in,
                                             meta_ignition_temp_in, meta_viscosity_in, meta_diff_in], dim=0)

            # Note : Output data
            fuel_subslice_out = self.fuel_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                self.fft_noise(noise_variance_out,self.fuel_slices[idx_output, slice_x_out, slice_y_out]).to(
                    self.device),nan=0.0)
            r_subslice_out = self.r_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                self.fft_noise(noise_variance_out,self.r_slices[idx_output, slice_x_out, slice_y_out]).to(
                    self.device),nan=0.0)
            g_subslice_out = self.g_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                self.fft_noise(noise_variance_out,self.g_slices[idx_output, slice_x_out, slice_y_out]).to(
                    self.device),nan=0.0)
            b_subslice_out = self.b_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                self.fft_noise(noise_variance_out,self.b_slices[idx_output, slice_x_out, slice_y_out]), nan=0.0)
            alpha_subslice_out = self.alpha_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                self.fft_noise(noise_variance_out,self.alpha_slices[idx_output, slice_x_out, slice_y_out]).to(self.device), nan=0.0)
            data_output_subslice = torch.cat([r_subslice_out, g_subslice_out, b_subslice_out, alpha_subslice_out],
                                             dim=0)

            meta_step_out = self.meta_binary_slices[idx_output][0]
            meta_step_out_numeric = self.meta_tensor[idx_output][0]
            meta_fuel_initial_speed_out = self.meta_binary_slices[idx_output][1]
            meta_fuel_cut_off_time_out = self.meta_binary_slices[idx_output][2]
            meta_igni_time_out = self.meta_binary_slices[idx_output][3]
            meta_ignition_temp_out = self.meta_binary_slices[idx_output][4]
            meta_viscosity_out = self.meta_binary_slices[idx_output][14]
            meta_diff_out = self.meta_binary_slices[idx_output][15]
            meta_output_subslice = torch.cat([meta_step_out, meta_fuel_initial_speed_out,
                                              meta_fuel_cut_off_time_out, meta_igni_time_out,
                                              meta_ignition_temp_out, meta_viscosity_out, meta_diff_out], dim=0)

            r_in_is_zero = torch.count_nonzero(r_subslice_in)
            r_out_is_zero = torch.count_nonzero(r_subslice_out)
            g_in_is_zero = torch.count_nonzero(g_subslice_in)
            g_out_is_zero = torch.count_nonzero(g_subslice_out)
            b_in_is_zero = torch.count_nonzero(b_subslice_in)
            b_out_is_zero = torch.count_nonzero(b_subslice_out)
            a_in_is_zero = torch.count_nonzero(alpha_subslice_out)
            a_out_is_zero = torch.count_nonzero(alpha_subslice_out)
            f_in_is_zero = torch.count_nonzero(fuel_subslice_out)
            f_out_is_zero = torch.count_nonzero(fuel_subslice_out)
            r_zero = r_in_is_zero == r_out_is_zero
            r_i0 = r_in_is_zero == 0
            r_o0 = r_out_is_zero == 0
            g_zero = g_in_is_zero == g_out_is_zero
            g_i0 = g_in_is_zero == 0
            g_o0 = g_out_is_zero == 0
            b_zero = b_in_is_zero == b_out_is_zero
            b_i0 = b_in_is_zero == 0
            b_o0 = b_out_is_zero == 0
            a_zero = a_in_is_zero == a_out_is_zero
            a_i0 = a_in_is_zero == 0
            a_o0 = a_out_is_zero == 0
            f_zero = f_in_is_zero == f_out_is_zero
            f_i0 = f_in_is_zero == 0
            f_o0 = f_out_is_zero == 0
            rzero = r_zero and r_i0 and r_o0
            gzero = g_zero and g_i0 and g_o0
            bzero = b_zero and b_i0 and b_o0
            azero = a_zero and a_i0 and a_o0
            fzero = f_zero and f_i0 and f_o0

            central_points_in = torch.cat([central_point_x_binary_in, central_point_y_binary_in], dim=0).to(self.device)
            central_points_out = torch.cat([central_point_x_binary_out, central_point_y_binary_out], dim=0).to(
                self.device)

            if create_val_dataset == 0:
                matches_points_in = (self.meta_input_h3_val == central_points_in).all(dim=1)
                matches_points_out = (self.meta_output_h3_val == central_points_out).all(dim=1)
                matches_time_in = (self.meta_input_h2_val == meta_step_in.to(self.device)).all(dim=1)
                matches_time_out = (self.meta_output_h2_val == meta_step_out.to(self.device)).all(dim=1)
                if True in matches_points_in and True in matches_time_in and True in matches_time_out and True in matches_points_out:
                    choose_diffrent_frame = 1

            mod = 4

            if self.epoch > self.num_of_epochs * 0.25 or create_val_dataset == 1:
                pass
            else:
                data_in_cnz = torch.count_nonzero(data_input_subslice)
                fuel_in_cnz = torch.count_nonzero(fuel_subslice_in)
                data_out_cnz = torch.count_nonzero(data_output_subslice)
                fuel_out_cnz = torch.count_nonzero(fuel_subslice_out)

                if (data_in_cnz < mod * int(
                        data_input_subslice.shape[0] * data_input_subslice.shape[1] / (self.epoch + mod)) or
                        fuel_in_cnz < mod * int(
                            data_input_subslice.shape[0] * data_input_subslice.shape[1] / (self.epoch + mod)) or
                        data_out_cnz < mod * int(
                            data_output_subslice.shape[0] * data_output_subslice.shape[1] / (self.epoch + mod)) or
                        fuel_out_cnz < mod * int(
                            data_output_subslice.shape[0] * data_output_subslice.shape[1] / (self.epoch + mod))):
                    choose_diffrent_frame = 1
            frame += 1
            if rzero and gzero and bzero and azero and fzero and idx_input > idx_output and choose_diffrent_frame:
                frame -= 1
            else:
                # Note: Data for the different layers
                data_input.append(data_input_subslice)
                structure_input.append(fuel_subslice_in)
                meta_input_h1.append(meta_input_subslice)
                meta_input_h2.append(meta_step_in)
                meta_input_h3.append(central_points_in)
                meta_input_h4.append(torch.cat([torch.tensor(window_x_in), torch.tensor(window_y_in)]))
                meta_input_h5.append(meta_step_in_numeric)
                noise_var_in.append(noise_variance_in_binary.to(torch.float))
                fmot_in.append(fmot_coef)
                fmot_in_binary.append(fmot_coef_binary.to(torch.float))
                data_output.append(data_output_subslice)
                structure_output.append(fuel_subslice_out)
                meta_output_h1.append(meta_output_subslice)
                meta_output_h2.append(meta_step_out)
                meta_output_h3.append(central_points_out)
                meta_output_h4.append(torch.cat([torch.tensor(window_x_out), torch.tensor(window_y_out)]))
                meta_output_h5.append(meta_step_out_numeric)
                noise_var_out.append(noise_variance_out_binary.to(torch.float))

        if create_val_dataset == 1:
            self.data_input_val = torch.stack(data_input, dim=0)[0:self.batch_size].to(self.device)
            self.structure_input_val = torch.stack(structure_input, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h1_val = torch.stack(meta_input_h1, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h2_val = torch.stack(meta_input_h2, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h3_val = torch.stack(meta_input_h3, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h4_val = torch.stack(meta_input_h4, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h5_val = torch.stack(meta_input_h5, dim=0)[0:self.batch_size].to(self.device)
            self.noise_var_in_val = torch.stack(noise_var_in, dim=0)[0:self.batch_size].to(self.device)
            self.fmot_in_val = torch.stack(fmot_in, dim=0)[0:self.batch_size].to(self.device)
            self.fmot_in_binary_val = torch.stack(fmot_in_binary, dim=0)[0:self.batch_size].to(self.device)
            self.data_output_val = torch.stack(data_output, dim=0)[0:self.batch_size].to(self.device)
            self.structure_output_val = torch.stack(structure_output, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h1_val = torch.stack(meta_output_h1, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h2_val = torch.stack(meta_output_h2, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h3_val = torch.stack(meta_output_h3, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h4_val = torch.stack(meta_output_h4, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h5_val = torch.stack(meta_output_h5, dim=0)[0:self.batch_size].to(self.device)
            self.noise_var_out_val = torch.stack(noise_var_out, dim=0)[0:self.batch_size].to(self.device)
        else:
            self.data_input = torch.stack(data_input, dim=0).to(self.device)
            self.structure_input = torch.stack(structure_input, dim=0).to(self.device)
            self.meta_input_h1 = torch.stack(meta_input_h1, dim=0).to(self.device)
            self.meta_input_h2 = torch.stack(meta_input_h2, dim=0).to(self.device)
            self.meta_input_h3 = torch.stack(meta_input_h3, dim=0).to(self.device)
            self.meta_input_h4 = torch.stack(meta_input_h4, dim=0).to(self.device)
            self.meta_input_h5 = torch.stack(meta_input_h5, dim=0).to(self.device)
            self.noise_diff_in = torch.stack(noise_var_in, dim=0).to(self.device)
            self.fmot_in = torch.stack(fmot_in, dim=0).to(self.device)
            self.fmot_in_binary = torch.stack(fmot_in_binary, dim=0).to(self.device)

            self.data_output = torch.stack(data_output, dim=0).to(self.device)
            self.structure_output = torch.stack(structure_output, dim=0).to(self.device)
            self.meta_output_h1 = torch.stack(meta_output_h1, dim=0).to(self.device)
            self.meta_output_h2 = torch.stack(meta_output_h2, dim=0).to(self.device)
            self.meta_output_h3 = torch.stack(meta_output_h3, dim=0).to(self.device)
            self.meta_output_h4 = torch.stack(meta_output_h4, dim=0).to(self.device)
            self.meta_output_h5 = torch.stack(meta_output_h5, dim=0).to(self.device)
            self.noise_diff_out = torch.stack(noise_var_out, dim=0).to(self.device)

    def examine(self, criterion, device, plot=0):
        matplotlib.use(self.mpl_backend)
        del self.model.NCA.axs2d
        del self.model.NCA.fig2d
        plt.ioff()
        plt.close('all')
        self.model.load_state_dict(torch.load('model.pt'))
        # self.model.NCA.fermion_features = None
        # self.model.NCA.boson_features = None
        no_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Final model no params:", no_params)
        spiking_probabilities = torch.zeros((self.model.nca_steps,)).to(self.device)
        # spiking_probabilities = torch.rand(self.model.nca_steps).to(self.device)
        folder_names = ['v', 'u', 'velocity_magnitude', 'fuel_density', 'oxidizer_density',
                        'product_density', 'pressure', 'temperature', 'rgb', 'alpha']
        data_tensor = []
        meta_tensor = []
        meta_binary = []
        field_names = []

        for name in folder_names:
            if os.path.exists(name):
                for i in range(self.first_frame, self.last_frame, self.frame_skip):
                    if name == 'rgb':
                        ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                        for j in range(0, 3):
                            data_tensor.append(ptfile['data'][:, :, j] / 255.)
                            meta_tensor.append(ptfile['metadata'])
                            field_names.append(ptfile['name'])
                    else:
                        ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                        data_tensor.append(ptfile['data'])
                        meta_tensor.append(ptfile['metadata'])
                        field_names.append(ptfile['name'])

        self.data_tensor = torch.stack(data_tensor, dim=0)
        self.meta_tensor = torch.stack(meta_tensor, dim=0)

        for i in range(self.meta_tensor.shape[0]):
            meta_temp = []
            for j in range(self.meta_tensor.shape[1]):
                binary_var = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', self.meta_tensor[i, j]))
                # Note : '!f' The '!' ensures that
                #     it's in network byte order (big-endian) and the 'f' says that it should be
                #     packed as a float. Use d for double precision
                binary_var = np.frombuffer(binary_var.encode("ascii"), dtype='u1') - 48
                # binary_var = torch.tensor([int(bit) for bit in binary_var], dtype=torch.uint8) - 48
                meta_temp.append(binary_var)
            meta_binary.append(meta_temp)
        self.meta_binary = torch.from_numpy(np.array(meta_binary))
        self.field_names = field_names
        fdens_idx = np.array([i for i, x in enumerate(self.field_names) if x == "fuel_density"])
        #frame_samples = random.sample(list(set(fdens_idx)), k=self.no_frame_samples)
        f_dens_pos = len(fdens_idx)

        #fdens_idx = fdens_idx[frame_samples]
        rgb_idx = np.array([i for i, x in enumerate(self.field_names) if x == "rgb"])
        r_idx = rgb_idx[::3]  #[frame_samples]
        g_idx = rgb_idx[::3] + 1  #[frame_samples]
        b_idx = rgb_idx[::3] + 2  #[frame_samples]
        alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])  #[frame_samples]
        fuel_slices = self.data_tensor[fdens_idx]
        self.w = fuel_slices.shape[1]
        self.h = fuel_slices.shape[2]
        self.n_frames = fuel_slices.shape[0]
        min_val = fuel_slices.min()
        max_val = fuel_slices.max()
        self.fuel_slices = (fuel_slices - min_val) / ((max_val - min_val) + 1e-12)
        self.r_slices = self.data_tensor[r_idx]
        self.g_slices = self.data_tensor[g_idx]
        self.b_slices = self.data_tensor[b_idx]
        self.alpha_slices = self.data_tensor[alpha_idx]
        self.meta_binary_slices = self.meta_binary[fdens_idx]


        # Note: IDX preparation
        central_points_x = np.arange(self.input_window_size, self.w - self.input_window_size + 1)
        central_points_y = np.arange(self.input_window_size, self.h - self.input_window_size + 1)

        central_points_x_pos = central_points_x + self.input_window_size
        central_points_x_neg = central_points_x - self.input_window_size
        central_points_y_pos = central_points_y + self.input_window_size
        central_points_y_neg = central_points_y - self.input_window_size

        windows_x = []
        windows_y = []

        central_points_x_binary = []
        central_points_y_binary = []
        v = int(central_points_x_pos.shape[0] / self.model.in_scale + 1)
        h = int(central_points_y_pos.shape[0] / self.model.in_scale + 1)
        j = 0
        for m in range(0, v):
            k = 0
            for n in range(0, h):
                wx_range = np.array(range(int(central_points_x_neg[j]), int(central_points_x_pos[j]) + 2))
                windows_x.append(wx_range)
                central_point_x_binary_pre = "{0:010b}".format(central_points_x[j])
                central_points_x_binary.append(
                    torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_x_binary_pre]))
                wy_range = np.array(range(int(central_points_y_neg[k]), int(central_points_y_pos[k]) + 2))
                windows_y.append(wy_range)
                central_point_y_binary_pre = "{0:010b}".format(central_points_y[k])
                central_points_y_binary.append(
                    torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_y_binary_pre]))
                k += self.model.in_scale
            j += self.model.in_scale

        central_points_x_binary = torch.tensor(np.array(central_points_x_binary))
        central_points_y_binary = torch.tensor(np.array(central_points_y_binary))
        central_points_xy_binary = []
        for g in range(len(central_points_x_binary)):
            xy_binary = torch.cat([central_points_x_binary[g], central_points_y_binary[g]])
            central_points_xy_binary.append(xy_binary)

        x_idx = torch.tensor(np.array(windows_x))
        y_idx = torch.tensor(np.array(windows_y))
        x_idx_start = np.array([sublist[0] for sublist in x_idx])
        x_idx_end = np.array([sublist[-1] for sublist in x_idx])
        y_idx_start = np.array([sublist[0] for sublist in y_idx])
        y_idx_end = np.array([sublist[-1] for sublist in y_idx])
        t = 0.
        ims = []
        fig = plt.figure(figsize=(10, 6))
        grid = (1, 3)
        ax1 = plt.subplot2grid(grid, (0, 0))
        ax2 = plt.subplot2grid(grid, (0, 1))
        ax3 = plt.subplot2grid(grid, (0, 2))
        #ax4 = plt.subplot2grid(grid, (1, 0), colspan=3)
        # ax3 = plt.subplot2grid(grid, (1, 0))
        # ax4 = plt.subplot2grid(grid, (1, 1))

        for ax in [ax1, ax2, ax3]:#, ax4]:
            ax.set_axis_off()
        weights_anim = torch.zeros(0)
        for param in self.model.parameters():
            param = torch.flatten(param, start_dim=0)
            weights_anim = torch.cat([weights_anim, param.cpu()])

        x, y = 1500, 4700
        target_len = x * y
        if target_len > weights_anim.shape[0]:
            n = target_len - weights_anim.shape[0]
            wfilling = torch.full((n,), 0.)
            weights_anim = torch.cat([weights_anim, wfilling])
        w_stat = weights_anim.view(x, y).detach().cpu().numpy()
        for i in range(0, self.n_frames - 1):
            idx_input = i
            idx_output = i + 1

            # Note : Input data
            fsin = []
            rsin = []
            gsin = []
            bsin = []
            asin = []

            fsout = []
            rsout = []
            gsout = []
            bsout = []
            asout = []

            for ii in range(len(x_idx_start)):
                fsin.append(self.fuel_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                rsin.append(self.r_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                gsin.append(self.g_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                bsin.append(self.b_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                asin.append(self.alpha_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])

                fsout.append(self.fuel_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                rsout.append(self.r_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                gsout.append(self.g_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                bsout.append(self.b_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                asout.append(
                    self.alpha_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])

            fuel_subslice_in = torch.stack(fsin, dim=0)
            r_subslice_in = torch.stack(rsin, dim=0)
            g_subslice_in = torch.stack(gsin, dim=0)
            b_subslice_in = torch.stack(bsin, dim=0)
            alpha_subslice_in = torch.stack(asin, dim=0)
            data_input_subslice = torch.cat([r_subslice_in, g_subslice_in, b_subslice_in, alpha_subslice_in], dim=1)
            meta_step_in = self.meta_binary_slices[idx_input][0]
            meta_step_in_numeric = self.meta_tensor[idx_input][0]
            meta_fuel_initial_speed_in = self.meta_binary_slices[idx_input][1]
            meta_fuel_cut_off_time_in = self.meta_binary_slices[idx_input][2]
            meta_igni_time_in = self.meta_binary_slices[idx_input][3]
            meta_ignition_temp_in = self.meta_binary_slices[idx_input][4]

            meta_viscosity_in = self.meta_binary_slices[idx_input][14]
            meta_diff_in = self.meta_binary_slices[idx_input][15]
            meta_input_subslice = torch.cat([meta_step_in, meta_fuel_initial_speed_in,
                                             meta_fuel_cut_off_time_in, meta_igni_time_in,
                                             meta_ignition_temp_in, meta_viscosity_in, meta_diff_in], dim=0)
            # Note : Output data
            f_subslice_out = torch.stack(fsout, dim=0)
            r_subslice_out = torch.stack(rsout, dim=0)
            g_subslice_out = torch.stack(gsout, dim=0)
            b_subslice_out = torch.stack(bsout, dim=0)
            alpha_subslice_out = torch.stack(asout, dim=0)

            data_output_subslice = torch.cat([r_subslice_out, g_subslice_out, b_subslice_out, alpha_subslice_out],
                                             dim=1)
            meta_step_out = self.meta_binary_slices[idx_output][0]
            meta_step_out_numeric = self.meta_tensor[idx_output][0]
            meta_fuel_initial_speed_out = self.meta_binary_slices[idx_output][1]
            meta_fuel_cut_off_time_out = self.meta_binary_slices[idx_output][2]
            meta_igni_time_out = self.meta_binary_slices[idx_output][3]
            meta_ignition_temp_out = self.meta_binary_slices[idx_output][4]
            meta_viscosity_out = self.meta_binary_slices[idx_output][14]
            meta_diff_out = self.meta_binary_slices[idx_output][15]
            meta_output_subslice = torch.cat([meta_step_out, meta_fuel_initial_speed_out,
                                              meta_fuel_cut_off_time_out, meta_igni_time_out,
                                              meta_ignition_temp_out, meta_viscosity_out, meta_diff_out], dim=0)
            # Note: Data for the different layers
            data_input = data_input_subslice
            self.model.batch_size = data_input.shape[0]
            structure_input = fuel_subslice_in
            meta_input_h1 = meta_input_subslice.unsqueeze(0).repeat(data_input.shape[0], 1)
            meta_input_h2 = meta_step_in.unsqueeze(0).repeat(data_input.shape[0], 1)
            meta_input_h3 = torch.tensor(np.array(central_points_xy_binary))
            meta_input_h4 = torch.cat([x_idx[:, 0:-1], y_idx[:, 0:-1]], dim=1)
            noise_var_in = torch.zeros((data_input.shape[0], 32))
            fmot_in_binary = torch.zeros((data_input.shape[0], 32))
            meta_input_h5 = meta_step_in_numeric.repeat(data_input.shape[0], 1).squeeze(1)
            data_output = data_output_subslice
            meta_output_h1 = meta_output_subslice.unsqueeze(0).repeat(data_input.shape[0], 1)
            meta_output_h2 = meta_step_out.unsqueeze(0).repeat(data_input.shape[0], 1)
            meta_output_h3 = torch.tensor(np.array(central_points_xy_binary))
            meta_output_h4 = torch.cat([x_idx[:, 0:-1], y_idx[:, 0:-1]], dim=1)
            meta_output_h5 = meta_step_out_numeric.repeat(data_input.shape[0], 1).squeeze(1)
            noise_var_out = torch.zeros((data_input.shape[0], 32))



            (data_input, structure_input, meta_input_h1, meta_input_h2,
             meta_input_h3, meta_input_h4, meta_input_h5, noise_var_in, fmot_in_binary, meta_output_h1,
             meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out) = \
                (data_input.to(device),
                 structure_input.to(device),
                 meta_input_h1.to(device),
                 meta_input_h2.to(device),
                 meta_input_h3.to(device),
                 meta_input_h4.to(device),
                 meta_input_h5.to(device),
                 noise_var_in.to(device),
                 fmot_in_binary.to(device),
                 meta_output_h1.to(device),
                 meta_output_h2.to(device),
                 meta_output_h3.to(device),
                 meta_output_h4.to(device),
                 meta_output_h5.to(device),
                 noise_var_out.to(device))

            # data_output = data_output.to(device)
            dataset = (data_input, structure_input, meta_input_h1, meta_input_h2,
                       meta_input_h3, meta_input_h4, meta_input_h5, noise_var_in, fmot_in_binary, meta_output_h1,
                       meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out)
            self.model.eval()
            with torch.no_grad():
                t_start = time.perf_counter()
                pred_r, pred_g, pred_b, pred_a, pred_s, _, _, _,_,_,_,_ = self.model(dataset, spiking_probabilities)
                # print(pred_r)
                # time.sleep(1000)
                t_pred = time.perf_counter()
            t = t_pred - t_start
            gpu_stats = gpustat.GPUStatCollection.new_query()
            mem_usage = [gpu.memory_used for gpu in gpu_stats]
            print(f'Pred Time: {t * 1e3:.2f} [ms] ',f'GPU MEM: {round(mem_usage[0] * 1e-3, 3)} [GB]',
                  f'CPU TEMP: {WinTmp.CPU_Temp()} [°C], ', f'GPU TEMP: {WinTmp.GPU_Temp()} [°C], ')

            r_v_true = np.array([]).reshape(0, h * self.model.in_scale)
            g_v_true = np.array([]).reshape(0, h * self.model.in_scale)
            b_v_true = np.array([]).reshape(0, h * self.model.in_scale)
            a_v_true = np.array([]).reshape(0, h * self.model.in_scale)

            r_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            g_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            b_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            a_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            idx = 0
            for m in range(0, v):
                r_h_true = np.array([]).reshape(self.model.in_scale, 0)
                g_h_true = np.array([]).reshape(self.model.in_scale, 0)
                b_h_true = np.array([]).reshape(self.model.in_scale, 0)
                a_h_true = np.array([]).reshape(self.model.in_scale, 0)

                r_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                g_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                b_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                a_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                for n in range(0, h):
                    iter_index = idx  #(n + m * h)
                    r_h_true = np.hstack([r_h_true, r_subslice_out[iter_index].cpu().detach().numpy()])
                    g_h_true = np.hstack([g_h_true, g_subslice_out[iter_index].cpu().detach().numpy()])
                    b_h_true = np.hstack([b_h_true, b_subslice_out[iter_index].cpu().detach().numpy()])
                    a_h_true = np.hstack([a_h_true, alpha_subslice_out[iter_index].cpu().detach().numpy()])

                    r_h_pred = np.hstack([r_h_pred, pred_r[iter_index].cpu().detach().numpy()])
                    g_h_pred = np.hstack([g_h_pred, pred_g[iter_index].cpu().detach().numpy()])
                    b_h_pred = np.hstack([b_h_pred, pred_b[iter_index].cpu().detach().numpy()])
                    a_h_pred = np.hstack([a_h_pred, pred_a[iter_index].cpu().detach().numpy()])
                    idx += 1
                r_v_true = np.vstack([r_v_true, r_h_true])
                g_v_true = np.vstack([g_v_true, g_h_true])
                b_v_true = np.vstack([b_v_true, b_h_true])
                a_v_true = np.vstack([a_v_true, a_h_true])

                r_v_pred = np.clip(abs(np.vstack([r_v_pred, r_h_pred])), 0., 1.)
                g_v_pred = np.clip(abs(np.vstack([g_v_pred, g_h_pred])), 0., 1.)
                b_v_pred = np.clip(abs(np.vstack([b_v_pred, b_h_pred])), 0., 1.)
                a_v_pred = np.clip(abs(np.vstack([a_v_pred, a_h_pred])), 0., 1.)

            prediction = np.stack((r_v_pred, g_v_pred, b_v_pred), axis=2)
            ground_truth = np.stack((r_v_true, g_v_true, b_v_true), axis=2)
            title_pred = ax1.set_title("Prediction")
            title_true = ax2.set_title("Ground Truth")
            title_rms = ax3.set_title("rms")

            rgb_pred_anim = ax1.imshow((prediction*255).astype(np.uint8), alpha=a_v_pred)
            rgb_true_anim = ax2.imshow((ground_truth*255).astype(np.uint8) , alpha=a_v_true)

            rms = np.sqrt(np.mean((prediction  - ground_truth)** 2 , axis=2))
            rms = rms / rms.max()
            rms_anim = ax3.imshow(rms, cmap='RdBu', vmin=0, vmax=1)
            #log_w_stat = np.log10(w_stat + 1e-10) # Note : for positive val only
            #log_w_stat = np.log10(np.abs(w_stat) + 1e-10)*np.sign(w_stat)
            #w_static = ax4.imshow(w_stat, cmap='seismic')
            ims.append([rgb_pred_anim, rgb_true_anim, rms_anim, title_pred, title_true, title_rms])
        fig.colorbar(rms_anim, ax=ax3)
        #fig.colorbar(w_static, ax=ax4)
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
        ani.save("flame_animation.gif", writer='imagemagick', fps=24,dpi=200)
        plt.show()

    def visualize_lerning(self, poly_degree=3):
        plt.close('all')
        plt.plot(self.train_loss, color='blue',label='train')
        plt.plot(self.val_loss, color='orange',label='test')
        plt.plot(np.array(torch.tensor(self.disc_loss).cpu()), color='red', label='disc')
        plt.plot(self.vsi_loss,label='vsi_loss',linestyle=':')
        # plt.plot(self.entropy_loss,label='entropy_loss',marker='p')
        plt.plot(self.grad_penalty,label='grad_penalty')
        #plt.plot(self.kl_loss,label='kl_loss',marker='h')
        plt.plot(self.sink_loss,label='sink_loss',marker='^')
        plt.plot(self.critical_loss,label='critical_loss',linestyle='--')
        plt.plot(self.hf_e_loss,label='hf_e_loss',marker='+')
        plt.plot(self.b_loss,label='b_loss',marker='o')
        plt.plot(self.value_loss,label='value_loss',marker='s')
        plt.plot(self.diff_fft_loss,label='diff_fft_loss',linestyle='-.')
        plt.plot(self.fft_loss,label='fft_loss',marker='v')
        #plt.plot(self.diff_loss,label='diff_loss')
        plt.plot(self.log_det_jacobian_loss,label='log_det_jacobian_loss')
        #plt.plot(self.freq_loss,label='freq_loss')
        # plt.plot(self.val_vsi_loss,label='val_vsi_loss',linestyle='--')
        # plt.plot(self.val_entropy_loss,label='val_entropy_loss',linestyle='--')
        # plt.plot(self.val_grad_penalty,label='val_grad_penalty',linestyle='--')
        # plt.plot(self.val_kl_loss,label='val_kl_loss',linestyle='--')
        # plt.plot(self.val_sink_loss,label='val_sink_loss',linestyle='--')
        # plt.plot(self.val_critical_loss,label='val_critical_loss',linestyle='--')
        # plt.plot(self.val_hf_e_loss,label='val_hf_e_loss',linestyle='--')
        # plt.plot(self.val_b_loss,label='val_b_loss',linestyle='--')
        # plt.plot(self.val_value_loss,label='val_value_loss',linestyle='--')
        # plt.plot(self.val_diff_fft_loss,label='val_diff_fft_loss',linestyle='--')
        # plt.plot(self.val_fft_loss,label='val_fft_loss',linestyle='--')
        # plt.plot(self.val_diff_loss,label='val_diff_loss',linestyle='--')
        # plt.plot(self.val_log_det_jacobian_loss, label='val_log_det_jacobian_loss',linestyle='--')
        # plt.plot(self.val_freq_loss, label='val_freq_loss',linestyle='--')
        plt.plot(self.cpu_temp,label='cpu_temp')
        plt.plot(self.gpu_temp,label='gpu_temp')

        # avg_train_loss = sum(self.train_loss) / len(self.train_loss)
        # avg_val_loss = sum(self.val_loss) / len(self.val_loss)
        # epochs = np.arange(len(self.train_loss))
        # train_poly_fit = np.poly1d(np.polyfit(epochs, self.train_loss, poly_degree))
        # val_poly_fit = np.poly1d(np.polyfit(epochs, self.val_loss, poly_degree))
        # plt.plot(epochs, train_poly_fit(epochs), color='blue', linestyle='--',
        #          label=f'Train: deg: {poly_degree} | Avg: {avg_train_loss:.3f}')
        # plt.plot(epochs, val_poly_fit(epochs), color='orange', linestyle='--',
        #          label=f'Val: deg: {poly_degree} | Avg: {avg_val_loss:.3f}')
        plt.yscale("log")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.06, 1.0),loc='upper center')
        plt.grid(True)
        plt.show()


    def reconstruction_loss(self, criterion, device, no_patches):
        spiking_probabilities = torch.zeros((self.model.nca_steps,)).to(self.device)
        # spiking_probabilities = torch.rand(self.model.nca_steps).to(self.device)
        if self.data_tensor is None:
            folder_names = ['v', 'u', 'velocity_magnitude', 'fuel_density', 'oxidizer_density',
                            'product_density', 'pressure', 'temperature', 'rgb', 'alpha']
            data_tensor = []
            meta_tensor = []
            meta_binary = []
            field_names = []

            for name in folder_names:
                if os.path.exists(name):
                    for i in range(self.first_frame, self.last_frame, self.frame_skip):
                        if name == 'rgb':
                            ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                            for j in range(0, 3):
                                data_tensor.append(ptfile['data'][:, :, j] / 255.)
                                meta_tensor.append(ptfile['metadata'])
                                field_names.append(ptfile['name'])
                        else:
                            ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                            data_tensor.append(ptfile['data'])
                            meta_tensor.append(ptfile['metadata'])
                            field_names.append(ptfile['name'])

            self.data_tensor = torch.stack(data_tensor, dim=0)
            self.meta_tensor = torch.stack(meta_tensor, dim=0)

            for i in range(self.meta_tensor.shape[0]):
                meta_temp = []
                for j in range(self.meta_tensor.shape[1]):
                    binary_var = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', self.meta_tensor[i, j]))
                    # Note : '!f' The '!' ensures that
                    #     it's in network byte order (big-endian) and the 'f' says that it should be
                    #     packed as a float. Use d for double precision
                    binary_var = np.frombuffer(binary_var.encode("ascii"), dtype='u1') - 48
                    # binary_var = torch.tensor([int(bit) for bit in binary_var], dtype=torch.uint8) - 48
                    meta_temp.append(binary_var)
                meta_binary.append(meta_temp)
            self.meta_binary = torch.from_numpy(np.array(meta_binary))
            self.field_names = field_names
            fdens_idx = np.array([i for i, x in enumerate(self.field_names) if x == "fuel_density"])
            # frame_samples = random.sample(list(set(fdens_idx)), k=self.no_frame_samples)
            f_dens_pos = len(fdens_idx)

            # fdens_idx = fdens_idx[frame_samples]
            rgb_idx = np.array([i for i, x in enumerate(self.field_names) if x == "rgb"])
            r_idx = rgb_idx[::3]  # [frame_samples]
            g_idx = rgb_idx[::3] + 1  # [frame_samples]
            b_idx = rgb_idx[::3] + 2  # [frame_samples]
            alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])  # [frame_samples]
            fuel_slices = self.data_tensor[fdens_idx]
            self.w = fuel_slices.shape[1]
            self.h = fuel_slices.shape[2]
            self.n_frames = fuel_slices.shape[0]
            min_val = fuel_slices.min()
            max_val = fuel_slices.max()
            self.fuel_slices = (fuel_slices - min_val) / ((max_val - min_val) + 1e-12)
            self.r_slices = self.data_tensor[r_idx]
            self.g_slices = self.data_tensor[g_idx]
            self.b_slices = self.data_tensor[b_idx]
            self.alpha_slices = self.data_tensor[alpha_idx]
            self.meta_binary_slices = self.meta_binary[fdens_idx]
        else:
            pass

        # Note: IDX preparation
        number_of_patches = no_patches
        central_points_x = np.random.randint(self.input_window_size, self.w - self.input_window_size + 1,
                                             size=number_of_patches)
        central_points_y = np.random.randint(self.input_window_size, self.h - self.input_window_size + 1,
                                             size=number_of_patches)

        central_points_x_pos = central_points_x + self.input_window_size
        central_points_x_neg = central_points_x - self.input_window_size
        central_points_y_pos = central_points_y + self.input_window_size
        central_points_y_neg = central_points_y - self.input_window_size

        windows_x = []
        windows_y = []

        central_points_x_binary = []
        central_points_y_binary = []
        v = int(central_points_x_pos.shape[0])
        h = int(central_points_y_pos.shape[0])
        for m in range(0, v):
            for n in range(0, h):
                wx_range = np.array(range(int(central_points_x_neg[m]), int(central_points_x_pos[m]) + 2))
                windows_x.append(wx_range)
                central_point_x_binary_pre = "{0:010b}".format(central_points_x[m])
                central_points_x_binary.append(
                    torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_x_binary_pre]))
                wy_range = np.array(range(int(central_points_y_neg[n]), int(central_points_y_pos[n]) + 2))
                windows_y.append(wy_range)
                central_point_y_binary_pre = "{0:010b}".format(central_points_y[n])
                central_points_y_binary.append(
                    torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_y_binary_pre]))

        central_points_x_binary = torch.tensor(np.array(central_points_x_binary))
        central_points_y_binary = torch.tensor(np.array(central_points_y_binary))
        central_points_xy_binary = []
        for g in range(len(central_points_x_binary)):
            xy_binary = torch.cat([central_points_x_binary[g], central_points_y_binary[g]])
            central_points_xy_binary.append(xy_binary)

        x_idx = torch.tensor(np.array(windows_x))
        y_idx = torch.tensor(np.array(windows_y))
        x_idx_start = np.array([sublist[0] for sublist in x_idx])
        x_idx_end = np.array([sublist[-1] for sublist in x_idx])
        y_idx_start = np.array([sublist[0] for sublist in y_idx])
        y_idx_end = np.array([sublist[-1] for sublist in y_idx])
        #for i in range(0, fuel_slices.shape[0] - 1):
        idx_input = random.randint(0, self.n_frames - 2)
        idx_output = idx_input + 1

        # Note : Input data
        fsin = []
        rsin = []
        gsin = []
        bsin = []
        asin = []

        fsout = []
        rsout = []
        gsout = []
        bsout = []
        asout = []

        for ii in range(len(x_idx_start)):
            d_shape = self.fuel_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]].shape
            if d_shape[0] != 15 or d_shape[1] != 15:
                pass
            else:
                fsin.append(self.fuel_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                rsin.append(self.r_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                gsin.append(self.g_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                bsin.append(self.b_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                asin.append(self.alpha_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])

                fsout.append(self.fuel_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                rsout.append(self.r_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                gsout.append(self.g_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                bsout.append(self.b_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                asout.append(
                    self.alpha_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])

        # if create_val_dataset == 0:
        #     self.data_input_val
        #     self.data_output_val
        # TODO: Exlude validation dataset from training dataset

        fuel_subslice_in = torch.stack(fsin, dim=0)
        r_subslice_in = torch.stack(rsin, dim=0)
        g_subslice_in = torch.stack(gsin, dim=0)
        b_subslice_in = torch.stack(bsin, dim=0)
        alpha_subslice_in = torch.stack(asin, dim=0)
        data_input_subslice = torch.cat([r_subslice_in, g_subslice_in, b_subslice_in, alpha_subslice_in], dim=1)
        meta_step_in = self.meta_binary_slices[idx_input][0]
        meta_step_in_numeric = self.meta_tensor[idx_input][0]
        meta_fuel_initial_speed_in = self.meta_binary_slices[idx_input][1]
        meta_fuel_cut_off_time_in = self.meta_binary_slices[idx_input][2]
        meta_igni_time_in = self.meta_binary_slices[idx_input][3]
        meta_ignition_temp_in = self.meta_binary_slices[idx_input][4]

        meta_viscosity_in = self.meta_binary_slices[idx_input][14]
        meta_diff_in = self.meta_binary_slices[idx_input][15]
        meta_input_subslice = torch.cat([meta_step_in, meta_fuel_initial_speed_in,
                                         meta_fuel_cut_off_time_in, meta_igni_time_in,
                                         meta_ignition_temp_in, meta_viscosity_in, meta_diff_in], dim=0)
        # Note : Output data
        f_subslice_out = torch.stack(fsout, dim=0)
        r_subslice_out = torch.stack(rsout, dim=0)
        g_subslice_out = torch.stack(gsout, dim=0)
        b_subslice_out = torch.stack(bsout, dim=0)
        alpha_subslice_out = torch.stack(asout, dim=0)

        data_output_subslice = torch.cat([r_subslice_out, g_subslice_out, b_subslice_out, alpha_subslice_out],
                                         dim=1)
        meta_step_out = self.meta_binary_slices[idx_output][0]
        meta_step_out_numeric = self.meta_tensor[idx_output][0]
        meta_fuel_initial_speed_out = self.meta_binary_slices[idx_output][1]
        meta_fuel_cut_off_time_out = self.meta_binary_slices[idx_output][2]
        meta_igni_time_out = self.meta_binary_slices[idx_output][3]
        meta_ignition_temp_out = self.meta_binary_slices[idx_output][4]
        meta_viscosity_out = self.meta_binary_slices[idx_output][14]
        meta_diff_out = self.meta_binary_slices[idx_output][15]
        meta_output_subslice = torch.cat([meta_step_out, meta_fuel_initial_speed_out,
                                          meta_fuel_cut_off_time_out, meta_igni_time_out,
                                          meta_ignition_temp_out, meta_viscosity_out, meta_diff_out], dim=0)
        # Note: Data for the different layers
        data_input = data_input_subslice
        self.model.batch_size = data_input.shape[0]
        structure_input = fuel_subslice_in
        structure_output = f_subslice_out
        meta_input_h1 = meta_input_subslice.unsqueeze(0).repeat(data_input.shape[0], 1)
        meta_input_h2 = meta_step_in.unsqueeze(0).repeat(data_input.shape[0], 1)
        meta_input_h3 = torch.tensor(np.array(central_points_xy_binary))
        meta_input_h4 = torch.cat([x_idx[:, 0:-1], y_idx[:, 0:-1]], dim=1)
        noise_var_in = torch.zeros((data_input.shape[0], 32))
        fmot_in_binary = torch.zeros((data_input.shape[0], 32))
        meta_input_h5 = meta_step_in_numeric.repeat(data_input.shape[0], 1).squeeze(1)
        data_output = data_output_subslice
        meta_output_h1 = meta_output_subslice.unsqueeze(0).repeat(data_input.shape[0], 1)
        meta_output_h2 = meta_step_out.unsqueeze(0).repeat(data_input.shape[0], 1)
        meta_output_h3 = torch.tensor(np.array(central_points_xy_binary))
        meta_output_h4 = torch.cat([x_idx[:, 0:-1], y_idx[:, 0:-1]], dim=1)
        meta_output_h5 = meta_step_out_numeric.repeat(data_input.shape[0], 1).squeeze(1)
        noise_var_out = torch.zeros((data_input.shape[0], 32))

        (data_input, structure_input, meta_input_h1, meta_input_h2,
         meta_input_h3, meta_input_h4, meta_input_h5, noise_var_in, fmot_in_binary, meta_output_h1,
         meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out) = \
            (data_input.to(device),
             structure_input.to(device),
             meta_input_h1.to(device),
             meta_input_h2.to(device),
             meta_input_h3.to(device),
             meta_input_h4.to(device),
             meta_input_h5.to(device),
             noise_var_in.to(device),
             fmot_in_binary.to(device),
             meta_output_h1.to(device),
             meta_output_h2.to(device),
             meta_output_h3.to(device),
             meta_output_h4.to(device),
             meta_output_h5.to(device),
             noise_var_out.to(device))

        # data_output = data_output.to(device)
        dataset = (data_input, structure_input, meta_input_h1, meta_input_h2,
                   meta_input_h3, meta_input_h4, meta_input_h5, noise_var_in, fmot_in_binary, meta_output_h1,
                   meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out)
        pred_r, pred_g, pred_b, pred_a, pred_s, _, _, _,_,_ = self.model(dataset, spiking_probabilities)
        prediction = torch.cat([pred_r, pred_g, pred_b, pred_a, pred_s], dim=1)
        ground_truth = torch.cat([r_subslice_out, g_subslice_out, b_subslice_out, alpha_subslice_out, structure_output],
                                 dim=1)
        rms = criterion(prediction, ground_truth)
        return rms

    def learning_phase(self, teacher, no_frame_samples, batch_size, input_window_size, first_frame, last_frame,
                       frame_skip, criterion, optimizer,criterion_disc,disc_optimizer, device, learning=1,
                       num_epochs=1500):
        (self.no_frame_samples, self.batch_size, self.input_window_size, self.first_frame,
         self.last_frame, self.frame_skip) = (no_frame_samples, batch_size,
                                              input_window_size, first_frame, last_frame, frame_skip)

        criterion_model = criterion
        self.num_of_epochs = num_epochs
        self.model.last_frame = self.last_frame
        model_to_Save = self.model
        if learning == 1:
            best_loss = float('inf')
            num_epochs = num_epochs
            t = 0.
            t_epoch = 0.
            grad_counter = 0
            reiterate_data = 1
            reiterate_counter = 0
            discriminator_learning = True
            norm = 'forward'
            print_every_nth_frame = 10
            best_models = []
            best_losses = []
            # zero = torch.tensor([0.],requires_grad=True).to(device).float()
            self.data_preparation(1)
            self.data_preparation()
            val_idx = torch.arange(self.data_input_val.shape[0])
            self.validation_dataset = (
                self.data_input_val, self.structure_input_val, self.meta_input_h1_val, self.meta_input_h2_val,
                self.meta_input_h3_val, self.meta_input_h4_val, self.meta_input_h5_val, self.noise_var_in_val,
                self.fmot_in_binary_val,
                self.meta_output_h1_val, self.meta_output_h2_val, self.meta_output_h3_val, self.meta_output_h4_val,
                self.meta_output_h5_val, self.noise_var_out_val)
            for epoch in range(num_epochs):
                self.epoch = epoch
                t_epoch_start = time.perf_counter()
                self.seed_setter(int(epoch + 1))
                if reiterate_data == 0:
                    self.data_preparation()
                    print("new sets of data prepared!")
                else:
                    reiterate_counter += 1

                m_idx = torch.arange(int(self.data_input.shape[0] / 2))  # TODO : Change to random selection

                dataset = (self.data_input[m_idx], self.structure_input[m_idx], self.meta_input_h1[m_idx],
                           self.meta_input_h2[m_idx],
                           self.meta_input_h3[m_idx], self.meta_input_h4[m_idx], self.meta_input_h5[m_idx],
                           self.noise_diff_in[m_idx], self.fmot_in_binary[m_idx], self.meta_output_h1[m_idx],
                           self.meta_output_h2[m_idx], self.meta_output_h3[m_idx], self.meta_output_h4[m_idx],
                           self.meta_output_h5[m_idx], self.noise_diff_out[m_idx])

                spiking_probabilities = torch.rand(self.model.nca_steps).to(self.device)
                t_start = time.perf_counter()
                model_output = self.model(dataset, spiking_probabilities)
                t_pred = time.perf_counter()

                if epoch == 0:
                    self.disc_loss = [torch.tensor([0.5], device=self.device)]  # Ensure it's initialized
                # if discriminator_learning:
                #     if self.disc_loss[-1] < 0.25:
                #         discriminator_learning = False
                #         self.discriminator.noise_variance = self.discriminator.noise_variance*1.1
                # if discriminator_learning:
                #     self.discriminator.noise_variance = 0.1
                if self.epoch % 2 == 0:
                    self.discriminator.noise_variance = 1 - ((1+self.epoch) / (1+self.num_of_epochs))
                else:
                    self.discriminator.noise_variance = torch.rand((1,)).to(self.device)
                disc_loss = self.discriminator_loss(m_idx, model_output, self.data_output,
                                                    self.structure_output,
                                                    criterion_disc)
                disc_optimizer.zero_grad(set_to_none=True)
                disc_loss.backward()
                disc_optimizer.step()


                if epoch > 0:
                    self.disc_loss.append(disc_loss.detach())

                loss = self.loss_calculation(self.model, m_idx, model_output, self.data_input, self.data_output,
                                             self.structure_input, self.structure_output, criterion_model, norm)

                optimizer.zero_grad(set_to_none=True)
                loss[0].backward()
                optimizer.step()

                if self.validation_dataset is not None:
                    with torch.no_grad():
                        val_model_output = self.model(self.validation_dataset, spiking_probabilities)
                        val_loss = self.loss_calculation(self.model, val_idx, val_model_output, self.data_input_val,
                                                         self.data_output_val, self.structure_input_val,
                                                         self.structure_output_val, criterion_model, norm)

                self.train_loss.append(loss[0].item())
                self.val_loss.append(val_loss[0].item())
                self.vsi_loss.append(loss[1].item())
                self.entropy_loss.append(loss[2].item())
                self.grad_penalty.append(loss[3].item())
                self.kl_loss.append(loss[4].item())
                #self.sink_loss.append(loss[5].item())
                self.critical_loss.append(loss[6].item())
                self.hf_e_loss.append(loss[7].item())
                self.b_loss.append(loss[8].item())
                self.value_loss.append(loss[9].item())
                self.diff_fft_loss.append(loss[10].item())
                self.fft_loss.append(loss[11].item())
                self.diff_loss.append(loss[12].item())
                self.log_det_jacobian_loss.append(loss[13].item())
                self.freq_loss.append(loss[14].item())
                self.val_vsi_loss.append(val_loss[1].item())
                self.val_entropy_loss.append(val_loss[2].item())
                self.val_grad_penalty.append(val_loss[3].item())
                self.val_kl_loss.append(val_loss[4].item())
                #self.val_sink_loss.append(val_loss[5].item())
                self.val_critical_loss.append(val_loss[6].item())
                self.val_hf_e_loss.append(val_loss[7].item())
                self.val_b_loss.append(val_loss[8].item())
                self.val_value_loss.append(val_loss[9].item())
                self.val_diff_fft_loss.append(val_loss[10].item())
                self.val_fft_loss.append(val_loss[11].item())
                self.val_diff_loss.append(val_loss[12].item())
                self.val_log_det_jacobian_loss.append(val_loss[13].item())
                self.val_freq_loss.append(val_loss[14].item())


                # if self.train_loss[-1] < min(self.train_loss):
                    #discriminator_learning = True

                cpu_temp = WinTmp.CPU_Temp()
                gpu_temp = WinTmp.GPU_Temp()
                if self.epoch > 3:
                    if cpu_temp < 20:
                        cpu_temp = self.cpu_temp[-2]
                    if gpu_temp < 20:
                        gpu_temp = self.gpu_temp[-2]

                self.cpu_temp.append(cpu_temp)
                self.gpu_temp.append(gpu_temp)

                # t_stop = time.perf_counter()
                t += (t_pred - t_start) / 4
                if epoch > 25:
                    if val_loss[0] < min(self.val_loss[:-1]):
                        model_to_Save = self.model
                        print('saved_checkpoint')

                if len(self.train_loss) > 20:
                    loss_recent_history = np.array(self.train_loss)[-10:-1]
                    val_loss_recent_history = np.array(self.val_loss)[-10:-1]
                    mean_hist_losses = np.mean(loss_recent_history)
                    if loss_recent_history[-1] > loss_recent_history[-2] or loss_recent_history[-1] < \
                            loss_recent_history[-2] * 0.9 or loss_recent_history[-1] > 1e-3:
                        reiterate_data = 1
                    else:
                        reiterate_counter = 0
                        reiterate_data = 0
                    if reiterate_counter > 20:
                        reiterate_counter = 0
                        reiterate_data = 0
                    gloss = abs(np.sum(np.gradient(loss_recent_history)))
                    #print(gloss)
                    g_val_loss = np.sum(np.gradient(val_loss_recent_history))

                    if gloss > 1e2 or g_val_loss > 1e2:
                        grad_counter = 0
                    else:
                        grad_counter += 1
                    # NOTE: lowering lr for  better performance and reset lr within conditions
                    if grad_counter == 20 or reiterate_data == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.995
                            if param_group['lr'] < 1e-5: #or reiterate_data == 0:
                                param_group['lr'] = 1e-3
                                reiterate_counter = 0
                                reiterate_data = 0
                                print('optimizer -> lr back to starting point')
                        grad_counter = 0

                t_epoch_stop = time.perf_counter()
                t_epoch += (t_epoch_stop - t_epoch_start)
                if self.cpu_temp[-1] > 100 or self.gpu_temp[-1] > 75:
                    print('too hot! lets cool down a little bit')
                    torch.cuda.synchronize()
                    time.sleep(2)

                if (epoch + 1) % print_every_nth_frame == 0:
                    gpu_stats = gpustat.GPUStatCollection.new_query()
                    mem_usage = [gpu.memory_used for gpu in gpu_stats]
                    t_epoch_total = num_epochs * t_epoch
                    t_epoch_current = epoch * t_epoch

                    print(
                        f'P: {self.period}/{self.no_of_periods} | E: {((t_epoch_total - t_epoch_current) / (print_every_nth_frame * 60)):.2f} [min], '
                        f'vL: {val_loss[0].item():.6f}, '
                        f'mL: {loss[0].item():.6f}, '
                        f'dL: {disc_loss.item():.6f}, '
                        f'tpf: {((self.fsim.grid_size_x * self.fsim.grid_size_y) / (self.model.in_scale ** 2)) * (t * 1e3 / print_every_nth_frame / self.batch_size):.2f} [ms] \n'
                        f'CPU TEMP: {cpu_temp} [°C], '
                        f'GPU TEMP: {gpu_temp} [°C], '
                        f'GPU MEM: {round(mem_usage[0]*1e-3,3)} [GB] '
                    )
                    t = 0.
                    t_epoch = 0.
        else:
            self.model.load_state_dict(torch.load('model.pt'))

        time.sleep(1)
        torch.save(model_to_Save.state_dict(), 'model.pt')
        print('model_saved on disk')
        return self.model

    def dreaming_phase(self):
        pass


    def loss_calculation(self, model, idx, model_output, data_input, data_output, structure_input, structure_output,
                         criterion,norm='backward'):
        pred_r, pred_g, pred_b, pred_a, pred_s, deepS, nca_var,ortho_mean,ortho_max,log_det_jacobian_loss,freq_loss, lw = model_output

        r_in = data_input[:, 0:self.model.in_scale, :][idx]
        g_in = data_input[:, self.model.in_scale:self.model.in_scale * 2, :][idx]
        b_in = data_input[:, self.model.in_scale * 2:self.model.in_scale * 3, :][idx]
        a_in = data_input[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx]
        s_in = structure_input[idx]

        # Note: multitask contrast loss
        mask = torch.full_like(pred_r,0).to(self.device)
        b_idx_pool = torch.arange(0, self.batch_size, 1)
        c_idx_pool = torch.cartesian_prod(b_idx_pool, torch.arange(0, self.model.in_scale, 1)).int()
        b_idx_pool = b_idx_pool.int()
        l = self.batch_size // 5
        k = self.batch_size * 2
        if self.epoch > self.num_of_epochs*0.1:
            k = k // 2
        elif self.epoch > self.num_of_epochs*0.3:
            k = k // 4
            l = l // 2
        elif self.epoch > self.num_of_epochs*0.8:
            k = k // 8
            l = l // 4
        else:pass
        #print(int(k*self.loss_coeffs[0]),int(k*self.loss_coeffs[1]),int(k*self.loss_coeffs[2]),int(k*self.loss_coeffs[3]),int(k*self.loss_coeffs[4]))
        r_c_idx = torch.randperm(c_idx_pool.shape[0])[:int(k*self.loss_coeffs[0])]
        g_c_idx = torch.randperm(c_idx_pool.shape[0])[:int(k*self.loss_coeffs[1])]
        b_c_idx = torch.randperm(c_idx_pool.shape[0])[:int(k*self.loss_coeffs[2])]
        a_c_idx = torch.randperm(c_idx_pool.shape[0])[:int(k*self.loss_coeffs[3])]
        s_c_idx = torch.randperm(c_idx_pool.shape[0])[:int(k*self.loss_coeffs[4])]

        r_b_idx = torch.randperm(b_idx_pool.shape[0])[:int(l*self.loss_coeffs[0])]
        g_b_idx = torch.randperm(b_idx_pool.shape[0])[:int(l*self.loss_coeffs[1])]
        b_b_idx = torch.randperm(b_idx_pool.shape[0])[:int(l*self.loss_coeffs[2])]
        a_b_idx = torch.randperm(b_idx_pool.shape[0])[:int(l*self.loss_coeffs[3])]
        s_b_idx = torch.randperm(b_idx_pool.shape[0])[:int(l*self.loss_coeffs[4])]

        pred_r[c_idx_pool[r_c_idx]] = mask[c_idx_pool[r_c_idx]]
        pred_g[c_idx_pool[g_c_idx]] = mask[c_idx_pool[g_c_idx]]
        pred_b[c_idx_pool[b_c_idx]] = mask[c_idx_pool[b_c_idx]]
        pred_a[c_idx_pool[a_c_idx]] = mask[c_idx_pool[a_c_idx]]
        pred_s[c_idx_pool[s_c_idx]] = mask[c_idx_pool[s_c_idx]]

        pred_r[b_idx_pool[r_b_idx]] = mask[b_idx_pool[r_b_idx]]
        pred_g[b_idx_pool[g_b_idx]] = mask[b_idx_pool[g_b_idx]]
        pred_b[b_idx_pool[b_b_idx]] = mask[b_idx_pool[b_b_idx]]
        pred_a[b_idx_pool[a_b_idx]] = mask[b_idx_pool[a_b_idx]]
        pred_s[b_idx_pool[s_b_idx]] = mask[b_idx_pool[s_b_idx]]

        r_out = data_output[:, 0:self.model.in_scale, :][idx]
        g_out = data_output[:, self.model.in_scale:self.model.in_scale * 2, :][idx]  #.view(self.batch_size, -1)
        b_out = data_output[:, self.model.in_scale * 2:self.model.in_scale * 3, :][idx]  #.view(self.batch_size, -1)
        a_out = data_output[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx]  #.view(self.batch_size, -1)
        s_out = structure_output[idx]  #.view(self.batch_size, -1)

        r_out[c_idx_pool[r_c_idx]] = mask[c_idx_pool[r_c_idx]]
        g_out[c_idx_pool[g_c_idx]] = mask[c_idx_pool[g_c_idx]]
        b_out[c_idx_pool[b_c_idx]] = mask[c_idx_pool[b_c_idx]]
        a_out[c_idx_pool[a_c_idx]] = mask[c_idx_pool[a_c_idx]]
        s_out[c_idx_pool[s_c_idx]] = mask[c_idx_pool[s_c_idx]]

        r_out[b_idx_pool[r_b_idx]] = mask[b_idx_pool[r_b_idx]]
        g_out[b_idx_pool[g_b_idx]] = mask[b_idx_pool[g_b_idx]]
        b_out[b_idx_pool[b_b_idx]] = mask[b_idx_pool[b_b_idx]]
        a_out[b_idx_pool[a_b_idx]] = mask[b_idx_pool[a_b_idx]]
        s_out[b_idx_pool[s_b_idx]] = mask[b_idx_pool[s_b_idx]]

        t = 1 - self.fmot_in[idx]
        t_1 = self.fmot_in[idx]
        t = tt = t.unsqueeze(1)
        t_1 = tt_1 = t_1.unsqueeze(1)
        if pred_r.shape[0] != self.batch_size:
            n = int(pred_r.shape[0] / self.batch_size)
            r_in = r_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, r_in.shape[1], r_in.shape[2])
            g_in = g_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, g_in.shape[1], g_in.shape[2])
            b_in = b_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, b_in.shape[1], b_in.shape[2])
            a_in = a_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, a_in.shape[1], a_in.shape[2])
            s_in = s_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, s_in.shape[1], s_in.shape[2])

            r_out = r_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, r_out.shape[1], r_out.shape[2])
            g_out = g_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, g_out.shape[1], g_out.shape[2])
            b_out = b_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, b_out.shape[1], b_out.shape[2])
            a_out = a_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, a_out.shape[1], a_out.shape[2])
            s_out = s_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, s_out.shape[1], s_out.shape[2])
            t = t.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, t.shape[1], t.shape[1])
            t_1 = t_1.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, t_1.shape[1], t_1.shape[1])
        # Solution for learning of the dynamics in loss calculation

        # Note: Iage Validation of RGB channels
        # pred_rgb = torch.sigmoid(torch.cat([pred_r.unsqueeze(1),pred_g.unsqueeze(1),pred_b.unsqueeze(1)], dim=1))
        # # true_rgb = torch.sigmoid(torch.cat([r_out.unsqueeze(1),g_out.unsqueeze(1),b_out.unsqueeze(1)], dim=1))
        #
        # p1 = self.preprocess(pred_rgb)
        # p2 = self.preprocess(pred_rgb)
        # pred_validator_score = self.validator(p1,p2, return_embedding=False, scale_score=True).clamp(0,1)
        # print(pred_validator_score)


        # NOTE: Firs order difference
        diff_r_true = r_out - r_in
        diff_r_pred = pred_r - r_in
        loss_diff_r = criterion(t * diff_r_pred + t_1 * diff_r_true, diff_r_true)

        diff_g_true = g_out - g_in
        diff_g_pred = pred_g - g_in
        loss_diff_g = criterion(t * diff_g_pred + t_1 * diff_g_true, diff_g_true)
        diff_b_true = b_out - b_in
        diff_b_pred = pred_b - b_in
        loss_diff_b = criterion(t * diff_b_pred + t_1 * diff_b_true, diff_b_true)
        diff_a_true = a_out - a_in
        diff_a_pred = pred_a - a_in
        loss_diff_a = criterion(t * diff_a_pred + t_1 * diff_a_true, diff_a_true)
        diff_s_true = s_out - s_in
        diff_s_pred = pred_s - s_in
        loss_diff_s = criterion(t * diff_s_pred + t_1 * diff_s_true, diff_s_true)

        r_loss = torch.mean(loss_diff_r, dim=[1, 2])
        g_loss = torch.mean(loss_diff_g, dim=[1, 2])
        b_loss = torch.mean(loss_diff_b, dim=[1, 2])
        a_loss = torch.mean(loss_diff_a, dim=[1, 2])
        s_loss = torch.mean(loss_diff_s, dim=[1, 2])
        diff_loss = torch.mean(loss_diff_r + loss_diff_g + loss_diff_b + loss_diff_a + loss_diff_s)
        # diff_loss = loss_diff_r + loss_diff_g + loss_diff_b + loss_diff_a + loss_diff_s

        # Note: Fourier loss
        fft_out_pred_r = torch.real(torch.fft.rfft2(pred_r, norm=norm))
        fft_out_true_r = torch.real(torch.fft.rfft2(r_out, norm=norm))
        fft_out_pred_g = torch.real(torch.fft.rfft2(pred_g, norm=norm))
        fft_out_true_g = torch.real(torch.fft.rfft2(g_out, norm=norm))
        fft_out_pred_b = torch.real(torch.fft.rfft2(pred_b, norm=norm))
        fft_out_true_b = torch.real(torch.fft.rfft2(b_out, norm=norm))
        fft_out_pred_a = torch.real(torch.fft.rfft2(pred_a, norm=norm))
        fft_out_true_a = torch.real(torch.fft.rfft2(a_out, norm=norm))
        fft_out_pred_s = torch.real(torch.fft.rfft2(pred_s, norm=norm))
        fft_out_true_s = torch.real(torch.fft.rfft2(s_out, norm=norm))

        fft_in_true_r = torch.real(torch.fft.rfft2(r_in, norm=norm))
        fft_in_true_g = torch.real(torch.fft.rfft2(g_in, norm=norm))
        fft_in_true_b = torch.real(torch.fft.rfft2(b_in, norm=norm))
        fft_in_true_a = torch.real(torch.fft.rfft2(a_in, norm=norm))
        fft_in_true_s = torch.real(torch.fft.rfft2(s_in, norm=norm))

        fft_loss_r = criterion(t * fft_out_pred_r + t_1 * fft_out_true_r, fft_out_true_r)
        fft_loss_g = criterion(t * fft_out_pred_g + t_1 * fft_out_true_r, fft_out_true_g)
        fft_loss_b = criterion(t * fft_out_pred_b + t_1 * fft_out_true_r, fft_out_true_b)
        fft_loss_a = criterion(t * fft_out_pred_a + t_1 * fft_out_true_r, fft_out_true_a)
        fft_loss_s = criterion(t * fft_out_pred_s + t_1 * fft_out_true_r, fft_out_true_s)
        fft_loss = torch.mean(fft_loss_r + fft_loss_g + fft_loss_b + fft_loss_a + fft_loss_s)
        # fft_loss = fft_loss_r + fft_loss_g + fft_loss_b + fft_loss_a + fft_loss_s
        r_loss = torch.mean(fft_loss_r, dim=[1, 2])
        g_loss = torch.mean(fft_loss_g, dim=[1, 2])
        b_loss = torch.mean(fft_loss_b, dim=[1, 2])
        a_loss = torch.mean(fft_loss_a, dim=[1, 2])
        s_loss = torch.mean(fft_loss_s, dim=[1, 2])

        # Note: Fourier Gradient Loss
        diff_fft_true_r = fft_out_true_r - fft_in_true_r
        diff_fft_pred_r = fft_out_pred_r - fft_in_true_r
        diff_fft_loss_r = criterion(t * diff_fft_pred_r + t_1 * diff_fft_true_r, diff_fft_true_r)
        diff_fft_true_g = fft_out_true_g - fft_in_true_g
        diff_fft_pred_g = fft_out_pred_g - fft_in_true_g
        diff_fft_loss_g = criterion(t * diff_fft_pred_g + t_1 * diff_fft_true_r, diff_fft_true_g)
        diff_fft_true_b = fft_out_true_b - fft_in_true_b
        diff_fft_pred_b = fft_out_pred_b - fft_in_true_b
        diff_fft_loss_b = criterion(t * diff_fft_pred_b + t_1 * diff_fft_true_r, diff_fft_true_b)
        diff_fft_true_a = fft_out_true_a - fft_in_true_a
        diff_fft_pred_a = fft_out_pred_a - fft_in_true_a
        diff_fft_loss_a = criterion(t * diff_fft_pred_a + t_1 * diff_fft_true_r, diff_fft_true_a)
        diff_fft_true_s = fft_out_true_s - fft_in_true_s
        diff_fft_pred_s = fft_out_pred_s - fft_in_true_s
        diff_fft_loss_s = criterion(t * diff_fft_pred_s + t_1 * diff_fft_true_r, diff_fft_true_s)
        diff_fft_loss = torch.mean(
            diff_fft_loss_r + diff_fft_loss_g + diff_fft_loss_b + diff_fft_loss_a + diff_fft_loss_s)
        # diff_fft_loss = diff_fft_loss_r + diff_fft_loss_g + diff_fft_loss_b + diff_fft_loss_a + diff_fft_loss_s

        r_loss += torch.mean(diff_fft_loss_r, dim=[1, 2])
        g_loss += torch.mean(diff_fft_loss_g, dim=[1, 2])
        b_loss += torch.mean(diff_fft_loss_b, dim=[1, 2])
        a_loss += torch.mean(diff_fft_loss_a, dim=[1, 2])
        s_loss += torch.mean(diff_fft_loss_s, dim=[1, 2])

        # Note : Exact value loss
        loss_r = criterion(t * pred_r + t_1 * r_out, r_out)
        loss_g = criterion(t * pred_g + t_1 * g_out, g_out)
        loss_b = criterion(t * pred_b + t_1 * b_out, b_out)
        loss_alpha = criterion(t * pred_a + t_1 * a_out, a_out)
        loss_s = criterion(t * pred_s + t_1 * s_out, s_out)
        value_loss = torch.mean(loss_r + loss_g + loss_b + loss_alpha + loss_s)
        #value_loss = loss_r + loss_g + loss_b + loss_alpha + loss_s
        r_loss += torch.mean(loss_r, dim=[1, 2])
        g_loss += torch.mean(loss_g, dim=[1, 2])
        b_loss += torch.mean(loss_b, dim=[1, 2])
        a_loss += torch.mean(loss_alpha, dim=[1, 2])
        s_loss += torch.mean(loss_s, dim=[1, 2])

        # Note: Edge Loss
        boundary_loss_r = self.boundary_loss(t * pred_r+ t_1 * r_out,r_out)
        boundary_loss_g = self.boundary_loss(t * pred_g + t_1 * g_out, g_out)
        boundary_loss_b = self.boundary_loss(t * pred_b + t_1 * b_out, b_out)
        boundary_loss_a = self.boundary_loss(t * pred_a + t_1 * a_out, a_out)
        boundary_loss_s = self.boundary_loss(t * pred_s + t_1 * s_out, s_out)
        boundary_loss  = torch.mean(boundary_loss_r+boundary_loss_g+boundary_loss_b+boundary_loss_a+boundary_loss_s)

        # v_loss_r = self.total_variation_loss(pred_r)
        # v_loss_g = self.total_variation_loss(pred_g)
        # v_loss_b = self.total_variation_loss(pred_b)
        # v_loss_a = self.total_variation_loss(pred_a)
        # v_loss_s = self.total_variation_loss(pred_s)
        # var_e_loss = torch.mean(v_loss_r + v_loss_g + v_loss_b + v_loss_a + v_loss_s)

        hf_loss_r = self.fft_high_frequency_loss(t * pred_r+ t_1 * r_out,r_out)
        hf_loss_g = self.fft_high_frequency_loss(t * pred_g + t_1 * g_out, g_out)
        hf_loss_b = self.fft_high_frequency_loss(t * pred_b + t_1 * b_out, b_out)
        hf_loss_a = self.fft_high_frequency_loss(t * pred_a + t_1 * a_out, a_out)
        hf_loss_s = self.fft_high_frequency_loss(t * pred_s + t_1 * s_out, s_out)
        hf_e_loss = torch.mean(hf_loss_r+hf_loss_g+hf_loss_b+hf_loss_a+hf_loss_s)

        r_loss += boundary_loss_r + hf_loss_r# + v_loss_r
        g_loss += boundary_loss_g  + hf_loss_g#+ v_loss_g
        b_loss += boundary_loss_b  + hf_loss_b#+ v_loss_b
        a_loss += boundary_loss_a + hf_loss_a# + v_loss_a
        s_loss += boundary_loss_s + hf_loss_s# + v_loss_s
        # # Note: Deep Supervision Loss cosine sim

        t = t.squeeze()
        t_1 = t_1.squeeze()
        # NCA Criticality loss
        target_variance = torch.full_like(nca_var,0.49,requires_grad=True)
        critical_loss = torch.mean(torch.abs(target_variance - nca_var))

        # Sinkhorn Loss
        sink_loss_r = torch.mean(
            self.sinkhorn_loss(t.unsqueeze(1) * pred_r.flatten(start_dim=1)+t_1.unsqueeze(1)*r_out.flatten(start_dim=1), r_out.flatten(start_dim=1)))
        sink_loss_g = torch.mean(
            self.sinkhorn_loss(t.unsqueeze(1) * pred_g.flatten(start_dim=1)+t_1.unsqueeze(1)*g_out.flatten(start_dim=1), g_out.flatten(start_dim=1)))
        sink_loss_b = torch.mean(
            self.sinkhorn_loss(t.unsqueeze(1) * pred_b.flatten(start_dim=1)+t_1.unsqueeze(1)*b_out.flatten(start_dim=1), b_out.flatten(start_dim=1)))
        sink_loss_a = torch.mean(
            self.sinkhorn_loss(t.unsqueeze(1) * pred_a.flatten(start_dim=1)+t_1.unsqueeze(1)*a_out.flatten(start_dim=1), a_out.flatten(start_dim=1)))
        sink_loss_s = torch.mean(
            self.sinkhorn_loss(t.unsqueeze(1) * pred_s.flatten(start_dim=1)+t_1.unsqueeze(1)*s_out.flatten(start_dim=1), s_out.flatten(start_dim=1)))
        sink_loss = sink_loss_r + sink_loss_g + sink_loss_b + sink_loss_a + sink_loss_s

        r_loss += sink_loss_r
        g_loss += sink_loss_g
        b_loss += sink_loss_b
        a_loss += sink_loss_a
        s_loss += sink_loss_s

        # KL Div LOSS
        pred_distribution_r = f.log_softmax(t.unsqueeze(1).unsqueeze(2)*pred_r.flatten(start_dim=1), dim=-1)
        target_distribution_r = f.softmax(r_out.flatten(start_dim=1), dim=-1)
        target_distribution_r_w = f.log_softmax(t_1.unsqueeze(1).unsqueeze(2)*r_out.flatten(start_dim=1), dim=-1)
        pred_distribution_g = f.log_softmax(t.unsqueeze(1).unsqueeze(2)*pred_g.flatten(start_dim=1), dim=-1)
        target_distribution_g = f.softmax(g_out.flatten(start_dim=1), dim=-1)
        target_distribution_g_w = f.log_softmax(t_1.unsqueeze(1).unsqueeze(2)*g_out.flatten(start_dim=1), dim=-1)
        pred_distribution_b = f.log_softmax(t.unsqueeze(1).unsqueeze(2)*pred_b.flatten(start_dim=1), dim=-1)
        target_distribution_b = f.softmax(b_out.flatten(start_dim=1), dim=-1)
        target_distribution_b_w = f.log_softmax(t_1.unsqueeze(1).unsqueeze(2)*b_out.flatten(start_dim=1), dim=-1)
        pred_distribution_a = f.log_softmax(t.unsqueeze(1).unsqueeze(2)*pred_a.flatten(start_dim=1), dim=-1)
        target_distribution_a = f.softmax(a_out.flatten(start_dim=1), dim=-1)
        target_distribution_a_w = f.log_softmax(t_1.unsqueeze(1).unsqueeze(2)*a_out.flatten(start_dim=1), dim=-1)
        pred_distribution_s = f.log_softmax(t.unsqueeze(1).unsqueeze(2)*pred_s.flatten(start_dim=1), dim=-1)
        target_distribution_s = f.softmax(s_out.flatten(start_dim=1), dim=-1)
        target_distribution_s_w = f.log_softmax(t_1.unsqueeze(1).unsqueeze(2)*s_out.flatten(start_dim=1), dim=-1)
        kl_loss_r = f.kl_div(pred_distribution_r+target_distribution_r_w, target_distribution_r, reduction="sum", log_target=False)
        kl_loss_g = f.kl_div(pred_distribution_g+target_distribution_g_w, target_distribution_g, reduction="sum", log_target=False)
        kl_loss_b = f.kl_div(pred_distribution_b+target_distribution_b_w, target_distribution_b, reduction="sum", log_target=False)
        kl_loss_a = f.kl_div(pred_distribution_a+target_distribution_a_w, target_distribution_a, reduction="sum", log_target=False)
        kl_loss_s = f.kl_div(pred_distribution_s+target_distribution_s_w, target_distribution_s, reduction="sum", log_target=False)

        # Combine the separate KL divergence losses if desired
        kl_loss = kl_loss_r + kl_loss_g + kl_loss_b + kl_loss_a + kl_loss_s

        r_loss += kl_loss_r
        g_loss += kl_loss_g
        b_loss += kl_loss_b
        a_loss += kl_loss_a
        s_loss += kl_loss_s
        grad_penalty = torch.mean(criterion.gradient_penalty(model))

        entropy_loss = 0
        for pred_channel,true_channel_w ,true_channel,losses in zip([t.unsqueeze(1).unsqueeze(2)*pred_r, t.unsqueeze(1).unsqueeze(2)*pred_g, t.unsqueeze(1).unsqueeze(2)*pred_b, t.unsqueeze(1).unsqueeze(2)*pred_a, t.unsqueeze(1).unsqueeze(2)*pred_s],
                                                                    [t_1.unsqueeze(1).unsqueeze(2)*r_out, t_1.unsqueeze(1).unsqueeze(2)*g_out, t_1.unsqueeze(1).unsqueeze(2)*b_out, t_1.unsqueeze(1).unsqueeze(2)*a_out, t_1.unsqueeze(1).unsqueeze(2)*s_out],
                                                                    [r_out, g_out, b_out, a_out, s_out],[r_loss,g_loss,b_loss,a_loss,s_loss]):
            p_pred = f.softmax(pred_channel.flatten(), dim=0)
            p_true = f.softmax(true_channel.flatten(), dim=0)
            p_true_w = f.softmax(true_channel_w.flatten(), dim=0)
            entropy_pred = torch.sum(-p_pred * torch.log(p_pred + 1e-9))
            entropy_true = torch.sum(-p_true * torch.log(p_true + 1e-9))
            entropy_true_w = torch.sum(-p_true_w * torch.log(p_true_w + 1e-9))
            entropy_loss += criterion(entropy_pred+entropy_true_w , entropy_true)
            losses += entropy_loss


        self.loss_coeffs[0] = r_loss.mean().item()
        self.loss_coeffs[1] = g_loss.mean().item()
        self.loss_coeffs[2] = b_loss.mean().item()
        self.loss_coeffs[3] = a_loss.mean().item()
        self.loss_coeffs[4] = s_loss.mean().item()

        rgb_pred = torch.cat([pred_r.unsqueeze(1),pred_g.unsqueeze(1),pred_b.unsqueeze(1)],dim=1)
        rgb_true = torch.cat([r_out.unsqueeze(1),g_out.unsqueeze(1),b_out.unsqueeze(1)],dim=1)
        rgb_pred = rgb_pred - rgb_pred.min()
        rgb_pred = rgb_pred / rgb_pred.max()
        rgb_true = rgb_true - rgb_true.min()
        rgb_true = rgb_true / rgb_true.max()
        rgb_pred = torch.nan_to_num(rgb_pred, nan=0.,posinf=0.,neginf=0.)#.clamp(0,1)
        rgb_true = torch.nan_to_num(rgb_true, nan=0.,posinf=0.,neginf=0.)#.clamp(0,1)
        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t_1 = t_1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #print(t*rgb_pred.max(),t_1*rgb_true.max(),t*rgb_pred.min(),t_1*rgb_true.min(),rgb_true.min(),rgb_true.max())
        vsi_loss = self.vsi(t*rgb_pred+t_1*rgb_true,rgb_true)
        loss_min = self.loss_coeffs.min()
        loss_range = self.loss_coeffs.max() - loss_min + 1e-12
        normalized_loss_coeffs = (self.loss_coeffs - loss_min) / loss_range
        temperature = 0.1 + 0.5 * normalized_loss_coeffs.std().item()
        self.loss_coeffs = f.softmax((normalized_loss_coeffs.max()-normalized_loss_coeffs) / temperature,dim=0)
        #print( self.loss_coeffs)
        disc_loss = -torch.log(self.disc_loss[-1])

        # if self.epoch == 5:
        #     ls = [vsi_loss ,entropy_loss   , kl_loss  , sink_loss  , diff_fft_loss  , critical_loss  , diff_loss  , fft_loss  ,  value_loss , ortho_mean  , log_det_jacobian_loss.mean()  , b_loss  , hf_e_loss  , freq_loss.mean()]
        #     adjusted_coeffs = [1 / loss if loss != 0 else 0 for loss in ls]
        #     A, B, C, D, E, F, G, H, I, J, K, L, M,N = adjusted_coeffs
        #     model.A, model.B, model.C, model.D, model.E, model.F, model.G, model.H, model.I, model.J, model.K, model.L, model.M,model.N = A.detach(), B.detach(), C.detach(), D.detach(), E.detach(), F.detach(), G.detach(), H.detach(), I.detach(), J.detach(), K.detach(),L.detach(), M.detach(),N.detach()
            #scaled_losses = [coeff * loss for coeff, loss in zip(adjusted_coeffs, ls)]
        # lw_min = lw.min()
        # lw_range = lw.max() - loss_min + 1e-12
        # normalized_lw_coeffs = (lw - lw_min) / lw_range
        # temperature = 30.# + 0.5 * normalized_lw_coeffs.std().item()
        # lw = f.softmax((normalized_lw_coeffs.max() - normalized_lw_coeffs) / temperature, dim=0)
        #final_loss =vsi_loss*2+disc_loss+entropy_loss*5e-2+grad_penalty*2e-2+kl_loss*5e-6+sink_loss*8e-1+diff_fft_loss*1e3+critical_loss*2e-1+diff_loss*1e1+fft_loss*2e4+2e1*value_loss+ortho_mean*1e-4+log_det_jacobian_loss.mean()*1e-6+b_loss*1e-5+hf_e_loss*5e-1+freq_loss.mean()*1e-3
        # final_loss =vsi_loss*2+disc_loss+entropy_loss*5e-2+grad_penalty*2e-2+kl_loss*5e-6+sink_loss*8e-1+diff_fft_loss*1e3+critical_loss*2e-1+diff_loss*1e1+fft_loss*2e4+2e1*value_loss+ortho_mean*1e-4+log_det_jacobian_loss.mean()*1e-6+b_loss*1e-5+hf_e_loss*5e-1+freq_loss.mean()*1e-3

        final_loss = torch.cat([vsi_loss.mean().unsqueeze(0)*2 * model.A.unsqueeze(0) ,
                                torch.abs((disc_loss.mean().unsqueeze(0)**2)-(disc_loss.mean().unsqueeze(0))+(0.5**2))*20,
                                entropy_loss.mean().unsqueeze(0)*1e-1 * model.B.unsqueeze(0) ,
                                grad_penalty.mean().unsqueeze(0) * 2e-2 ,
                                kl_loss.mean().unsqueeze(0)*1e-6 * model.C.unsqueeze(0) ,
                                sink_loss.mean().unsqueeze(0)*2e-1 * model.D.unsqueeze(0) ,
                                diff_fft_loss.mean().unsqueeze(0)*1e3 * model.E.unsqueeze(0) ,
                                critical_loss.mean().unsqueeze(0)*2e-1 * model.F.unsqueeze(0) ,
                                diff_loss.mean().unsqueeze(0)*1e1 * model.G.unsqueeze(0) ,
                                fft_loss.mean().unsqueeze(0)*5e3 * model.H.unsqueeze(0) ,
                                model.I.unsqueeze(0) * 2e1*value_loss.mean().unsqueeze(0) ,
                                ortho_mean.mean().unsqueeze(0)*1e-4 * model.J.unsqueeze(0) ,
                                log_det_jacobian_loss.mean().unsqueeze(0)*1e-6 * model.K.unsqueeze(0) ,
                                boundary_loss.mean()*5e-1 * model.L.unsqueeze(0) ,
                                hf_e_loss.mean().unsqueeze(0)*5e-1 * model.M.unsqueeze(0) ,
                                freq_loss.mean().unsqueeze(0)*1e-3 * model.N.unsqueeze(0),
                               ],dim=0)
        #print(final_loss.tolist())
        #print(disc_loss.item(),entropy_loss.item()*5e-2,grad_penalty.item()*2e-2,kl_loss.item()*5e-6,sink_loss.item()*8e-1,torch.mean(diff_fft_loss).item()*1e3,critical_loss.item()*2e-1,torch.mean(diff_loss).item()*3e1,torch.mean(fft_loss).item()*2e4,2e1*torch.mean(value_loss).item(),ortho_mean.item()*1e-4,log_det_jacobian_loss.mean().item()*1e-5,b_loss.item()*1e-5,hf_e_loss.item()*5e-1,freq_loss.mean().item()*1e-3)
        return (torch.mean(final_loss),
                model.A*2*vsi_loss ,
                model.B*entropy_loss*1e-1,
                grad_penalty* 2e-2 ,
                model.C*1e-6*kl_loss,
                sink_loss* model.D*2e-1,
                critical_loss* model.F*2e-1,
                hf_e_loss* model.M*5e-1,
                boundary_loss* model.L*5e-1,
                2e1*value_loss* model.I,
                diff_fft_loss* model.E*1e3,
                fft_loss* model.H*5e3,
                model.G*1e1*diff_loss,
                log_det_jacobian_loss.mean()* model.K*1e-6,
                model.N*freq_loss.mean()*1e-3)


    def boundary_loss(self,pred, target):
        grad_pred_h = torch.abs(pred[:,  0:1, :] - pred[:,  -2:-1, :])
        grad_pred_w = torch.abs(pred[:,  :, 0:1] - pred[:,  :, -2:-1])
        grad_target_h = torch.abs(target[:,  0:1, :] - target[:, -2:-1, :])
        grad_target_w = torch.abs(target[:,  :, 0:1] - target[:,  :, -2:-1])
        boundary_h = torch.abs(grad_pred_h - grad_target_h).mean()
        boundary_w = torch.abs(grad_pred_w - grad_target_w).mean()
        return boundary_h + boundary_w

    def total_variation_loss(self,pred):
        tv_h = torch.abs(pred[:,  1:, :] - pred[:,  :-1, :]).mean()
        tv_w = torch.abs(pred[:,  :, 1:] - pred[:,  :, :-1]).mean()
        return tv_h + tv_w

    def fft_high_frequency_loss(self, pred, target, cutoff_ratio=0.7):
        B, H, W = pred.shape
        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_target = torch.fft.fft2(target, dim=(-2, -1))
        fft_shifted_pred = torch.fft.fftshift(fft_pred, dim=(-2, -1))
        fft_shifted_target = torch.fft.fftshift(fft_target, dim=(-2, -1))
        mask = torch.ones((B, H, W ), device=self.device)
        center_x = (H-1) // 2
        center_y = (W-1) // 2
        cutoff_x = int(cutoff_ratio * center_x)
        cutoff_y = int(cutoff_ratio * center_y)
        mask[:, center_x - cutoff_x:center_x + cutoff_x, center_y - cutoff_y:center_y + cutoff_y] = 0.0
        high_freq_pred = fft_shifted_pred * mask
        high_freq_target = fft_shifted_target * mask
        diff = torch.abs(high_freq_pred - high_freq_target)**2
        loss = torch.mean(diff)
        return loss

    def fft_noise(self,noise_iter, noise):
        cutoff_ratio = torch.rand((1,)).to(self.device)
        # if cutoff_ratio < 0.5:
        #     pass
        # else:
        noise = torch.sum(torch.rand((noise_iter.int().item(),*noise.shape)).to(self.device),dim=0)
        noise = noise / (noise_iter +1e-12)
        H, W = noise.shape
        fft_noise = torch.fft.fft2(noise, dim=(-2, -1))
        fft_noise_shifted = torch.fft.fftshift(fft_noise, dim=(-2, -1))
        mask = torch.ones_like(fft_noise_shifted, device=self.device)
        center_x, center_y = (H-1) // 2, (W-1) // 2
        cutoff_x = int(cutoff_ratio * center_x)
        cutoff_y = int(cutoff_ratio * center_y)
        mask[center_x - cutoff_x:center_x + cutoff_x,center_y - cutoff_y:center_y + cutoff_y] = 0.0
        fft_noise = fft_noise * mask
        fft_noise_shifted = torch.fft.ifftshift(fft_noise, dim=(-2, -1))
        noise = torch.fft.ifft2(fft_noise_shifted, s=(H, W), dim=(-2, -1)).real
        return noise

    def fft_data(self, data):
        augment_flag = torch.rand((1,)).to(self.device)
        if augment_flag < 0.5:
            return data
        else:
            hf_flag = torch.rand((1,))
            H, W = data.shape
            fft_data = torch.fft.fft2(data, dim=(-2, -1))
            fft_data_shifted = torch.fft.fftshift(fft_data, dim=(-2, -1))
            mask = torch.zeros((H, W ), device=self.device)
            center_x, center_y = (H-1) // 2, (W-1) // 2
            epoch_fraction = 2*(1+self.epoch) / (1+self.num_of_epochs)
            cutoff_ratio = 0.1 + 0.9 * epoch_fraction
            if cutoff_ratio > 1.:
                return data
            cutoff_x = int(cutoff_ratio * center_x)
            cutoff_y = int(cutoff_ratio * center_y)
            if hf_flag < cutoff_ratio:
                mask[center_x - cutoff_x:center_x + cutoff_x, center_y - cutoff_y:center_y + cutoff_y] = 1
                high_freq_pred = fft_data_shifted * (1 - mask)
            else:
                mask[center_x - cutoff_x:center_x + cutoff_x, center_y - cutoff_y:center_y + cutoff_y] = 1
                high_freq_pred = fft_data * mask
            high_freq_pred_shifted = torch.fft.ifftshift(high_freq_pred, dim=(-2, -1))
            data = torch.fft.ifft2(high_freq_pred_shifted, s=(H, W), dim=(-2, -1)).real
            return data

    def discriminator_loss(self, idx, model_output, data_output, structure_output, criterion):

        dataset = (
            self.data_input[idx], self.structure_input[idx], self.meta_input_h1[idx], self.meta_input_h2[idx],
            self.meta_input_h3[idx], self.meta_input_h4[idx], self.meta_input_h5[idx], self.noise_diff_in[idx],
            self.fmot_in_binary[idx],
            self.meta_output_h1[idx],
            self.meta_output_h2[idx], self.meta_output_h3[idx], self.meta_output_h4[idx], self.meta_output_h5[idx],
            self.noise_diff_out[idx])

        pred_r, pred_g, pred_b, pred_a, pred_s, _, _,_,_,_,_,_ = model_output
        r_out = data_output[:, 0:self.model.in_scale, :][idx].detach()
        g_out = data_output[:, self.model.in_scale:self.model.in_scale * 2, :][idx].detach()
        b_out = data_output[:, self.model.in_scale * 2:self.model.in_scale * 3, :][idx].detach()
        a_out = data_output[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx].detach()
        s_out = structure_output[idx].detach()

        pred = torch.cat([pred_r.detach().unsqueeze(1), pred_g.detach().unsqueeze(1), pred_b.detach().unsqueeze(1), pred_a.detach().unsqueeze(1), pred_s.detach().unsqueeze(1)], dim=1)
        true = torch.cat([r_out.unsqueeze(1), g_out.unsqueeze(1), b_out.unsqueeze(1), a_out.unsqueeze(1), s_out.unsqueeze(1)], dim=1)
        # Note : Fill value for label smoothing
        fake_labels = torch.full((pred.shape[0], 1), 0.05).to(self.device)
        true_labels = torch.full((true.shape[0], 1), 0.95).to(self.device)
        combined_data = torch.cat([pred, true], dim=0)
        # _,_, h, w = combined_data.shape
        # mask = torch.ones_like(combined_data, device=self.device)
        # center_x, center_y = w // 2, h // 2
        # mask_radius = torch.randint(0,5,(1,))
        # mask[:, :, center_y-mask_radius:center_y+mask_radius+1, center_x-mask_radius:center_x+mask_radius+1] = 0.0
        # combined_data = combined_data * mask
        combined_labels = torch.cat([fake_labels, true_labels], dim=0)
        shuffle_idx = torch.randint(0, combined_data.shape[0], (int(combined_data.shape[0] / 2),)).to(self.device)
        shuffled_labels = combined_labels[shuffle_idx]
        disc_pred = self.discriminator(combined_data, dataset, shuffle_idx)
        disc_loss = criterion(disc_pred, shuffled_labels)
        return disc_loss

    @staticmethod
    def seed_setter(seed):
        s = torch.randint(0, seed, (1,))
        torch.manual_seed(2024 + s)
        s = np.random.randint(0, seed)
        np.random.seed(2024 + s)
        s = random.randint(0, seed)
        random.seed(2024 + s)


class AddGaussianNoise(object):
    def __init__(self,device, mean=0., std=1.):
        self.device = device
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)