import copy
import itertools
import os.path
import random
import struct
import time
from statistics import mean
import ssim
import kornia
import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from torch.autograd import grad
import torch.nn.utils as nn_utils
import torch.nn.functional as f


class teacher(object):
    def __init__(self, model, device):
        super(teacher, self).__init__()
        #self.t = None
        self.validation_dataset = None
        self.max_seed = int(1e2)
        self.model = model
        self.device = device
        self.fsim = None
        self.period = 1
        self.no_of_periods = 1
        self.data_tensor = None
        self.meta_tensor = None
        self.meta_binary = None
        self.field_names = None
        self.no_frame_samples, self.first_frame, self.last_frame, self.frame_skip = None, None, None, None
        self.ssim_loss = ssim.MS_SSIM(data_range=1., channel=5, use_padding=True, window_size=1).to(self.device)

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
        self.val_loss = []

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
                            data_tensor.append(ptfile['data'][:, :, j] / 255)
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
        frame_samples = random.sample(list(set(fdens_idx)), k=self.no_frame_samples)
        f_dens_pos = len(fdens_idx)
        fdens_idx = frame_samples
        # Attention! : RGB is not 000000,111111,222222 but 012,012,012,012...
        rgb_idx = np.array([i for i, x in enumerate(self.field_names) if x == "rgb"])
        r_idx = rgb_idx[::3][frame_samples]
        g_idx = (rgb_idx[::3] + 1)[frame_samples]
        b_idx = (rgb_idx[::3] + 2)[frame_samples]
        # fs = np.array(frame_samples)+150

        alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])[frame_samples]
        fuel_slices = self.data_tensor[fdens_idx]
        min_val = fuel_slices.min()
        max_val = fuel_slices.max()
        fuel_slices = (fuel_slices - min_val) / ((max_val - min_val) + 1e-10)

        r_slices = self.data_tensor[r_idx]
        g_slices = self.data_tensor[g_idx]
        b_slices = self.data_tensor[b_idx]
        alpha_slices = self.data_tensor[alpha_idx]
        meta_binary_slices = self.meta_binary[fdens_idx]

        # gt = np.stack((r_slices[0].cpu().numpy(), g_slices[0].cpu().numpy(), b_slices[0].cpu().numpy()), axis=2)
        # print(gt.shape)
        # plt.imshow(gt.astype(np.uint8) , alpha=alpha_slices[0].cpu().numpy())
        # plt.show()
        x_range = range(self.fsim.N_boundary + self.input_window_size,
                        fuel_slices[0].shape[0] - self.fsim.N_boundary - self.input_window_size)
        y_range = range(self.fsim.N_boundary + self.input_window_size,
                        fuel_slices[0].shape[1] - self.fsim.N_boundary - self.input_window_size)
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
            noise_mod = 1.
            if noise_flag < 3:
                noise_variance_in = torch.tensor(0.).to(self.device)
                noise_variance_out = torch.tensor(0.).to(self.device)
                noise_variance_in_binary = torch.zeros(32).to(self.device)
                noise_variance_out_binary = torch.zeros(32).to(self.device)
            elif 3 < noise_flag < 8:
                noise_variance_in = torch.rand(size=(1,)) * noise_mod
                noise_variance_in_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_in).tobytes())
                noise_variance_in = noise_variance_in.to(self.device)
                noise_variance_in_binary = [int(noise_variance_in_binary[i], 2) for i in
                                            range(0, len(noise_variance_in_binary), 1)]
                noise_variance_in_binary = torch.tensor(np.array(noise_variance_in_binary)).to(self.device)
                noise_variance_out = torch.tensor(0.).to(self.device)
                noise_variance_out_binary = torch.zeros(32).to(self.device)
            elif 8 < noise_flag < 10:
                noise_variance_out = torch.rand(size=(1,)) * noise_mod
                noise_variance_out_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_out).tobytes())
                noise_variance_out = noise_variance_out.to(self.device)
                noise_variance_out_binary = [int(noise_variance_out_binary[i], 2) for i in
                                             range(0, len(noise_variance_out_binary), 1)]
                noise_variance_out_binary = torch.tensor(np.array(noise_variance_out_binary)).to(self.device)
                noise_variance_in = torch.tensor(0.).to(self.device)
                noise_variance_in_binary = torch.zeros(32).to(self.device)
            else:
                noise_variance_in = torch.rand(size=(1,)) * noise_mod
                noise_variance_in_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_in).tobytes())
                noise_variance_in = noise_variance_in.to(self.device)
                noise_variance_in_binary = [int(noise_variance_in_binary[i], 2) for i in
                                            range(0, len(noise_variance_in_binary), 1)]
                noise_variance_in_binary = torch.tensor(np.array(noise_variance_in_binary)).to(self.device)
                noise_variance_out = torch.rand(size=(1,)) * noise_mod
                noise_variance_out_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_out).tobytes())
                noise_variance_out = noise_variance_out.to(self.device)
                noise_variance_out_binary = [int(noise_variance_out_binary[i], 2) for i in
                                             range(0, len(noise_variance_out_binary), 1)]
                noise_variance_out_binary = torch.tensor(np.array(noise_variance_out_binary)).to(self.device)

            # Note: Flow Matching OT Noise gen
            fmot_coef = torch.rand(size=(1,))
            fmot_coef_binary = ''.join(f'{c:08b}' for c in np.float32(fmot_coef).tobytes())
            fmot_coef_binary = [int(fmot_coef_binary[i], 2) for i in range(0, len(fmot_coef_binary), 1)]
            fmot_coef_binary = torch.tensor(np.array(fmot_coef_binary)).to(self.device)

            idx_input = random.choice(range(0, fuel_slices.shape[0]))
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

            idx_output = random.choice(range(0, fuel_slices.shape[0]))
            offset_x = random.randint(int(-self.input_window_size), int(self.input_window_size))
            offset_y = random.randint(int(-self.input_window_size), int(self.input_window_size))
            central_point_x_out = central_point_x_in + offset_x
            central_point_y_out = central_point_y_in + offset_y

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
            fuel_subslice_in = fuel_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                noise_variance_in * torch.rand_like(fuel_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0)
            r_subslice_in = r_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                noise_variance_in * torch.rand_like(r_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0)
            g_subslice_in = g_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                noise_variance_in * torch.rand_like(g_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0)
            b_subslice_in = b_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                noise_variance_in * torch.rand_like(b_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0)
            alpha_subslice_in = alpha_slices[idx_input, slice_x_in, slice_y_in] + torch.nan_to_num(
                noise_variance_in * torch.rand_like(alpha_slices[idx_input, slice_x_in, slice_y_in]).to(self.device),
                nan=0.0)
            data_input_subslice = torch.cat([r_subslice_in, g_subslice_in, b_subslice_in, alpha_subslice_in], dim=0)

            meta_step_in = meta_binary_slices[idx_input][0]
            meta_step_in_numeric = self.meta_tensor[idx_input][0]

            meta_fuel_initial_speed_in = meta_binary_slices[idx_input][1]
            meta_fuel_cut_off_time_in = meta_binary_slices[idx_input][2]
            meta_igni_time_in = meta_binary_slices[idx_input][3]
            meta_ignition_temp_in = meta_binary_slices[idx_input][4]
            meta_viscosity_in = meta_binary_slices[idx_input][14]
            meta_diff_in = meta_binary_slices[idx_input][15]
            meta_input_subslice = torch.cat([meta_step_in, meta_fuel_initial_speed_in,
                                             meta_fuel_cut_off_time_in, meta_igni_time_in,
                                             meta_ignition_temp_in, meta_viscosity_in, meta_diff_in], dim=0)

            # Note : Output data
            fuel_subslice_out = fuel_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                noise_variance_out * torch.rand_like(fuel_slices[idx_output, slice_x_out, slice_y_out]).to(self.device),
                nan=0.0)
            r_subslice_out = r_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                noise_variance_out * torch.rand_like(r_slices[idx_output, slice_x_out, slice_y_out]).to(self.device),
                nan=0.0)
            g_subslice_out = g_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                noise_variance_out * torch.rand_like(g_slices[idx_output, slice_x_out, slice_y_out]).to(self.device),
                nan=0.0)
            b_subslice_out = b_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                noise_variance_out * torch.rand_like(b_slices[idx_output, slice_x_out, slice_y_out]), nan=0.0)
            alpha_subslice_out = alpha_slices[idx_output, slice_x_out, slice_y_out] + torch.nan_to_num(
                noise_variance_out * torch.rand_like(alpha_slices[idx_output, slice_x_out, slice_y_out]).to(
                    self.device), nan=0.0)
            data_output_subslice = torch.cat([r_subslice_out, g_subslice_out, b_subslice_out, alpha_subslice_out],
                                             dim=0)

            meta_step_out = meta_binary_slices[idx_output][0]
            meta_step_out_numeric = self.meta_tensor[idx_output][0]
            meta_fuel_initial_speed_out = meta_binary_slices[idx_output][1]
            meta_fuel_cut_off_time_out = meta_binary_slices[idx_output][2]
            meta_igni_time_out = meta_binary_slices[idx_output][3]
            meta_ignition_temp_out = meta_binary_slices[idx_output][4]
            meta_viscosity_out = meta_binary_slices[idx_output][14]
            meta_diff_out = meta_binary_slices[idx_output][15]
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
            if self.epoch > self.num_of_epochs * 0.03 or create_val_dataset == 1:
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
        self.model.load_state_dict(torch.load('model.pt'))
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
        min_val = fuel_slices.min()
        max_val = fuel_slices.max()
        fuel_slices = (fuel_slices - min_val) / ((max_val - min_val) + 1e-12)

        r_slices = self.data_tensor[r_idx]
        g_slices = self.data_tensor[g_idx]
        b_slices = self.data_tensor[b_idx]
        alpha_slices = self.data_tensor[alpha_idx]
        meta_binary_slices = self.meta_binary[fdens_idx]

        # Note: IDX preparation
        central_points_x = np.arange(self.input_window_size, fuel_slices.shape[1] - self.input_window_size + 1)
        central_points_y = np.arange(self.input_window_size, fuel_slices.shape[2] - self.input_window_size + 1)

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
        # print('START\n',x_idx_start[0:150],x_idx_start[-150:-1],'START \n')
        # print('STOP \n',x_idx_end[0:150], x_idx_end[-150:-1],'STOP \n')
        y_idx_start = np.array([sublist[0] for sublist in y_idx])
        y_idx_end = np.array([sublist[-1] for sublist in y_idx])
        # print('START\n', y_idx_start[0:150], y_idx_start[-150:-1], 'START \n')
        # print('STOP \n', y_idx_end[0:150], y_idx_end[-150:-1], 'STOP \n')
        t = 0.
        ims = []
        fig = plt.figure(figsize=(10, 6))
        grid = (1, 3)
        ax1 = plt.subplot2grid(grid, (0, 0))
        ax2 = plt.subplot2grid(grid, (0, 1))
        ax3 = plt.subplot2grid(grid, (0, 2))
        # ax3 = plt.subplot2grid(grid, (1, 0))
        # ax4 = plt.subplot2grid(grid, (1, 1))
        for ax in [ax1, ax2, ax3]:
            ax.set_axis_off()

        for i in range(0, fuel_slices.shape[0] - 1):
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
                fsin.append(fuel_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                rsin.append(r_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                gsin.append(g_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                bsin.append(b_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                asin.append(alpha_slices[idx_input, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])

                fsout.append(fuel_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                rsout.append(r_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                gsout.append(g_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                bsout.append(b_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                asout.append(alpha_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])

            fuel_subslice_in = torch.stack(fsin, dim=0)
            r_subslice_in = torch.stack(rsin, dim=0)
            g_subslice_in = torch.stack(gsin, dim=0)
            b_subslice_in = torch.stack(bsin, dim=0)
            alpha_subslice_in = torch.stack(asin, dim=0)
            data_input_subslice = torch.cat([r_subslice_in, g_subslice_in, b_subslice_in, alpha_subslice_in], dim=1)
            meta_step_in = meta_binary_slices[idx_input][0]
            meta_step_in_numeric = self.meta_tensor[idx_input][0]
            meta_fuel_initial_speed_in = meta_binary_slices[idx_input][1]
            meta_fuel_cut_off_time_in = meta_binary_slices[idx_input][2]
            meta_igni_time_in = meta_binary_slices[idx_input][3]
            meta_ignition_temp_in = meta_binary_slices[idx_input][4]

            meta_viscosity_in = meta_binary_slices[idx_input][14]
            meta_diff_in = meta_binary_slices[idx_input][15]
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
            meta_step_out = meta_binary_slices[idx_output][0]
            meta_step_out_numeric = self.meta_tensor[idx_output][0]
            meta_fuel_initial_speed_out = meta_binary_slices[idx_output][1]
            meta_fuel_cut_off_time_out = meta_binary_slices[idx_output][2]
            meta_igni_time_out = meta_binary_slices[idx_output][3]
            meta_ignition_temp_out = meta_binary_slices[idx_output][4]
            meta_viscosity_out = meta_binary_slices[idx_output][14]
            meta_diff_out = meta_binary_slices[idx_output][15]
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

            self.model.eval()

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

            t_start = time.perf_counter()
            pred_r, pred_g, pred_b, pred_a, pred_s, deppS = self.model(dataset)
            t_pred = time.perf_counter()
            t = t_pred - t_start
            print(f'Pred Time: {t * 1e6:.1f} [us]')

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

            rgb_pred_anim = ax1.imshow(prediction.astype(np.uint8) * 255, alpha=a_v_pred)
            rgb_true_anim = ax2.imshow(ground_truth.astype(np.uint8) * 255, alpha=a_v_true)

            rms = np.mean(np.sqrt(abs(prediction ** 2 - ground_truth ** 2)), axis=2)
            rms_anim = ax3.imshow(rms, cmap='RdBu', vmin=0, vmax=1)

            ims.append([rgb_pred_anim, rgb_true_anim, rms_anim, title_pred, title_true, title_rms])
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
        ani.save("flame_animation.gif")
        fig.colorbar(rms_anim, ax=ax3)
        plt.show()

    def learning_phase(self, teacher, no_frame_samples, batch_size, input_window_size, first_frame, last_frame,
                       frame_skip, criterion, optimizer ,device, learning=1,
                       num_epochs=1500):
        (self.no_frame_samples, self.batch_size, self.input_window_size, self.first_frame,
         self.last_frame, self.frame_skip) = (no_frame_samples, batch_size,
                                              input_window_size, first_frame, last_frame, frame_skip)

        criterion_model = criterion
        self.num_of_epochs = num_epochs
        model_to_Save = self.model
        if learning == 1:
            best_loss = float('inf')
            num_epochs = num_epochs
            t = 0.
            t_epoch = 0.
            grad_counter = 0
            reiterate_data = 1
            reiterate_counter = 0
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

                m_idx = torch.arange(int(self.data_input.shape[0] / 2)) # TODO : Change to random selection

                dataset = (self.data_input[m_idx], self.structure_input[m_idx], self.meta_input_h1[m_idx],
                           self.meta_input_h2[m_idx],
                           self.meta_input_h3[m_idx], self.meta_input_h4[m_idx], self.meta_input_h5[m_idx],
                           self.noise_diff_in[m_idx], self.fmot_in_binary[m_idx], self.meta_output_h1[m_idx],
                           self.meta_output_h2[m_idx], self.meta_output_h3[m_idx], self.meta_output_h4[m_idx],
                           self.meta_output_h5[m_idx], self.noise_diff_out[m_idx])

                t_start = time.perf_counter()
                self.seed_setter(int((epoch + 1) * 2))
                model_output = self.model(dataset)
                t_pred = time.perf_counter()

                loss = self.loss_calculation(self.model, m_idx, model_output, self.data_input, self.data_output,
                                             self.structure_input, self.structure_output, criterion_model, norm)

                if loss.dim() > 0:
                    lidx = torch.argmin(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss[lidx].backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    max_norm = 1.
                    nn_utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    optimizer.step()
                # if (epoch + 1) % 5 == 0:

                if self.validation_dataset is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_model_output = self.model(self.validation_dataset)
                        val_loss = self.loss_calculation(self.model, val_idx, val_model_output, self.data_input_val,
                                                         self.data_output_val, self.structure_input_val,
                                                         self.structure_output_val, criterion_model, norm)
                    self.model.train()

                self.train_loss.append(loss.item())
                self.val_loss.append(val_loss.item())

                # t_stop = time.perf_counter()
                t += (t_pred - t_start) / 4
                if epoch > 25:
                    if val_loss < min(self.val_loss[:-1]):
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
                    if reiterate_counter > 100:
                        reiterate_counter = 0
                        reiterate_data = 0
                    gloss = abs(np.sum(np.gradient(loss_recent_history)))
                    #print(gloss)
                    g_val_loss = np.sum(np.gradient(val_loss_recent_history))
                    if g_val_loss > 1e2:
                        reiterate_data = 0
                    if gloss > 1e2:
                        grad_counter = 0
                    else:
                        grad_counter += 1
                    # NOTE: lowering lr for  better performance and reset lr within conditions
                    if grad_counter == 3 or reiterate_data == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.999
                            if param_group['lr'] < 5e-6 or reiterate_data == 0:
                                param_group['lr'] = 1e-2
                                reiterate_counter = 0
                                reiterate_data = 0
                                print('optimizer -> lr back to starting point')

                        grad_counter = 0

                t_epoch_stop = time.perf_counter()
                t_epoch += (t_epoch_stop - t_epoch_start)
                if (epoch + 1) % print_every_nth_frame == 0:
                    t_epoch_total = num_epochs * t_epoch
                    t_epoch_current = epoch * t_epoch
                    print(
                        f'P: {self.period}/{self.no_of_periods} | E: {((t_epoch_total - t_epoch_current) / (print_every_nth_frame * 60)):.2f} [min], '
                        f'vL: {val_loss.item():.4f}, '
                        f'mL: {loss.item():.4f}, '
                        f'tpf: {((self.fsim.grid_size_x * self.fsim.grid_size_y) / (self.model.in_scale ** 2)) * (t * 1e3 / print_every_nth_frame / self.batch_size):.2f} [ms]')
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

    def visualize_lerning(self, poly_degree=5):
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)
        avg_train_loss = sum(self.train_loss) / len(self.train_loss)
        avg_val_loss = sum(self.val_loss) / len(self.val_loss)
        epochs = np.arange(len(self.train_loss))
        train_poly_fit = np.poly1d(np.polyfit(epochs, self.train_loss, poly_degree))
        val_poly_fit = np.poly1d(np.polyfit(epochs, self.val_loss, poly_degree))
        plt.plot(epochs, train_poly_fit(epochs), color='blue', linestyle='--',
                 label=f'Train: deg: {poly_degree} | Avg: {avg_train_loss:.3f}')
        plt.plot(epochs, val_poly_fit(epochs), color='orange', linestyle='--',
                 label=f'Val: deg: {poly_degree} | Avg: {avg_val_loss:.3f}')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def loss_calculation(self, model, idx, model_output, data_input, data_output, structure_input, structure_output,
                         criterion,
                         norm='backward'):
        pred_r, pred_g, pred_b, pred_a, pred_s, deepS = model_output

        r_in = data_input[:, 0:self.model.in_scale, :][idx]
        g_in = data_input[:, self.model.in_scale:self.model.in_scale * 2, :][idx]
        b_in = data_input[:, self.model.in_scale * 2:self.model.in_scale * 3, :][idx]
        a_in = data_input[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx]
        s_in = structure_input[idx]

        r_out = data_output[:, 0:self.model.in_scale, :][idx]
        g_out = data_output[:, self.model.in_scale:self.model.in_scale * 2, :][idx]  #.view(self.batch_size, -1)
        b_out = data_output[:, self.model.in_scale * 2:self.model.in_scale * 3, :][idx]  #.view(self.batch_size, -1)
        a_out = data_output[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx]  #.view(self.batch_size, -1)
        s_out = structure_output[idx]  #.view(self.batch_size, -1)
        t = 1 - self.fmot_in[idx]
        t_1 = self.fmot_in[idx]
        t = tt = t.unsqueeze(1)
        t_1 = tt_1 = t_1.unsqueeze(1)
        if pred_r.shape[0] != self.batch_size:
            n = int(pred_r.shape[0] / self.batch_size)
            r_in = r_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, r_in.shape[1], r_in.shape[2]).detach()
            g_in = g_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, g_in.shape[1], g_in.shape[2]).detach()
            b_in = b_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, b_in.shape[1], b_in.shape[2]).detach()
            a_in = a_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, a_in.shape[1], a_in.shape[2]).detach()
            s_in = s_in.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, s_in.shape[1], s_in.shape[2]).detach()

            r_out = r_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, r_out.shape[1], r_out.shape[2]).detach()
            g_out = g_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, g_out.shape[1], g_out.shape[2]).detach()
            b_out = b_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, b_out.shape[1], b_out.shape[2]).detach()
            a_out = a_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, a_out.shape[1], a_out.shape[2]).detach()
            s_out = s_out.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, s_out.shape[1], s_out.shape[2]).detach()
            t = t.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, t.shape[1], t.shape[1]).detach()
            t_1 = t_1.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, t_1.shape[1], t_1.shape[1]).detach()
        # Solution for learning of the dynamics in loss calculation

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
        diff_loss = torch.mean(loss_diff_r + loss_diff_g + loss_diff_b + loss_diff_a + loss_diff_s, dim=[1, 2])
        # diff_loss = loss_diff_r + loss_diff_g + loss_diff_b + loss_diff_a + loss_diff_s

        # Note: Gradient loss
        grad_r_true = torch.gradient(r_out, dim=[1])[0]
        grad_r_pred = torch.gradient(pred_r)[0]
        grad_r = criterion(t * grad_r_pred + t_1 * grad_r_true, grad_r_true)
        grad_g_true = torch.gradient(g_out, dim=[1])[0]
        grad_g_pred = torch.gradient(pred_g)[0]
        grad_g = criterion(t * grad_g_pred + t_1 * grad_g_true, grad_g_true)
        grad_b_true = torch.gradient(b_out, dim=[1])[0]
        grad_b_pred = torch.gradient(pred_b)[0]
        grad_b = criterion(t * grad_b_pred + t_1 * grad_b_true, grad_b_true)
        grad_a_true = torch.gradient(a_out, dim=[1])[0]
        grad_a_pred = torch.gradient(pred_a)[0]
        grad_a = criterion(t * grad_a_pred + t_1 * grad_a_true, grad_a_true)
        grad_s_true = torch.gradient(s_out, dim=[1])[0]
        grad_s_pred = torch.gradient(pred_s)[0]
        grad_s = criterion(t * grad_s_pred + t_1 * grad_s_true, grad_s_true)

        grad_loss = torch.mean(grad_r + grad_g + grad_b + grad_a + grad_s, dim=[1, 2])
        # grad_loss = grad_r + grad_g + grad_b + grad_a + grad_s

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
        fft_loss = torch.mean(fft_loss_r + fft_loss_g + fft_loss_b + fft_loss_a + fft_loss_s, dim=[1, 2])
        # fft_loss = fft_loss_r + fft_loss_g + fft_loss_b + fft_loss_a + fft_loss_s

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
            diff_fft_loss_r + diff_fft_loss_g + diff_fft_loss_b + diff_fft_loss_a + diff_fft_loss_s, dim=[1, 2])
        # diff_fft_loss = diff_fft_loss_r + diff_fft_loss_g + diff_fft_loss_b + diff_fft_loss_a + diff_fft_loss_s

        # Note : Exact value loss
        loss_r = criterion(t * pred_r + t_1 * r_out, r_out)
        loss_g = criterion(t * pred_g + t_1 * g_out, g_out)
        loss_b = criterion(t * pred_b + t_1 * b_out, b_out)
        loss_alpha = criterion(t * pred_a + t_1 * a_out, a_out)
        loss_s = criterion(t * pred_s + t_1 * s_out, s_out)
        value_loss = torch.mean(loss_r + loss_g + loss_b + loss_alpha + loss_s, dim=[1, 2])
        # value_loss = loss_r + loss_g + loss_b + loss_alpha + loss_s

        t = t.squeeze(1)
        t_1 = t_1.squeeze(1)
        # Solution for learning and maintaining of the proper color and other element space
        bandwidth = torch.tensor(0.1).to(self.device)  # Note: Higher value less noise (gaussian smoothing)
        bins = 100  # Note: 255 values
        r_out = torch.flatten(r_out, start_dim=1)
        pred_r = torch.flatten(pred_r, start_dim=1)
        bins_true = torch.linspace(r_out.min(), r_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_r.min().tolist(), pred_r.max().tolist(), bins).to(self.device)
        r_true_hist = kornia.enhance.histogram(r_out, bins=bins_true, bandwidth=bandwidth)
        r_pred_hist = kornia.enhance.histogram(pred_r, bins=bins_pred, bandwidth=bandwidth)
        r_hist_loss = criterion(t * r_pred_hist + t_1 * r_true_hist, r_true_hist)

        g_out = torch.flatten(g_out, start_dim=1)
        pred_g = torch.flatten(pred_g, start_dim=1)
        bins_true = torch.linspace(g_out.min(), g_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_g.min().tolist(), pred_g.max().tolist(), bins).to(self.device)
        g_true_hist = kornia.enhance.histogram(g_out, bins=bins_true, bandwidth=bandwidth)
        g_pred_hist = kornia.enhance.histogram(pred_g, bins=bins_pred, bandwidth=bandwidth)
        g_hist_loss = criterion(t * g_pred_hist + t_1 * g_true_hist, g_true_hist)

        b_out = torch.flatten(b_out, start_dim=1)
        pred_b = torch.flatten(pred_b, start_dim=1)
        bins_true = torch.linspace(b_out.min(), b_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_b.min().tolist(), pred_b.max().tolist(), bins).to(self.device)
        b_true_hist = kornia.enhance.histogram(b_out, bins=bins_true, bandwidth=bandwidth)
        b_pred_hist = kornia.enhance.histogram(pred_b, bins=bins_pred, bandwidth=bandwidth)
        b_hist_loss = criterion(t * b_pred_hist + t_1 * b_true_hist, b_true_hist)

        a_out = torch.flatten(a_out, start_dim=1)
        pred_a = torch.flatten(pred_a, start_dim=1)
        bins_true = torch.linspace(a_out.min(), a_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_a.min().tolist(), pred_a.max().tolist(), bins).to(self.device)
        a_true_hist = kornia.enhance.histogram(a_out, bins=bins_true, bandwidth=bandwidth)
        a_pred_hist = kornia.enhance.histogram(pred_a, bins=bins_pred, bandwidth=bandwidth)
        a_hist_loss = criterion(t * a_pred_hist + t_1 * a_true_hist, a_true_hist)

        s_out = torch.flatten(s_out, start_dim=1)
        pred_s = torch.flatten(pred_s, start_dim=1)
        bins_true = torch.linspace(s_out.min(), s_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_s.min().tolist(), pred_s.max().tolist(), bins).to(self.device)
        s_true_hist = kornia.enhance.histogram(s_out, bins=bins_true, bandwidth=bandwidth)
        s_pred_hist = kornia.enhance.histogram(pred_s, bins=bins_pred, bandwidth=bandwidth)
        s_hist_loss = criterion(t * s_pred_hist + t_1 * s_true_hist, s_true_hist)
        hist_loss = torch.mean(r_hist_loss + b_hist_loss + g_hist_loss + a_hist_loss + s_hist_loss, dim=1)
        # hist_loss = r_hist_loss + b_hist_loss + g_hist_loss + a_hist_loss + s_hist_loss

        # Note: Deep Supervision Loss
        rres, gres, bres, ares, sres = deepS
        dpSWeight = 0.3
        rres_target = torch.rand_like(rres) * dpSWeight
        gres_target = torch.rand_like(gres) * dpSWeight
        bres_target = torch.rand_like(bres) * dpSWeight
        ares_target = torch.rand_like(ares) * dpSWeight
        sres_target = torch.rand_like(sres) * dpSWeight
        loss_rres, loss_gres, loss_bres, loss_ares, loss_sres = (
            f.mse_loss(rres, rres_target),
            f.mse_loss(gres, gres_target),
            f.mse_loss(bres, bres_target),
            f.mse_loss(ares, ares_target),
            f.mse_loss(sres, sres_target))
        deepSLoss = torch.mean(loss_rres) + torch.mean(loss_gres) + torch.mean(loss_bres) + torch.mean(loss_ares) + torch.mean(loss_sres)
        # deepSLoss = loss_x + loss_x_mod + loss_rgbas_prod + loss_rres + loss_gres + loss_bres + loss_ares + loss_sres


        # Note: SSIM Loss
        l = self.batch_size
        indices = torch.randperm(r_out.size(0))
        selected_indices = indices[:l]
        r_out_img = r_out[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        g_out_img = g_out[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        b_out_img = b_out[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        a_out_img = a_out[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        s_out_img = s_out[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        r_pred_img = pred_r[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        g_pred_img = pred_g[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        b_pred_img = pred_b[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        a_pred_img = pred_a[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        s_pred_img = pred_s[selected_indices].view(self.batch_size, self.model.in_scale, self.model.in_scale)
        rgbas_out = torch.stack([r_out_img, g_out_img, b_out_img, a_out_img, s_out_img], dim=-1)
        rgbas_pred = torch.stack([r_pred_img, g_pred_img, b_pred_img, a_pred_img, s_pred_img], dim=-1)
        rgbas_out = torch.permute(rgbas_out, (0, 3, 1, 2))
        rgbas_pred = torch.permute(rgbas_pred, (0, 3, 1, 2))
        ssim_val = 1 - self.ssim_loss(tt.unsqueeze(2) * rgbas_out + tt_1.unsqueeze(2) * rgbas_pred, rgbas_pred).mean()

        A, B, C, D, E, F, G, H, I = 1., 1., 1., 1e2, 1e2, 1., 1., 1., 1.  # Note: loss weights for Custom
        # A, B, C, D, E, F, G, H, I = 1e1, 1e1, 3e1, 5e2, 5e2, 5e2, 1e-1, 5e-1, 5.  # Note: loss weights for MSE
        # A, B, C, D, E, F, G, H, I = 0.2, 0.2, 0.85, 2e2, 2e2, 5e1, 1e-1, 1., 5.,1.  # Note: loss weights for Sinkhorn

        loss_weights = (A, B, C, D, E, F, G, H, I)
        criterion.batch_size = value_loss.shape[0]
        gradient_penalty_loss = criterion.gradient_penalty(model)
        J = 1. / 2  #len(loss_weights)
        LOSS = (value_loss, diff_loss, grad_loss, fft_loss, diff_fft_loss, hist_loss, deepSLoss,
                ssim_val, gradient_penalty_loss)  # Attention: Aggregate all losses here

        weighted_losses = [loss * weight for loss, weight in zip(LOSS, loss_weights)]
        num_losses = len(weighted_losses)
        mse_matrix = torch.zeros((value_loss.shape[0], num_losses, num_losses)).to(self.device)
        for i in range(num_losses):
            for j in range(num_losses):
                mse = (weighted_losses[i] - weighted_losses[j]) ** 2
                mse_matrix[:, i, j] = mse

        std_between_losses = mse_matrix.std(dim=(1, 2))
        mean_between_losses = mse_matrix.mean(dim=(1, 2))
        dispersion_loss = std_between_losses / mean_between_losses
        # print(gradient_penalty_loss[0])
        LOSS = (value_loss, diff_loss, grad_loss, fft_loss, diff_fft_loss, hist_loss, deepSLoss,
                ssim_val, gradient_penalty_loss, dispersion_loss)
        loss_weights = (A, B, C, D, E, F, G, H, I, J)

        # print(A * value_loss.mean().item(), "<-value_loss: A", B * diff_loss.mean().item(),
        #       "<-diff_loss: B", C * grad_loss.mean().item(), "<-grad_loss: C", D * fft_loss.mean().item(),
        #       "<-fft_loss: D",
        #       E * diff_fft_loss.mean().item(), "<-diff_fft_loss: E", F * hist_loss.mean().item(), "<-hist_loss: F",
        #       G * deepSLoss.mean().item(), "<-deepSLoss: G",
        #       I * ssim_val.mean().item(),
        #       "<-ssim_val: I", gradient_penalty_loss.mean().item() * J, "<-gradient_penalty_loss")

        final_loss, i = 0., 0
        for losses in LOSS:
            if pred_r.shape[0] != self.batch_size:
                n = int(pred_r.shape[0] / self.batch_size)
                losses = torch.chunk(losses, n, dim=0)
                final_loss += torch.stack([loss_weights[i] * split.mean(dim=0) for split in losses])
            else:
                final_loss += loss_weights[i] * torch.mean(losses)
            i += 1
            return final_loss

    @staticmethod
    def seed_setter(seed):
        s = torch.randint(0, seed, (1,))
        torch.manual_seed(2024 + s)
        s = np.random.randint(0, seed)
        np.random.seed(2024 + s)
        s = random.randint(0, seed)
        random.seed(2024 + s)