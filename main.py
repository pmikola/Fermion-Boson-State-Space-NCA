import os
import random
import sys
from os import path
import numpy as np
import torch
from torch import nn


from teacher import teacher
from flameEngine import flame as fl
from CustomLoss import CustomLoss

# Start ................
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.autograd.set_detect_anomaly(True) # Note : Tremendously slowing down program - Attention: Be careful!

no_frame_samples = 50
batch_size = 256
input_window_size = 7

no_frames = 1000
first_frame, last_frame, frame_skip = 0, no_frames, 10

model -= HyperRadialNeuralCelularAutomata()

t = teacher(model, device)
t.fsim = fl.flame_sim(no_frames=no_frames, frame_skip=frame_skip)

OT_backend = 'tensorized'
criterion = CustomLoss(device)

optimizer = torch.optim.Adam(t.model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)

# torch.autograd.set_detect_anomaly(True)
# Note: Eon > Era > Period > Epoch
no_periods = 1
t.no_of_periods = no_periods
# model.load_state_dict(torch.load('model.pt'Ś))
for period in range(1, no_periods + 1):
    t.period = period
    t.fsim = fl.flame_sim(no_frames=1000, frame_skip=frame_skip)
    t.fsim.igni_time = no_frames
    # t.generate_structure()
    t.fsim.fuel_dens_modifier = 1 / t.fsim.dt
    t.fsim.simulate(simulate=0, save_rgb=1, save_alpha=1, save_fuel=1, delete_data=0)
    t.learning_phase(t, no_frame_samples, batch_size, input_window_size, first_frame,
                     last_frame, frame_skip * 2, criterion, optimizer ,device, learning=1,
                     num_epochs=1000)
    # t.fsim.simulate(simulate=0,delete_data=1)

t.visualize_lerning()
t.examine(criterion, device, plot=1)