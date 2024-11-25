import os
import random
import sys
from os import path
import numpy as np
import torch
import importlib.util
from torch import nn
from performer_pytorch import Performer
from model import Fermionic_Bosonic_Space_State_NCA
from teacher import teacher
from flameEngine import flame as fl
from CustomLoss import CustomLoss
import warnings
from discriminator import discriminator
import torchopt
# from igt.torch_igt import IGTransporter
# Start ................
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Computational Environment used : ",device)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.elastic")

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.autograd.set_detect_anomaly(True) # Note : Tremendously slowing down program - Attention: Be careful!

no_frame_samples = 50
batch_size = 256
input_window_size = 7

no_frames = 1000
first_frame, last_frame, frame_skip = 0, no_frames, 10
hdc_dim = 5
rbf_probes_number = 5
nca_steps = 5
learning = 1

model = Fermionic_Bosonic_Space_State_NCA(batch_size,no_frame_samples, input_window_size,hdc_dim,rbf_probes_number,nca_steps, device).to(device)
no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters in trained architecture :", round(no_params*1e-6,2),' [M]')

# torch.save(model.state_dict(), 'model.pt')
if learning == 0:
    discriminator = None
    t = teacher(model, discriminator, device)
    disc_optimizer = None
else:
    discriminator = discriminator(no_frame_samples, batch_size, input_window_size, device).to(device)
    t = teacher(model, discriminator, device)
    disc_optimizer = torch.optim.Adam(t.discriminator.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=1e-6, amsgrad=True)

t.seed_setter(2024)
t.fsim = fl.flame_sim(no_frames=no_frames, frame_skip=frame_skip)

criterion = CustomLoss(device)
criterion_disc = nn.BCELoss(reduction='mean')
# optimizer = torchopt.Adam(t.model.parameters(), lr=5e-3) # High level api
# optim = torchopt.Optimizer(net.parameters(), torchopt.adam(lr=learning_rate)) # low level api
# optimizer = torch.optim.SGD(t.model.parameters(), lr=5e-3, momentum=0.1, weight_decay=1e-4)
optimizer = torch.optim.Adam(t.model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True)

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
                     last_frame, frame_skip*2, criterion, optimizer,criterion_disc, disc_optimizer ,device, learning=learning,
                     num_epochs=1000)
    # t.fsim.simulate(simulate=0,delete_data=1)+

t.visualize_lerning(5)
t.examine(criterion, device, plot=1)