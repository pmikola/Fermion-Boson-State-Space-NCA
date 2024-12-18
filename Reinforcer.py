import random
import time
from collections import deque

import torch
from linformer import Linformer
from torch import nn


class Reinforcer(nn.Module):
    def __init__(self, batch_size, channels, num_steps, memory_size, device):
        super(Reinforcer, self).__init__()
        self.last_frame = None
        self.device = device
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.channels = channels
        self.num_steps = num_steps
        self.memory = deque(maxlen=self.memory_size)
        self.reward = torch.tensor([1], device=self.device)
        self.gamma = torch.tensor([0.1], device=self.device)

        self.p_space = nn.Conv1d(in_channels=self.channels*2,out_channels=self.channels*2,kernel_size=1)
        self.p_quality = nn.Conv1d(in_channels=2,out_channels=self.channels*2,kernel_size=1)
        self.p_iter = nn.Conv1d(in_channels=1,out_channels=self.channels*2,kernel_size=1)
        self.p_qi = nn.Conv1d(in_channels=2*self.channels*2,out_channels=1,kernel_size=1)
        self.downlift_s0 = nn.Conv1d(in_channels=self.channels*2,out_channels=1,kernel_size=1)
        self.downlift_s1 = nn.Conv1d(in_channels=700,out_channels=1,kernel_size=1)
        self.p_state2action = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.act = nn.ELU(alpha=1.)
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
        print("Weight initialization in Reinforcer complete ")

    def remember(self,state,action):
        state = [s.detach() for s in state]
        action = action.detach()
        self.memory.append([state,action,self.reward.detach()])

    def calculate_reward(self,pred,true):
        t_check = (torch.abs(pred - true) <= 1e-6)
        self.reward = t_check.sum().view(1).detach()

    def sample_memory(self, sample_size):
        if len(self.memory) < sample_size:
            return None
        return random.sample(self.memory, sample_size)

    def compute_returns(self, rewards, states):
        discounted_rewards_s = torch.zeros_like(states[0][0], device=self.device)
        discounted_rewards_q = torch.zeros_like(states[0][1], device=self.device)
        discounted_rewards_i = torch.zeros_like(states[0][2], device=self.device)

        for t in range(len(rewards)-1):
            next_state_value_p = states[t + 1][0]
            next_state_value_q = states[t + 1][1]
            next_state_value_i = states[t + 1][2]
            running_reward_s = rewards[t].unsqueeze(1).unsqueeze(2) + self.gamma.unsqueeze(1).unsqueeze(2) * next_state_value_p
            running_reward_q = rewards[t].unsqueeze(1).unsqueeze(2) + self.gamma.unsqueeze(1).unsqueeze(2) * next_state_value_q
            running_reward_i = rewards[t].unsqueeze(1).unsqueeze(2) + self.gamma.unsqueeze(1).unsqueeze(2) * next_state_value_i
            discounted_rewards_s += running_reward_s
            discounted_rewards_q += running_reward_q
            discounted_rewards_i += running_reward_i
        return discounted_rewards_s,discounted_rewards_q,discounted_rewards_i

    def replay_memory(self):
        states  = [t[0] for t in self.memory]
        actions = [t[1] for t in self.memory]
        rewards = [t[2] for t in self.memory]
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        discounted_rewards_s,discounted_rewards_q,discounted_rewards_i = self.compute_returns(rewards,states)
        self.optimizer.zero_grad()
        loss = 0.0
        for i in range(len(self.memory)):
            particles_space, particles_quality, iteration = states[i]
            Q_next = self.forward((particles_space, particles_quality, iteration))
            step_loss = self.loss_fn(Q_next, actions[i]) + (Q_next - discounted_rewards_s).pow(2).mean()
            loss += torch.exp(-step_loss*(1e-7))
        #print(loss)
        loss.backward()
        self.optimizer.step()


    def forward(self, state):
        particles_space, particles_quality,i = state
        p_s = self.act(self.p_space(particles_space))
        p_q = self.act(self.p_quality(particles_quality))
        p_i =self.act( self.p_iter(i))
        p_trans_qi = torch.cat([ p_q,p_i], dim=1)
        p_qi = self.act(self.p_qi(p_trans_qi))
        p_trans = p_s + p_qi
        downlifted_s0 = self.act(self.downlift_s0(p_trans).squeeze().unsqueeze(2))
        downlifted_s1 = self.downlift_s1(downlifted_s0)
        action_probs = torch.softmax(self.p_state2action(downlifted_s1),dim=0)
        return action_probs

