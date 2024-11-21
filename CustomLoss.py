import torch
from torch import nn
from geomloss import SamplesLoss


class CustomLoss(nn.Module):
    def __init__(self,device):
        super().__init__()
        #self.loss_alpha = nn.MSELoss(reduction='none')
        self.loss_alpha = nn.HuberLoss(reduction='none',delta=0.5)
        # SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.device = device
        self.batch_size = torch.tensor(256).to(self.device)

    def forward(self, pred, data):
        loss = self.loss_alpha(pred, data)
        return loss

    def gradient_penalty(self, model, lambda_gp=1e-3, lambda_wp=1e-2, epsilon=1e-9, weight_threshold=2e0, epsilon_large=2e0, epsilon_small=5e-3):
        grad_norms_large = []
        grad_norms_small = []
        weight_penalty = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2)
                if grad_norm < epsilon_small:
                    grad_norms_small.append((epsilon_small - grad_norm) ** 2)
                if grad_norm > epsilon_large:
                    grad_norms_large.append((grad_norm - epsilon_large) ** 2)
            weight_exceed = torch.clamp(param.abs() - weight_threshold, min=0)
            if weight_exceed.numel() > 0:
                weight_penalty.append(torch.sum(weight_exceed ** 2))
        if grad_norms_large or grad_norms_small:
            loss_large = torch.sqrt(
                torch.sum(torch.stack(grad_norms_large)) + epsilon) if grad_norms_large else torch.tensor(0.0,device=self.device)
            loss_small = torch.sqrt(
                torch.sum(torch.stack(grad_norms_small)) + epsilon) if grad_norms_small else torch.tensor(0.0,device=self.device)
            grad_loss = lambda_gp * (loss_large + loss_small)
        else:
            grad_loss = torch.tensor(0.0, device=self.device)
        if weight_penalty:
            weight_loss = lambda_wp * torch.sqrt(torch.sum(torch.stack(weight_penalty)) + epsilon)
        else:
            weight_loss = torch.tensor(0.0, device=self.device)
        total_loss = grad_loss + weight_loss
        return total_loss