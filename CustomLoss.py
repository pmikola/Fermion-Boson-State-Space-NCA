import torch
from torch import nn
from geomloss import SamplesLoss


class CustomLoss(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.loss_alpha = nn.MSELoss(reduction='none')
        # SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.device = device
        self.batch_size = torch.tensor(256).to(self.device)

    def forward(self, pred, data):
        loss = self.loss_alpha(pred, data)
        return loss

    def gradient_penalty(self, model, lambda_gp=1, epsilon=1e-9, epsilon_large=1e-1, epsilon_small=1e-5):
        grad_norms_large = []
        grad_norms_small = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2)
                if grad_norm < epsilon_small:
                    grad_norms_small.append((epsilon_small - grad_norm) ** 2)
                if grad_norm > epsilon_large:
                    grad_norms_large.append((grad_norm - epsilon_large) ** 2)
        if grad_norms_large or grad_norms_small:
            loss_large = torch.sqrt(
                torch.sum(torch.stack(grad_norms_large)) + epsilon) if grad_norms_large else torch.tensor(0.0,device=self.device)
            loss_small = torch.sqrt(
                torch.sum(torch.stack(grad_norms_small)) + epsilon) if grad_norms_small else torch.tensor(0.0,device=self.device)
            loss = loss_large + loss_small
            return lambda_gp * loss.unsqueeze(0).repeat(self.batch_size)
        else:
            return torch.tensor([0.0], requires_grad=True).to(self.device).repeat(self.batch_size)