import time

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn

# Note : https://ieeexplore.ieee.org/document/6873260
class VS_ESSIM(torch.nn.Module):
    def __init__(self, device,h=0.5, L=3., K1=200.):
        super().__init__()
        self.device = device
        self.h = h
        self.L = L
        self.K1 = K1
        self.C = (self.K1 * L) ** (2 * 0.5)
        # Note : Scharr filters
        self.K1_kernel = torch.tensor([[-3, -10, -3],
                                       [0, 0, 0],
                                       [3, 10, 3]], dtype=torch.float32).to(self.device)

        self.K2_kernel = torch.tensor([[-3, 0, 3],
                                       [-10, 0, 10],
                                       [-3, 0, 3]], dtype=torch.float32).to(self.device)
        self.HVS_matrix = torch.tensor([[0.06, 0.63, 0.27],
                                       [0.3,0.04,-0.35],
                                       [0.34, -0.6, 0.17]], dtype=torch.float32).to(self.device)

    def forward(self, ref_img, dis_img,make_it_gray=False):
        if make_it_gray:
            ref_img = 0.299 * ref_img[:, 0:1] + 0.587 * ref_img[:, 1:2] + 0.114 * ref_img[:, 2:3]
            dis_img = 0.299 * dis_img[:, 0:1] + 0.587 * dis_img[:, 1:2] + 0.114 * dis_img[:, 2:3]


        #ref_img, dis_img = self.resample(ref_img, dis_img)
        grad_ref_lmn = self.directional_gradient(ref_img)
        grad_dis_lmn = self.directional_gradient(dis_img)
        ref_img = ref_img.permute(0, 2, 3, 1)
        ref_img = torch.matmul(ref_img, self.HVS_matrix)
        ref_img = ref_img.permute(0, 3, 1, 2)
        dis_img = dis_img.permute(0, 2, 3, 1)
        dis_img = torch.matmul(dis_img, self.HVS_matrix)
        dis_img = dis_img.permute(0, 3, 1, 2)

        grad_ref_hvs = self.directional_gradient(ref_img)
        grad_dis_hvs = self.directional_gradient(dis_img)

        grad1_lmn = grad_ref_lmn[..., 0] - grad_ref_lmn[..., 1] + 1e-8
        grad2_lmn = grad_dis_lmn[..., 0] - grad_dis_lmn[..., 1] + 1e-8
        grad1_hvs = grad_ref_hvs[..., 0] - grad_ref_hvs[..., 1] + 1e-8
        grad2_hvs = grad_dis_hvs[..., 0] - grad_dis_hvs[..., 1] + 1e-8


        #edge_normalized = edge / edge.max()
        C1,C2,C3 = 1.,1.,1.#self.C * torch.exp(-edge / self.h)
        alpha,beta = 2.,2.
        similarity_map_lmn = (2 * grad1_lmn * grad2_lmn + C1) / (grad1_lmn**2 + grad2_lmn**2 + C1 + 1e-8)
        similarity_map_hvs = (2 * grad1_hvs * grad2_hvs + C2) / (grad1_hvs**2 + grad2_hvs**2 + C2 + 1e-8)

        S = (torch.abs(similarity_map_lmn)**alpha) ** (torch.abs(similarity_map_hvs)**beta)
        vsi_score = S.mean()
        return vsi_score

    def resample(self, ref_img, dis_img, target_size=256):
        _, _, H, W = ref_img.shape
        # f = max(1, round(min(H, W) / target_size))
        # if f > 1:
        #     ref_img = F.avg_pool2d(ref_img, kernel_size=f)
        #     dis_img = F.avg_pool2d(dis_img, kernel_size=f)
        # elif min(H, W) < target_size:
        ref_img = F.interpolate(ref_img, size=(target_size, target_size), mode='bilinear', align_corners=False)
        dis_img = F.interpolate(dis_img, size=(target_size, target_size), mode='bilinear', align_corners=False)
        return ref_img, dis_img

    def directional_gradient(self, img):
        K1 = self.K1_kernel.unsqueeze(0).unsqueeze(0).expand(1, img.shape[1], 3, 3)
        K2 = self.K2_kernel.unsqueeze(0).unsqueeze(0).expand(1, img.shape[1], 3, 3)
        gradients = torch.cat([F.conv2d(img, K1, padding=2), F.conv2d(img, K2, padding=2)], dim=1)
        gradients = gradients.permute(0, 2, 3, 1)
        return gradients
