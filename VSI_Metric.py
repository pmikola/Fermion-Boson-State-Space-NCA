import torch
import torch.nn.functional as F

class VS_ESSIM(torch.nn.Module):
    def __init__(self, h=0.5, L=1., K1=200.0):
        super().__init__()
        self.h = h
        self.L = L
        self.K1 = K1
        self.C = (K1 * L) ** (2 * 0.5)
        self.K1_kernel = torch.tensor([[0.0, 0.0, 3.0, 0.0, 0.0],
                                       [0.0, 10.0, 0.0, 0.0, 0.0],
                                       [3.0, 0.0, 0.0, 0.0, -3.0],
                                       [0.0, 0.0, 0.0, -10.0, 0.0],
                                       [0.0, 0.0, -3.0, 0.0, 0.0]], dtype=torch.float32)
        self.K2_kernel = torch.rot90(self.K1_kernel, k=1, dims=[0, 1])

    def forward(self, ref_img, dis_img):
        if ref_img.size(1) == 3:
            ref_img = 0.299 * ref_img[:, 0:1] + 0.587 * ref_img[:, 1:2] + 0.114 * ref_img[:, 2:3]
            dis_img = 0.299 * dis_img[:, 0:1] + 0.587 * dis_img[:, 1:2] + 0.114 * dis_img[:, 2:3]

        ref_img, dis_img = self.downsample(ref_img, dis_img)

        grad_ref = self.directional_gradient(ref_img)
        grad_dis = self.directional_gradient(dis_img)

        grad1 = torch.sqrt(torch.abs(grad_ref[..., 0] - grad_ref[..., 1]) + 1e-8)*10
        grad2 = torch.sqrt(torch.abs(grad_dis[..., 0] - grad_dis[..., 1]) + 1e-8)*10
        edge = (grad1 + grad2) / 2
        C1 = self.C * torch.exp(-edge / self.h)

        similarity_map = (2 * grad1 * grad2 + C1) / (grad1**2 + grad2**2 + C1 + 1e-8)
        weights = 1 / (1 + edge)#torch.exp(-edge)
        vsi_score = (similarity_map * weights).sum() / weights.sum()
        return vsi_score

    def downsample(self, ref_img, dis_img, min_size=256):
        _, _, H, W = ref_img.shape
        f = max(1, round(min(H, W) / min_size))
        if f > 1:
            ref_img = F.avg_pool2d(ref_img, kernel_size=f)
            dis_img = F.avg_pool2d(dis_img, kernel_size=f)
        elif min(H, W) < min_size:
            ref_img = F.interpolate(ref_img, size=(min_size, min_size), mode='bilinear', align_corners=False)
            dis_img = F.interpolate(dis_img, size=(min_size, min_size), mode='bilinear', align_corners=False)
        return ref_img, dis_img

    def directional_gradient(self, img):
        K1 = self.K1_kernel.unsqueeze(0).unsqueeze(0).to(img.device)
        K2 = self.K2_kernel.unsqueeze(0).unsqueeze(0).to(img.device)
        gradients = torch.cat([F.conv2d(img, K1, padding=2), F.conv2d(img, K2, padding=2)], dim=1)
        gradients = gradients.permute(0, 2, 3, 1)
        return gradients
