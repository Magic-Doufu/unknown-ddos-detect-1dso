import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_class=10, num_feature=2, use_gpu=False, random_state=222):
        super(CenterLoss, self).__init__()
        torch.manual_seed(random_state)
        self.num_class = num_class
        self.num_feature = num_feature
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss
