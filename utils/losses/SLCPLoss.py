import torch
import torch.nn as nn
import torch.nn.functional as f
from utils.losses.Dist import Dist


class SLCPLoss(nn.CrossEntropyLoss):
    def __init__(self, num_class=6, num_feature=6, use_gpu=True, weight = 1.0):
        super(SLCPLoss, self).__init__()
        self.use_gpu = use_gpu
        self.weight_pl = float(weight)
        self.Dist = Dist(num_classes=num_class, feat_dim=num_feature)
        self.points = self.Dist.centers

    def forward(self, x, y, labels=None):
        dist_l2_p = self.Dist(x, center=self.points)
        logits = f.softmax(-dist_l2_p, dim=1)

        if labels is None:
            return logits, 0
        loss_main = f.cross_entropy(-dist_l2_p, labels)
        
        center_batch = self.points[labels, :]
        loss_r = f.mse_loss(x, center_batch) / 2

        o_center = self.points.mean(0)
        l_ = (self.points - o_center).pow(2).mean(1)
        # loss_outer = torch.exp(-l_.mean(0))
        loss_outer_std = torch.std(l_)

        loss = loss_main + self.weight_pl * loss_r + loss_outer_std
        return logits, loss