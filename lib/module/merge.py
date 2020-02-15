import torch
import torch.nn as nn
import torch.nn.functional as F


class MergeModule(nn.Module):
    def __init__(self):
        super(MergeModule, self).__init__()

    def forward(self, xs):
        x3 = F.interpolate(xs[3], size=xs[0].size()[-2:], mode="bilinear", align_corners=True)
        x2 = F.interpolate(xs[2], size=xs[0].size()[-2:], mode="bilinear", align_corners=True)
        x1 = F.interpolate(xs[1], size=xs[0].size()[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x3, x2, x1, xs[0]], dim=1)
        return x
