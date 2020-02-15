import torch.nn as nn
import torch
import torch.nn.functional as F
from .base import ConvBlock, ASPPBlock, ResidualStage


class MIModule(nn.Module):
    def __init__(self, cfg, in_c=472):
        super(MIModule, self).__init__()
        self.t_fg = cfg.t_fg
        self.t_afg = cfg.t_afg
        self.t_abg = cfg.t_abg

        self.aspp = ASPPBlock(in_c, 32)
        self.all_transformer = ResidualStage(32, 32, 3, stride=1)
        self.certain_transformer = ResidualStage(32, 32, 3, stride=1)
        self.discriminator = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64), nn.PReLU(), nn.Conv2d(64, 1, 1, 1), nn.Sigmoid()
        )

    def forward(self, feats, priors):
        feats = self.aspp(feats)
        certain_feats = self.certain_transformer(feats)
        feats = self.all_transformer(feats)

        feats = F.interpolate(feats, size=priors.size()[-2:], mode="bilinear", align_corners=True)
        certain_feats = F.interpolate(certain_feats, size=priors.size()[-2:], mode="bilinear", align_corners=True)

        certain_flag = torch.ge(priors, self.t_fg).float().detach()
        certain_priors = priors * certain_flag
        certain_priors = certain_priors / certain_priors.sum(2, keepdim=True).sum(3, keepdim=True)
        certain_feats = (certain_feats * certain_priors).sum(2).sum(2)
        certain_feats = certain_feats.unsqueeze(2).unsqueeze(2).repeat(1, 1, feats.size(2), feats.size(3))

        if torch.rand(1) > 0.5:
            pairs = torch.cat((feats, certain_feats), dim=1)
        else:
            pairs = torch.cat((certain_feats, feats), dim=1)

        results = self.discriminator(pairs)

        pos_flag = torch.ge(priors, self.t_afg)
        neg_flag = torch.le(priors, self.t_abg)

        targets = torch.zeros_like(results)
        targets.masked_fill_(pos_flag, 1)

        valid_flag = (pos_flag + neg_flag).float()

        return results, targets, valid_flag

    @staticmethod
    def sample_k(flag, k=5000):
        assert flag.size(0) == 1
        flag_h, flag_w = flag.size()[-2:]
        flatten_flag = flag.view(-1)
        idx = torch.nonzero(flatten_flag)
        if idx.size(0) > k:
            select_idx = torch.randint(high=idx.size(0), size=(k,), device=flag.device)
            idx = idx[select_idx]
            flag = torch.zeros_like(flatten_flag)
            flag[idx] = 1
            flag = flag.view(1, 1, flag_h, flag_w)
        return flag


if __name__ == "__main__":
    torch.random.manual_seed(1)
    imgs = torch.randn(2, 472, 32, 32)
    priors = torch.randn(2, 1, 32, 32).sigmoid()

    net = MIModule(None)
    a, b = net(imgs, priors)
    print(a.size())
    print(b.size())
