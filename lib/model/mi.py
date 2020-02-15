import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.backbone import make_backbone
from lib.module.mi_module import MIModule
from lib.module.merge import MergeModule


class MINet(nn.Module):
    def __init__(self, cfg):
        super(MINet, self).__init__()

        self.feats_extractor, in_c = make_backbone(cfg.backbone)
        self.mi_module = MIModule(cfg, in_c=in_c)
        self.merge = MergeModule()

    def extract_feats(self, imgs):
        feats = self.feats_extractor(imgs)
        feats = self.merge(feats)
        return feats

    def forward(self, imgs, priors, mode="train"):
        feats = self.extract_feats(imgs)
        results, targets, valid_flag = self.mi_module(feats, priors)
        if mode == "train":
            return results, targets, valid_flag
        else:
            assert imgs.size(0) == 1
            results = F.interpolate(results, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
            refined_results = self.fuse_result(results, priors)
            return refined_results

    @staticmethod
    def fuse_result(result, prior):
        size1 = result.size(2)
        size2 = result.size(3)

        result = result.view(-1)
        prior = prior.view(-1)
        result_confidential = 9 * (result - 0.5).pow(2)
        prior_confidential = 9 * (prior - 0.5).pow(2)

        prior_confidential_idx = prior_confidential >= result_confidential
        result_confidential_idx = prior_confidential < result_confidential

        new_result = result_confidential_idx.float() * result
        new_prior = prior_confidential_idx.float() * prior

        new_result += new_prior

        new_result = new_result.view(1, 1, size1, size2)
        return new_result
