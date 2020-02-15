import torch
import numpy as np
import torch.distributed as dist


class Metrics(object):
    def __init__(self):
        super(Metrics, self).__init__()
        self.total_fm = torch.zeros(1).cuda()
        self.total_mae = torch.zeros(1).cuda()
        self.num = torch.zeros(1).cuda()

    @staticmethod
    def fmeasure(predict, gt, th=None):
        predict = predict.squeeze().detach().cpu().numpy()
        gt = gt.squeeze().detach().cpu().numpy()
        eps = np.finfo(float).eps
        gt[gt != 0] = 1
        # threshold fm
        binary = np.zeros(predict.shape)
        th = 2 * predict.mean() if th is None else th
        if th > 1:
            th = 0.999
        binary[predict >= th] = 1
        sb = (binary * gt).sum()
        pre = sb / (binary.sum() + eps)
        rec = sb / (gt.sum() + eps)
        thfm = 1.3 * pre * rec / (0.3 * pre + rec + eps)
        return thfm, pre, rec

    @staticmethod
    def mae(predict, gt):
        predict = predict.to(gt.device)
        gt = (gt != 0).float()
        mae = torch.abs(predict - gt).mean()
        return mae.item()

    def add_batch(self, pred, label):
        assert pred.shape == label.shape
        fm, mae = self.fmeasure(pred, label), self.mae(pred, label)

        self.total_fm += fm[0]
        self.total_mae += mae
        self.num += 1

    def all_reduce(self):
        dist.all_reduce(self.total_fm, dist.ReduceOp.SUM)
        dist.all_reduce(self.total_mae, dist.ReduceOp.SUM)
        dist.all_reduce(self.num, dist.ReduceOp.SUM)

    def get_fm(self):
        return (self.total_fm / self.num).item()

    def get_mae(self):
        return (self.total_mae / self.num).item()
