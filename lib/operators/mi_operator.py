import torch.nn.functional as F
import random
import torch.optim as optim
import torch
import cv2
import os
import numpy as np

from lib.model.mi import MINet
from lib.dataset import make_dataloader
from lib.utils.logger import Logger
from lib.utils.metrics import Metrics
from lib.utils.lr_scheduler import PolyLR


class MIOperator(object):
    def __init__(self, cfg):
        self.cfg = cfg

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        mi_net = MINet(self.cfg).cuda()

        if cfg.sync_bn:
            mi_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mi_net)

        train_params = [
            {"params": mi_net.feats_extractor.parameters(), "lr": cfg.lr},
            {"params": mi_net.mi_module.parameters(), "lr": cfg.lr * 10},
        ]
        self.optimizer = optim.SGD(train_params, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.lr_sch = PolyLR(self.optimizer, self.cfg.iter_num)
        if cfg.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(mi_net, device_ids=[self.cfg.gpu_id])
        else:
            self.model = mi_net

        self.train_data_loader = make_dataloader(self.cfg, distributed=cfg.distributed)
        self.test_data_loader = make_dataloader(self.cfg, distributed=cfg.distributed, mode="test")

    def training_process(self):
        self.model.train()
        logger = Logger(self.cfg.iter_num, self.cfg.print_interval) if self.cfg.gpu_id == 0 else None

        for iter_step in range(self.cfg.iter_num):
            self.optimizer.zero_grad()
            imgs, priors, gts, _ = self.train_data_loader.get_batch()
            results, targets, valid_flag = self.model(imgs, priors)
            loss = F.binary_cross_entropy(results, targets, reduction="none")
            loss = (loss * valid_flag).sum() / valid_flag.sum()

            loss.backward()
            self.optimizer.step()

            if logger is not None:
                logger.step(loss, iter_step)

            if iter_step % self.cfg.eval_interval == self.cfg.eval_interval - 1:
                self.evaluate_process()
            if iter_step % self.cfg.ckp_interval == self.cfg.ckp_interval - 1 and self.cfg.gpu_id == 0:
                torch.save(self.model.module.state_dict(), "./ckp/model_{}.pth".format(iter_step))
            self.lr_sch.step()

    def evaluate_process(self):
        self.model.eval()
        with torch.no_grad():
            base_metrics = Metrics()
            mi_metrics = Metrics()

            if self.cfg.save_result:
                try:
                    os.makedirs("./outputs/{}_{}/".format(self.cfg.test_dataset, self.cfg.test_prior))
                except Exception as e:
                    print("=> ./outputs/{}_{}/ exists.".format(self.cfg.test_dataset, self.cfg.test_prior))

            for i, batch in enumerate(self.test_data_loader):
                img, prior, gt, name = batch
                img, prior, gt = img.cuda(), prior.cuda(), gt.cuda()
                base_metrics.add_batch(prior.squeeze(), gt.squeeze())

                refined_result = self.model(img, prior, "test")
                refined_result -= refined_result.min()
                refined_result /= refined_result.max() + 1e-12

                mi_metrics.add_batch(refined_result.squeeze(), gt.squeeze())

                if self.cfg.save_result:
                    refined_result = (refined_result[0].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
                    cv2.imwrite(
                        "./outputs/{}_{}/{}".format(
                            self.cfg.test_dataset, self.cfg.test_prior, name[0].replace("jpg", "png")
                        ),
                        refined_result,
                    )

            if self.cfg.distributed:
                base_metrics.all_reduce()
                mi_metrics.all_reduce()
            if self.cfg.gpu_id == 0:
                print("---------------------------------------------")
                print("Baseline        - FM: {:.4}, MAE : {:.4}".format(base_metrics.get_fm(), base_metrics.get_mae()))
                print("Refined with MI - FM: {:.4}, MAE : {:.4}".format(mi_metrics.get_fm(), mi_metrics.get_mae()))
                print("---------------------------------------------")

        self.model.train()
