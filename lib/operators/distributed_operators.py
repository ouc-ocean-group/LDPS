import torch
import torch.multiprocessing as mp
import torch.distributed as dist


class DistributedOperator(object):
    def __init__(self, cfg, operator):
        self.cfg = cfg
        self.operator = operator

    def setup(self):
        try:
            ngpus_per_node = torch.cuda.device_count()
            self.cfg.ngpus_per_node = ngpus_per_node
            self.cfg.world_size = ngpus_per_node * self.cfg.world_size
        except ValueError:
            raise ValueError("Can not get gpu numbers!")

    def init_operator(self, gpu, ngpus_per_node, cfg):
        cfg.gpu_id = gpu
        if cfg.gpu_id is not None:
            print("=> Use GPU: {}".format(gpu))

        # ========= init distributed process group ========== #
        cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend="nccl", init_method=cfg.dist_url, world_size=cfg.world_size, rank=cfg.rank)
        torch.cuda.set_device(gpu)
        # ========= init base operator ========== #
        return self.operator(cfg)

    def train(self):
        self.setup()
        mp.spawn(self.dist_training_process, nprocs=self.cfg.ngpus_per_node, args=(self.cfg.ngpus_per_node, self.cfg))

    def dist_training_process(self, gpu, ngpus_per_node, cfg):
        operator = self.init_operator(gpu, ngpus_per_node, cfg)
        # ========= start training ========== #
        operator.training_process()
