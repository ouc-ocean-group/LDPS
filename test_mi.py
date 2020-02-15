from lib.cfg import Parameters
from lib.operators.mi_operator import MIOperator

import torch


if __name__ == "__main__":
    cfg = Parameters().parse()
    op = MIOperator(cfg)
    ckp = torch.load(cfg.ckp)
    op.model.load_state_dict(ckp)
    op.evaluate_process()
