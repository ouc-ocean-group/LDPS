from lib.cfg import Parameters
from lib.operators.distributed_operators import DistributedOperator
from lib.operators.mi_operator import MIOperator


if __name__ == "__main__":
    cfg = Parameters().parse()
    op = DistributedOperator(cfg, MIOperator)
    op.train()
