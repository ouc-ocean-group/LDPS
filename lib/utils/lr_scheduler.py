import torch.optim.lr_scheduler as lr_s


class PolyLR(lr_s.LambdaLR):
    def __init__(self, optimizer, iter_num, power=0.9, last_epoch=-1):
        self.power = power
        self.iter_num = iter_num
        poly_lambda = lambda epoch: (1 - epoch / self.iter_num) ** self.power
        super(PolyLR, self).__init__(optimizer, poly_lambda, last_epoch)
