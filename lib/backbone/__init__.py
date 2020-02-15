import torch
from .mobile_net import MobileNetV2
from .vgg import VGG
from .dfnet import DFNet


def make_backbone(name):
    if name == "mobilenetv2":
        net = MobileNetV2()
        net.load_state_dict(torch.load("./ckp/mobilenetv2.pth", map_location="cpu"), strict=False)
        return net, 472
    elif name == "vgg":
        return VGG(), 1408
    elif name == "dfnet":
        net = DFNet()
        net.load_state_dict(torch.load("./ckp/dfnet.pth", map_location="cpu"), strict=False)
        return net, 480
    else:
        raise NotImplementedError
