import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module.base import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(inplanes, planes, 3, stride=stride, bias=True)
        self.conv2 = ConvBlock(planes, planes, 3, 1, bias=True, relu=False)
        self.skip_conv = (
            ConvBlock(inplanes, planes, 1, stride=stride, padding=0, bias=False, relu=False)
            if stride != 1 or inplanes != planes
            else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        out = out + x
        return out


class ResidualStage(nn.Module):
    def __init__(self, inplanes, planes, num, stride=2):
        super(ResidualStage, self).__init__()
        layers = [ResidualBlock(inplanes, planes, stride)]
        for i in range(num - 1):
            layers.append(ResidualBlock(planes, planes, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DFNet(nn.Module):
    def __init__(self, layer_nums=(3, 3, 3, 1)):
        super(DFNet, self).__init__()
        self.conv_stage0 = ConvBlock(3, 32, 3, 2, bias=False)
        self.conv_stage1 = ConvBlock(32, 64, 3, 2, bias=False)
        self.conv_stage2 = ResidualStage(64, 64, layer_nums[0])
        self.conv_stage3 = ResidualStage(64, 128, layer_nums[1])
        self.conv_stage4 = ResidualStage(128, 256, layer_nums[2])
        # self.conv_stage5 = ResidualStage(256, 512, layer_nums[3], stride=1)

    def forward(self, x):
        """
        torch.Size([1, 32, 112, 112])
        torch.Size([1, 64, 56, 56])
        torch.Size([1, 64, 28, 28])
        torch.Size([1, 128, 14, 14])
        torch.Size([1, 256, 7, 7])
        torch.Size([1, 512, 7, 7])
        """
        x0 = self.conv_stage0(x)
        x1 = self.conv_stage1(x0)
        x2 = self.conv_stage2(x1)
        x3 = self.conv_stage3(x2)
        x4 = self.conv_stage4(x3)
        # x5 = self.conv_stage5(x4)

        return x0, x2, x3, x4


if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224).cuda()
    net = DFNet().cuda()
    net.eval()
    net(img)

    import torch2trt
    import time
    import numpy as np

    net = torch2trt.TensorRTModuleWrapper(net, 1, 1 << 30, param_exclude=None, verbose=False).eval()

    net_trt = net.float().cuda()
    out = net_trt(img)  # build trt engine
    times = []
    for i in range(1000):
        torch.cuda.synchronize()
        t = time.time()
        out = net_trt(img)
        torch.cuda.synchronize()
        times.append(time.time() - t)

    print("TensorRT Speed:", 1 / np.mean(times[2:]), "fps")
