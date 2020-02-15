import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, relu=True
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu is not None:
            x = self.relu(x)
        return x


class ASPPBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPPBlock, self).__init__()
        self.down_conv = nn.Sequential(
            ConvBlock(inplanes, inplanes, 1, padding=0, bias=False),
            ConvBlock(inplanes, planes, 3, padding=1, bias=False),
            ConvBlock(planes, planes, 1, padding=0, bias=False),
        )
        self.conv2 = ConvBlock(planes, planes, 1, 1, padding=0, dilation=1, bias=False)
        self.conv4 = ConvBlock(planes, planes, 3, 1, padding=6, dilation=6, bias=False)
        self.conv8 = ConvBlock(planes, planes, 3, 1, padding=12, dilation=12, bias=False)
        self.conv16 = ConvBlock(planes, planes, 3, 1, padding=18, dilation=18, bias=False)
        self.out_conv = ConvBlock(4 * planes, planes, 1, padding=0, bias=False)

    def forward(self, x):
        x = self.down_conv(x)
        x2 = self.conv2(x)
        x4 = self.conv4(x)
        x8 = self.conv8(x)
        x16 = self.conv16(x)
        x = torch.cat((x2, x4, x8, x16), dim=1)
        x = self.out_conv(x)
        return x


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
