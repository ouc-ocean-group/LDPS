import torch.nn as nn
import torchvision as tv


class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        if pretrained is True:
            self.copy_params_from_vgg16()

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        f1 = self.relu1_2(self.conv1_2(h))
        h = self.pool1(f1)
        h = self.relu2_1(self.conv2_1(h))
        f2 = self.relu2_2(self.conv2_2(h))
        h = self.pool2(f2)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        f3 = self.relu3_3(self.conv3_3(h))
        h = self.pool3(f3)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        f4 = self.relu4_3(self.conv4_3(h))
        h = self.pool4(f4)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        f5 = self.relu5_3(self.conv5_3(h))
        return f2, f3, f4, f5

    def copy_params_from_vgg16(self):
        features = [
            self.conv1_1,
            self.relu1_1,
            self.conv1_2,
            self.relu1_2,
            self.pool1,
            self.conv2_1,
            self.relu2_1,
            self.conv2_2,
            self.relu2_2,
            self.pool2,
            self.conv3_1,
            self.relu3_1,
            self.conv3_2,
            self.relu3_2,
            self.conv3_3,
            self.relu3_3,
            self.pool3,
            self.conv4_1,
            self.relu4_1,
            self.conv4_2,
            self.relu4_2,
            self.conv4_3,
            self.relu4_3,
            self.pool4,
            self.conv5_1,
            self.relu5_1,
            self.conv5_2,
            self.relu5_2,
            self.conv5_3,
        ]

        vgg16 = tv.models.vgg16(pretrained=True)
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
