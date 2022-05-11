import torch
import torch.nn as nn


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class AngleInference(nn.Module):
    def __init__(self, width_factor=1, img_size=112):
        super(AngleInference, self).__init__()

        self.conv1 = nn.Conv2d(3, int(16 * width_factor), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16 * width_factor))
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(int(16 * width_factor), int(64 * width_factor), kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(int(64 * width_factor))
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(int(16 * width_factor), int(16 * width_factor), 2, False, 2)
        self.block3_2 = InvertedResidual(int(16 * width_factor), int(16 * width_factor), 1, True, 2)
        self.block3_3 = InvertedResidual(int(16 * width_factor), int(16 * width_factor), 1, True, 2)
        self.block3_4 = InvertedResidual(int(16 * width_factor), int(16 * width_factor), 1, True, 2)
        self.block3_5 = InvertedResidual(int(16 * width_factor), int(16 * width_factor), 1, True, 2)

        self.conv4_1 = InvertedResidual(int(16 * width_factor), int(32 * width_factor), 2, False, 2)

        self.conv5_1 = InvertedResidual(int(32 * width_factor), int(32 * width_factor), 1, False, 4)
        self.block5_2 = InvertedResidual(int(32 * width_factor), int(32 * width_factor), 1, True, 4)
        self.block5_3 = InvertedResidual(int(32 * width_factor), int(32 * width_factor), 1, True, 4)
        self.block5_4 = InvertedResidual(int(32 * width_factor), int(32 * width_factor), 1, True, 4)
        self.block5_5 = InvertedResidual(int(32 * width_factor), int(32 * width_factor), 1, True, 4)
        self.block5_6 = InvertedResidual(int(32 * width_factor), int(32 * width_factor), 1, True, 4)

        self.conv6 = conv_bn(int(32 * width_factor), int(8 * width_factor), 3, 2)
        self.conv7 = conv_bn(int(8 * width_factor), int(32 * width_factor), img_size // 16, 1)
        self.avg_pool1 = nn.AvgPool2d(img_size // 8)
        self.avg_pool2 = nn.AvgPool2d(img_size // 8)
        self.avg_pool3 = nn.AvgPool2d(3)

        self.fc1 = nn.Linear(int((32 + 32 + 32) * width_factor), int(16 * width_factor))
        self.fc2 = nn.Linear(int(16 * width_factor), 6)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)

        x = self.conv4_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        feature = self.block5_6(x)
        x2 = self.avg_pool2(feature)
        x2 = x2.view(x2.size(0), -1)
        x = self.conv6(feature)
        x = self.conv7(x)
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)
        multi_scale = torch.cat([x1, x2, x3], 1)
        x = self.fc1(multi_scale)
        mat6 = self.fc2(x)
        return mat6, feature


class LandmarkNet(nn.Module):
    def __init__(self, width_factor=1, img_size=112, landmarks_size=68):
        super(LandmarkNet, self).__init__()
        self.conv1 = InvertedResidual(int(32 * width_factor), int(4 * width_factor), 1, False, 2)
        self.conv2 = conv_bn(int(4 * width_factor), int(4 * width_factor), 3, 2)
        self.conv3 = nn.Conv2d(int(4 * width_factor), int(32 * width_factor), img_size // 16, 1, 0)
        self.bn4 = nn.BatchNorm2d(int(32 * width_factor))
        self.avg_pool1 = nn.AvgPool2d(img_size // 8)
        self.avg_pool2 = nn.AvgPool2d(img_size // 16)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(int((16 + 32 + 128) * width_factor), landmarks_size * 2)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv2(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv3(x))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
        return landmarks
