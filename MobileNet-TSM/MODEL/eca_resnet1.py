import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
from MODEL.eca_ns import eca_layer
from MODEL.ECAAttention import ECAAttention
from MODEL.SEAttention import SEAttention
from MODEL.SKAttention import SKAttention
from MODEL.CBAM import CBAMBlock
# from MODEL.BAM import BAMBlock
# from MODEL.BAMATT import BAM

import math
import torch
class FCABlock(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel, reduction=16, dct_weight=None):
        super(FCABlock, self).__init__()
        mid_channel = channel // reduction
        self.dct_weight = dct_weight
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.sum(x*self.dct_weight, dim=[2,3])
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        # self.ECA = ECAAttention(planes * 4)
        # self.SE = SEAttention(planes * 4)
        # self.CBAM = CBAMBlock(planes * 4)
        # self.BAM = BAM(planes * 4)


        # self.planes = planes
        # if self.planes == 256 or self.planes == 512:
            # self.eca = FCABlock(planes * 4)
            # self.eca = eca_layer(planes * 4, k_size)
            # self.ECA = ECAAttention(planes * 4)
            # self.SE = SEAttention(planes * 4, reduction=16)
            # self.SK = SKAttention(planes * 4, reduction=16, group=1, L=32)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.ECA(out)
        # out = self.SE(out)
        # out = self.CBAM(out)
        # out = self.BAM(out)

        # if self.planes == 256 or self.planes == 512:
            # out = self.eca(out)
            # out = self.ECA(out)
            # out = self.SE(out)
            # out = self.SK(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class ResNet1(nn.Module):

    def __init__(self, block, layers, num_classes=2, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # freeze(self.conv1)
        self.bn1 = nn.BatchNorm2d(64)
        # freeze(self.bn1)
        self.relu = nn.ReLU(inplace=True)
        # freeze(self.relu)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # freeze(self.maxpool)

        for param in self.parameters():
            param.requires_grad = False

        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        freeze(self.layer1)

        # self.SE1 = SEAttention(256)
        self.CBAM1 = CBAMBlock(256)
        # self.ECA1 = ECAAttention(256)

        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        freeze(self.layer2)

        # self.SE2 = SEAttention(512)
        self.CBAM2 = CBAMBlock(512)
        # self.ECA2 = ECAAttention(512)

        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        freeze(self.layer3)

        # self.SE3 = SEAttention(1024)
        self.CBAM3 = CBAMBlock(1024)
        # self.ECA3 = ECAAttention(1024)

        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        freeze(self.layer4)

        # self.SE4 = SEAttention(2048)
        self.CBAM4 = CBAMBlock(2048)
        # self.ECA4 = ECAAttention(2048)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for param in self.parameters():
        #     param.requires_grad = False





        # self.ECA = ECAAttention(planes * 4)
        # self.BAM = BAM(planes * 4)
        # self.CBAM = CBAMBlock(planes * 4)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # x = self.SE1(x)
        x = self.CBAM1(x)
        # x = self.ECA1(x)
        # out = self.ECA(out)
        # out = self.BAM(out)
        # out = self.CBAM(out)


        x = self.layer2(x)
        # x = self.SE2(x)
        x = self.CBAM2(x)
        # x = self.ECA2(x)

        x = self.layer3(x)
        # x = self.SE3(x)
        x = self.CBAM3(x)
        # x = self.ECA3(x)

        x = self.layer4(x)
        # x = self.SE4(x)
        x = self.CBAM4(x)
        # x = self.ECA4(x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet1(ECABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet1(ECABasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet50(k_size=[3, 3, 3, 3], num_classes=2, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing eca_resnet50......")
    model = ResNet1(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet1(ECABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet1(ECABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
# model=eca_resnet18()
# print(model)
# from torchsummary import summary
# print(summary(model,(3,224,224)))
# if isinstance(eca_resnet50(),ResNet1):
#     print("true")