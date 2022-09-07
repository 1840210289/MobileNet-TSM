from torch import nn
from MODEL.eca_module import eca_layer
from torch.hub import load_state_dict_from_url
from math import log

from MODEL.SEAttention import SEAttention

__all__ = ['ECA_MobileNetV2', 'eca_mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

# 原件
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio, k_size):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         layers = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         ])
#         layers.append(eca_layer(oup, k_size))
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)

# 改件
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, k_size):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

        self.SEAttention = SEAttention(channel=oup, reduction=8)

        if expand_ratio == 1:
            layers.extend([
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
        else:
            layers.extend([
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
        self.gamma = 2
        self.b = 1
        t = int(abs(log(oup, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        k_size = k
        # layers.append(eca_layer(oup, k_size))
        # layers.append(SEAttention(channel=oup, reduction=8))
        self.conv = nn.Sequential(*layers)

    # def forward(self, x):
    #     if self.use_res_connect:
    #         return x + self.conv(x)
    #     else:
    #         return self.conv(x)


    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)

            # y = self.SEblock(out)
            y = self.SEAttention(out)
            return y
        else:
            return self.conv(x)


class ECA_MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.0):
        super(ECA_MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        # print('input_channel:', input_channel)
        # print('self.last_channel:', self.last_channel)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if c < 96:
                    ksize = 1
                else:
                    ksize = 3
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, k_size=ksize))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(-1).mean(-1)
        x = self.classifier(x)
        return x


def eca_mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ECA_MobileNetV2 architecture from

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ECA_MobileNetV2(**kwargs)
    if pretrained:
        model_dict = model.state_dict()
        # sd = model.state_dict()
        sd = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)

        # print(state_dict)
        for k, v in list(sd.items()):
            if 'classifier' in k:
                print('=> New dataset, do not load classifier weights: {}'.format(k))
                # del sd[k]

        sd = {k: v for k, v in sd.items() if 'classifier' not in k}
        model_dict.update(sd)
        # print(state_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    net = eca_mobilenet_v2(False)
    # print(net)
