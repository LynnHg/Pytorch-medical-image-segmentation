import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import initialize_weights


activation = nn.ReLU


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='relu', bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.conv.append(Mish())
        elif activation == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'linear':
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, use_res=False):
        super(EncoderBlock, self).__init__()

        self.use_res = use_res

        self.conv = [nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            activation(),
        )]

        for i in range(1, depth):
            self.conv.append(nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                           nn.BatchNorm2d(ch_out),
                                           nn.Sequential() if use_res and i == depth-1 else activation()
                                           ))
        self.conv = nn.Sequential(*self.conv)
        if use_res:
            self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        if self.use_res:
            residual = self.conv1x1(x)

        x = self.conv(x)

        if self.use_res:
            x += residual
            x = F.relu(x)

        return x


class DecoderBlock(nn.Module):
    """
    Interpolate
    """

    def __init__(self, ch_in, ch_out, use_deconv=False):
        super(DecoderBlock, self).__init__()
        if use_deconv:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                activation()
            )

    def forward(self, x):
        return self.up(x)


class ResBlockV1(nn.Module):
    """
    Post-activation
    """

    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(ResBlockV1, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)

        return out


class ResBlockV2(nn.Module):
    """
    Post-activation
    """

    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(ResBlockV2, self).__init__()

        self.bn1 = nn.BatchNorm2d(ch_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size, 1, kernel_size // 2, bias=False)

        self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))

        out += residual

        return out


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        return self.conv(x)


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        # print('spatial',x.size())
        x = torch.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels, out_channels=int(out_channels / 2), kernel_size=1, padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        # print('channel',x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # print('x',x.size())
        # print('e',e.size())
        if e is not None:
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        # print('x_new',x.size())
        g1 = self.spatial_gate(x)
        # print('g1',g1.size())
        g2 = self.channel_gate(x)
        # print('g2',g2.size())
        x = g1 * x + g2 * x
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, scale_factor=2):
        super(Bottleneck, self).__init__()
        reduction_chs = out_chs // scale_factor
        self.bt_1 = nn.Sequential(
            nn.Conv2d(in_chs, reduction_chs, 1, bias=False),
            nn.BatchNorm2d(reduction_chs),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            nn.Conv2d(reduction_chs, reduction_chs, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(reduction_chs),
            nn.ReLU(inplace=True)
        )

        self.bt_2 = nn.Sequential(
            nn.Conv2d(reduction_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.bt_1(x)
        x = self.features(x)
        x = self.bt_2(x)

        return x

if __name__ == '__main__':
    x = torch.randn([1, 64, 16, 16])
    conv = Conv_Bn_Activation(16, 32, 3)
    print(conv)
