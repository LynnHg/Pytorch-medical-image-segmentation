import torch
from torch import nn
from torchvision import models
from utils.misc import initialize_weights
import torch.nn.functional as F
from networks.custom_modules.basic_modules import *

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels / 2
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class Baseline(nn.Module):
    def __init__(self, img_ch=1, num_classes=6):
        super(Baseline, self).__init__()
        chs = [64, 128, 256, 512, 512]
        self.enc1 = EncoderBlock(img_ch, chs[0], depth=2)
        self.enc2 = EncoderBlock(chs[0], chs[1], depth=2)
        self.enc3 = EncoderBlock(chs[1], chs[2], depth=3)
        self.enc4 = EncoderBlock(chs[2], chs[3], depth=3)
        self.enc5 = EncoderBlock(chs[3], chs[4], depth=3)

        self.dec5 = EncoderBlock(chs[4], chs[3], depth=3)
        self.dec4 = EncoderBlock(chs[3], chs[2], depth=3)
        self.dec3 = EncoderBlock(chs[2], chs[1], depth=3)
        self.dec2 = EncoderBlock(chs[1], chs[0], depth=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chs[0]),
            nn.ReLU(),
            nn.Conv2d(chs[0], num_classes, kernel_size=1, padding=0, bias=False),
        )

        initialize_weights(self)

    def forward(self, x):
        # p1 encoder
        p1_x1 = self.enc1(x)

        p1_x2, p1_id1 = F.max_pool2d(p1_x1, kernel_size=2, stride=2, return_indices=True)
        p1_x2 = self.enc2(p1_x2)

        p1_x3, p1_id2 = F.max_pool2d(p1_x2, kernel_size=2, stride=2, return_indices=True)
        p1_x3 = self.enc3(p1_x3)

        p1_x4, p1_id3 = F.max_pool2d(p1_x3, kernel_size=2, stride=2, return_indices=True)
        p1_x4 = self.enc4(p1_x4)

        p1_x5, p1_id4 = F.max_pool2d(p1_x4, kernel_size=2, stride=2, return_indices=True)
        p1_x5 = self.enc5(p1_x5)

        p1_x6, p1_id5 = F.max_pool2d(p1_x5, kernel_size=2, stride=2, return_indices=True)

        # p1 decoder
        p1_d5 = F.max_unpool2d(p1_x6, p1_id5, kernel_size=2, stride=2)
        d5 = self.dec5(p1_d5)

        p1_d4 = F.max_unpool2d(d5, p1_id4, kernel_size=2, stride=2)
        d4 = self.dec4(p1_d4)

        p1_d3 = F.max_unpool2d(d4, p1_id3, kernel_size=2, stride=2)
        d3 = self.dec3(p1_d3)

        p1_d2 = F.max_unpool2d(d3, p1_id2, kernel_size=2, stride=2)
        d2 = self.dec2(p1_d2)

        p1_d1 = F.max_unpool2d(d2, p1_id1, kernel_size=2, stride=2)
        d1 = self.dec1(p1_d1)

        return d1

if __name__ == "__main__":
    x = torch.randn([2, 1, 256, 256]).cuda()
    # test output size

    net = Baseline().cuda()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    output = net(x)
    print(output.shape)