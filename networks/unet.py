import os
import torch
import math
from torch import nn
import torch.nn.functional as F
from networks.custom_modules.basic_modules import *


'''
================================================================
Total params: 59,393,538
Trainable params: 59,393,538
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 116.25
Params size (MB): 226.57
Estimated Total Size (MB): 342.81
----------------------------------------------------------------
'''

class Baseline(nn.Module):
    def __init__(self, img_ch=1, num_classes=3, depth=2, use_deconv=False):
        super(Baseline, self).__init__()
        chs = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool2d(2, 2)
        # p1 encoder
        self.p1_enc1 = EncoderBlock(img_ch, chs[0], depth=depth)
        self.p1_enc2 = EncoderBlock(chs[0], chs[1], depth=depth)
        self.p1_enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.p1_enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.p1_cen = EncoderBlock(chs[3], chs[4], depth=depth)

        self.p1_dec4 = DecoderBlock(chs[4], chs[3], use_deconv=use_deconv)
        self.p1_decconv4 = EncoderBlock(chs[3] * 2, chs[3])

        self.p1_dec3 = DecoderBlock(chs[3], chs[2], use_deconv=use_deconv)
        self.p1_decconv3 = EncoderBlock(chs[2] * 2, chs[2])

        self.p1_dec2 = DecoderBlock(chs[2], chs[1], use_deconv=use_deconv)
        self.p1_decconv2 = EncoderBlock(chs[1] * 2, chs[1])

        self.p1_dec1 = DecoderBlock(chs[1], chs[0], use_deconv=use_deconv)
        self.p1_decconv1 = EncoderBlock(chs[0] * 2, chs[0])

        self.p1_conv_1x1 = nn.Conv2d(chs[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # p1 encoder
        p1_x1 = self.p1_enc1(x)
        p1_x2 = self.pool(p1_x1)
        p1_x2 = self.p1_enc2(p1_x2)
        p1_x3 = self.pool(p1_x2)
        p1_x3 = self.p1_enc3(p1_x3)
        p1_x4 = self.pool(p1_x3)
        p1_x4 = self.p1_enc4(p1_x4)
        p1_center = self.pool(p1_x4)
        p1_center = self.p1_cen(p1_center)

        """
          first path decoder
        """
        d4 = self.p1_dec4(p1_center)
        d4 = torch.cat((p1_x4, d4), dim=1)
        d4 = self.p1_decconv4(d4)

        d3 = self.p1_dec3(d4)
        d3 = torch.cat((p1_x3, d3), dim=1)
        d3 = self.p1_decconv3(d3)

        d2 = self.p1_dec2(d3)
        d2 = torch.cat((p1_x2, d2), dim=1)
        d2 = self.p1_decconv2(d2)

        d1 = self.p1_dec1(d2)
        d1 = torch.cat((p1_x1, d1), dim=1)
        d1 = self.p1_decconv1(d1)

        p1_out = self.p1_conv_1x1(d1)

        return p1_out


if __name__ == '__main__':
    from torchsummary import summary

    x1 = torch.randn([2, 1, 160, 160]).cuda()
    net = Baseline().cuda()
    summary(net, input_size=[(1, 160, 160)])
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    pred = net(x1)
    print(pred.shape)
