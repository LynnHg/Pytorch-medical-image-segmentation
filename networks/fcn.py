import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.misc import initialize_weights


class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, use_bn=False):
        super(EncoderBlock, self).__init__()

        self.conv = [nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out) if use_bn else nn.Sequential(),
            nn.ReLU(),
        )]

        for i in range(1, depth):
            self.conv.append(nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                           nn.BatchNorm2d(ch_out) if use_bn else nn.Sequential(),
                                           nn.ReLU()
                                           ))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)

class Baseline(nn.Module):

    def __init__(self, img_ch=1, num_classes=6):
        super().__init__()

        chs = [64, 128, 256, 512, 512]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.enc1 = EncoderBlock(img_ch, chs[0], depth=2)
        self.enc2 = EncoderBlock(chs[0], chs[1], depth=2)
        self.enc3 = EncoderBlock(chs[1], chs[2], depth=3)
        self.enc4 = EncoderBlock(chs[2], chs[3], depth=3)
        self.enc5 = EncoderBlock(chs[3], chs[4], depth=3)

        # p1
        self.deconv1 = nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(chs[3])
        self.deconv2 = nn.ConvTranspose2d(chs[3], chs[2], kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(chs[2])
        self.deconv3 = nn.ConvTranspose2d(chs[2], chs[1], kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(chs[1])
        self.deconv4 = nn.ConvTranspose2d(chs[1], chs[0], kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(chs[0])
        self.deconv5 = nn.ConvTranspose2d(chs[0], 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

        initialize_weights(self)

    def forward(self, x):
        # p1
        p1_x1 = self.enc1(x)
        p1_x1 = self.maxpool(p1_x1)

        p1_x2 = self.enc2(p1_x1)
        p1_x2 = self.maxpool(p1_x2)

        p1_x3 = self.enc3(p1_x2)
        p1_x3 = self.maxpool(p1_x3)

        p1_x4 = self.enc4(p1_x3)
        p1_x4 = self.maxpool(p1_x4)

        p1_x5 = self.enc5(p1_x4)
        p1_x5 = self.maxpool(p1_x5)

        # p1 decoder
        score = self.relu(self.deconv1(p1_x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + p1_x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + p1_x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


if __name__ == "__main__":
    x = torch.randn([2, 1, 256, 256]).cuda()

    # test output size

    net = Baseline().cuda()
    # print(fcn_model)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    output = net(x)
    print(output.shape)



