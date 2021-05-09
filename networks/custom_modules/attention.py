import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ['SpatialAttention', 'ChannelAttention']


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(SpatialAttention, self).__init__()

        pad = (kernel_size - 1) // 2

        self.conv_l_1xk = nn.Conv2d(in_channels, in_channels, (1, kernel_size), padding=(0, pad))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_l_kx1 = nn.Conv2d(in_channels, 1, (kernel_size, 1), padding=(pad, 0))
        self.bn2 = nn.BatchNorm2d(1)

        self.conv_r_kx1 = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), padding=(pad, 0))
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.conv_r_1xk = nn.Conv2d(in_channels, 1, (1, kernel_size), padding=(0, pad))
        self.bn4 = nn.BatchNorm2d(1)

    def forward(self, x):
        # left conv
        f1 = F.relu(self.bn1(self.conv_l_1xk(x)))
        f1 = F.relu(self.bn2(self.conv_l_kx1(f1)))

        # right conv
        f2 = F.relu(self.bn3(self.conv_r_kx1(x)))
        f2 = F.relu(self.bn4(self.conv_r_1xk(f2)))

        out = torch.sigmoid(f1 + f2)
        out = out.expand_as(x)

        return out * x + x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        self.in_channels = in_channels

        self.fc1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.fc2 = nn.Linear(self.in_channels // 4, self.in_channels)

    def forward(self, x):
        n, c, h, w = x.size()

        out = F.adaptive_avg_pool2d(x, (1, 1)).view((n, c))
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        out = out.view((n, c, 1, 1))
        out = out.expand_as(x).clone()

        return torch.mul(out, x)


class SELayer(nn.Module):
    """
    Proposed by Squeeze-and-Excitation Networks(SENet)
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PAM(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # 可学习参数，论文中分别为alpha和beta. 初始化为0，逐渐优化
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """

        B, C, H, W = x.size()
        # B: reshape and transpose. reshape to B×C×N, where N = H×W
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        # C: just reshape
        proj_key = self.key_conv(x).view(B, -1, W * H)

        # perform a matrix multiplication
        energy = torch.bmm(proj_query, proj_key)
        # get S: apply a softmax layer to calculate the spatial attention map
        attention = self.softmax(energy)
        # D: just reshape
        proj_value = self.value_conv(x).view(B, -1, W * H)
        # S transpose and perform a matrix multiplication
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class CAM(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)

        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # get X: apply a softmax layer to calculate the spatial attention map
        attention = self.softmax(energy_new)
        proj_value = x.view(B, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class sSE(nn.Module):
    def __init__(self, in_channels):
        super(sSE, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, in_channels):
        super(cSE, self).__init__()

        reduction = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=reduction, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduction)
        self.conv2 = nn.Conv2d(in_channels=reduction, out_channels=reduction, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(reduction)

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    y = torch.randn(2, 64, 16, 16)
    # Test Spatial Attention
    sa = cSE(64)
    print(sa(y).shape)
