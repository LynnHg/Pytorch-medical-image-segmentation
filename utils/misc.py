import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import helpers
from PIL import Image
import datetime


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def initialize_weights(*models, a=0):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, a=a)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def data_rotate(im, angle):
    M = cv2.getRotationMatrix2D((im.shape[1] // 2, im.shape[0] // 2), angle, 1.0)
    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST)
    return im


def edge_detection(ims, dataset):
    edges = []
    n, c, h, w = ims.shape
    robertsY = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype='float32')
    robertsX = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype='float32')

    kernely = torch.FloatTensor(robertsY).expand(1, 1, 3, 3)
    kernelx = torch.FloatTensor(robertsX).expand(1, 1, 3, 3)
    weighty = nn.Parameter(data=kernely, requires_grad=False)
    weightx = nn.Parameter(data=kernelx, requires_grad=False)

    for i in range(n):
        im = ims[i].cpu().detach().numpy().transpose((1, 2, 0))
        im = np.float32(helpers.onehot_to_mask(im, dataset.palette))
        im = torch.from_numpy(im).unsqueeze(0).permute((0, 3, 1, 2)).contiguous()

        edgey = torch.abs(F.conv2d(im, weighty, padding=1))
        edgex = torch.abs(F.conv2d(im, weightx, padding=1))
        edge = torch.where(edgex > 0, edgex, edgey)
        # 二进制边界
        m_edge = edge.detach()
        # edge = np.array(edge.squeeze(0)).transpose([1, 2, 0])
        #
        # edge = helpers.mask_to_onehot(edge, dataset.palette)
        # edge = np.expand_dims(edge, 3)
        # edge = edge.transpose([3, 2, 0, 1])
        # edge = torch.from_numpy(edge)
        # edges.append(edge)
    # 语义边界
    # edges = torch.cat([*edges], dim=0)
    return m_edge
    # return edges, m_edge


def test_edge_detection(dataset):
    im = np.load(r'C:\Learning\python\LYNNet\media\LIBRARY\Datasets\MyoPS2020\npy\Labels\myops_training_124_3.npy')
    im = np.load(r'C:\Learning\python\LYNNet\results\myops_training_125_1.png')
    im = np.array(im, dtype='float32')
    im = np.expand_dims(im, axis=2)
    gt = helpers.mask_to_onehot(im, dataset.palette)

    gt = np.expand_dims(gt, axis=3)
    gt = gt.transpose([3, 2, 0, 1])
    gt = torch.from_numpy(gt)

    m_edge = edge_detection(gt, dataset)


    m_edge = np.array(m_edge.squeeze(0)).transpose([1, 2, 0])
    # edge = np.array(edge.squeeze(0)).transpose([1, 2, 0])
    cv2.imshow('eg', m_edge)
    cv2.waitKey(0)

def log(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        # print(time_stamp + " " + X)
        print(X)
    else:
        f.write(time_stamp + " " + X)


if __name__ == '__main__':
    from datasets import mscmr2019

    test_edge_detection(mscmr2019)