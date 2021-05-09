import random

import numpy as np
import torch
from PIL import Image, ImageFilter


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.float32))


class NpyToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.float32))


class NpyToTensorV2(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.float32)).div(255)


class ImgToTensor(object):
    def __call__(self, img):
        img = torch.from_numpy(np.array(img))
        if isinstance(img, torch.ByteTensor):
            return img.float()


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


