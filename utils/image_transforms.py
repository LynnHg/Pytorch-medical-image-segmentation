import cv2
import math
import sys
import numbers
import random
from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class CenterCropV2(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class SingleCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size=0, scale_rate=0.95, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_rate = scale_rate
        self.fill = fill

    def __call__(self, im, gt):
        img = im.copy()
        mask = gt.copy()
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.scale_rate), int(self.base_size * self.scale_rate))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        # ramdom crop
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class ScaleCrop(object):
    def __init__(self, base_size, crop_size=0, scale_rate=0.95, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_rate = scale_rate
        self.fill = fill

    def __call__(self, im, gt):
        img = im.copy()
        mask = gt.copy()
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.scale_rate), int(self.base_size * self.scale_rate))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = (self.crop_size - oh) // 2 if oh < self.crop_size else 0
            padw = (self.crop_size - ow) // 2 if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        # ramdom crop
        x1 = (w - self.crop_size) // 2
        y1 = (h - self.crop_size) // 2
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask





