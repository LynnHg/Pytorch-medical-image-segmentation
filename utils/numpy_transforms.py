import cv2
import math
import sys
import numbers
import numpy as np
from skimage import measure
from utils import helpers


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if (self.size <= img.shape[0]) and (self.size <= img.shape[1]):
            x = math.ceil((img.shape[0] - self.size) / 2.)
            y = math.ceil((img.shape[1] - self.size) / 2.)

            if len(img.shape) == 3:
                return img[x:x + self.size, y:y + self.size, :]
            else:
                return img[x:x + self.size, y:y + self.size]
        else:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                self.size, self.size, img.shape[0], img.shape[1]))


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if (self.size <= img.shape[0]) and (self.size <= img.shape[1]):
            x = img.shape[0] - self.size
            y = img.shape[1] - self.size
            offsetx = np.random.randint(x)
            offsety = np.random.randint(y)

            if len(img.shape) == 3:
                return img[offsetx:offsetx + self.size, offsety:offsety + self.size, :], mask[offsetx:offsetx + self.size, offsety:offsety + self.size, :]
            else:
                return img[offsetx:offsetx + self.size, offsety:offsety + self.size], mask[offsetx:offsetx + self.size, offsety:offsety + self.size]
        else:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                self.size, self.size, img.shape[0], img.shape[1]))


class FixedRandomCrop(object):
    def __init__(self, size, crop_size):
        self.size = crop_size
        x = size - crop_size
        self.offset = np.random.randint(x)

    def __call__(self, img, mask):

        return img[self.offset:self.offset + self.size, self.offset:self.offset + self.size],\
               mask[self.offset:self.offset + self.size, self.offset:self.offset + self.size]


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        '''
        :param img:  shape of [H, W, C]
        :return: img shape of [H, W, C]
        '''
        flag = np.random.randint(2)
        if flag:
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), self.angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            return img
        return img


class ROICenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def ConnectComponent(self, bw_img):
        labeled_img, num = measure.label(bw_img, background=0, connectivity=1, return_num=True)

        lcc = np.array(labeled_img, dtype=np.uint8)
        return lcc, num

    def get_bounding_box(self, mask):
        stan = np.array(helpers.array_to_img(np.expand_dims(mask, axis=2)))
        max_value = np.max(stan)
        ret, bmask = cv2.threshold(stan, 1, max_value, cv2.THRESH_BINARY)
        lcc, _ = self.ConnectComponent(bmask)
        props = measure.regionprops(lcc)
        MINC, MINR, MAXC, MAXR = sys.maxsize, sys.maxsize, 0, 0
        for i in range(len(props)):
            minr, minc, maxr, maxc = props[i].bbox
            MINC, MINR = min(MINC, minc), min(MINR, minr)
            MAXC, MAXR = max(MAXC, maxc), max(MAXR, maxr)

        return MINR, MINC, MAXR, MAXC

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        th, tw = self.size

        minr, minc, maxr, maxc = self.get_bounding_box(mask)

        w = maxc - minc
        h = maxr - minr
        offset_w = math.ceil((th - w) / 2.)
        offset_h = math.ceil((th - h) / 2.)
        col, row = minc - offset_w, minr - offset_h

        # visualization
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(mask, cmap='gray')
        # bbox = Rectangle((col, row), tw, th, fill=False, edgecolor='blue', linewidth=2)
        # ax.add_patch(bbox)
        # plt.show()

        return img[row:row+th, col:col+tw], mask[row:row+th, col:col+tw]


# just test
class ROICrop(object):

    def __init__(self):
        pass

    def ConnectComponent(self, bw_img):
        labeled_img, num = measure.label(bw_img, background=0, connectivity=1, return_num=True)

        lcc = np.array(labeled_img, dtype=np.uint8)
        return lcc, num

    def get_bounding_box(self, mask):
        stan = np.array(helpers.array_to_img(np.expand_dims(mask, axis=2)))
        max_value = np.max(stan)
        ret, bmask = cv2.threshold(stan, 1, max_value, cv2.THRESH_BINARY)
        lcc, _ = self.ConnectComponent(bmask)
        props = measure.regionprops(lcc)
        MINC, MINR, MAXC, MAXR = sys.maxsize, sys.maxsize, 0, 0
        for i in range(len(props)):
            minr, minc, maxr, maxc = props[i].bbox
            MINC, MINR = min(MINC, minc), min(MINR, minr)
            MAXC, MAXR = max(MAXC, maxc), max(MAXR, maxr)

        return MINR, MINC, MAXR, MAXC

    def __call__(self, img, mask):
        assert img.shape == mask.shape

        minr, minc, maxr, maxc = self.get_bounding_box(mask)

        w = maxc - minc
        h = maxr - minr
        leader = max(w, h)

        crop_size = math.ceil(leader / 16) * 16

        col, row = minc, minr

        # visualization
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(mask, cmap='gray')
        # bbox = Rectangle((col, row), crop_size, crop_size, fill=False, edgecolor='blue', linewidth=2)
        # ax.add_patch(bbox)
        # plt.show()

        return img[row:row+crop_size, col:col+crop_size], mask[row:row+crop_size, col:col+crop_size]


