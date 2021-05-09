import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
from utils import helpers
from tqdm import tqdm

'''
0 = background 
200 = LV normal myocardium
500 = LV
600 = RV 
'''
palette = [[0], [200], [500], [600]]
palette_color = [[0, 0, 0], [255, 255, 255], [101, 12, 68], [68, 104, 118]]
num_classes = 4

c0_lge_t2_mean_std = ((398.816, 395.903), (242.600, 158.449), (164.044, 182.646))


def get_mean_std(train_loader):
    samples, mean, std = 0, 0, 0
    for (input, mask, mask_copy) in tqdm(train_loader):
        samples += 1
        mean += np.mean(input.numpy(), axis=(0, 2, 3))
        std += np.std(input.numpy(), axis=(0, 2, 3))
    mean /= samples
    std /= samples
    print(mean, std)


def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')

        if 'Augdata' in root:
            data_list = os.listdir(os.path.join(root, 'Labels'))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]
        for it in data_list:
            items.append((os.path.join(img_path, it), os.path.join(mask_path, it)))
    elif mode == 'val':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val{}.txt'.format(fold))).readlines()]
        for it in data_list:
            items.append((os.path.join(img_path, it), os.path.join(mask_path, it)))
    else:
        img_path = os.path.join(root, 'Images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'test.txt')).readlines()]
        for it in data_list:
            items.append(os.path.join(img_path, it))
    return items


class MSCMR2019(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, roi_crop=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.roi_crop = roi_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # 记录原始尺寸
        init_size = 0
        if self.mode is not 'test':
            img_path, mask_path = self.imgs[index]
            file_name = mask_path.split('\\')[-1]

            # 多模态原图像加载
            img = np.load(img_path)
            mask = np.load(mask_path)
            init_size = mask.shape

            img = Image.fromarray(img)
            mask = Image.fromarray(mask)

            if self.joint_transform is not None:
                img, mask= self.joint_transform(img, mask)

            if self.center_crop is not None:
                img, mask = self.center_crop(img, mask)

            img = np.array(img)
            img = np.expand_dims(img, axis=2)
            img = img.transpose([2, 0, 1])

            if self.transform is not None:
                img = self.transform(img)
                # Z-Score
                img = (img - c0_lge_t2_mean_std[0][0]) / c0_lge_t2_mean_std[0][1]

            # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
            # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
            mask = np.expand_dims(mask, axis=2)
            mask = helpers.mask_to_onehot(mask, self.palette)

            # shape from (H, W, C) to (C, H, W)
            mask = mask.transpose([2, 0, 1])

            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return img, mask, file_name
        else:
            img_path = self.imgs[index]
            file_name = img_path[0].split('\\')[-1]
            # 多模态原图像加载
            img = np.load(img_path)
            init_size = img.shape
            img = Image.fromarray(img)

            if self.joint_transform is not None:
                img = self.joint_transform(img)
            if self.center_crop is not None:
                img = self.center_crop(img)

            img = np.array(img)
            img = np.expand_dims(img, axis=2)
            img = img.transpose([2, 0, 1])

            if self.transform is not None:
                img = self.transform(img)
                # Z-Score
                img = (img - c0_lge_t2_mean_std[0][0]) / c0_lge_t2_mean_std[0][1]

            return img, file_name, init_size

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    from torch.utils.data import DataLoader
    import utils.image_transforms as joint_transforms
    import utils.transforms as extended_transforms

    def demo():

        train_path = r'../media/Datasets/MSCMR2019/T2/npy'

        center_crop = joint_transforms.CenterCrop(160)
        tes_center_crop = joint_transforms.SingleCenterCrop(160)
        train_input_transform = extended_transforms.NpyToTensor()

        target_transform = extended_transforms.MaskToTensor()

        train_set = MSCMR2019(train_path, 'train', 1,
                              joint_transform=None, roi_crop=None, center_crop=center_crop,
                              transform=train_input_transform, target_transform=target_transform)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

        # 求训练集的均值和方差
        # get_mean_std(train_loader)

        for input, mask, mask_copy, _ in train_loader:
            print(input.shape, mask.shape, mask_copy.shape)
            a = mask_copy.squeeze()
            b = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), palette_color)
            c = input.squeeze() * c0_lge_t2_mean_std[0][1] + c0_lge_t2_mean_std[0][0]

            cv2.imshow('mat1', np.uint8(np.array(b)))
            cv2.imshow('mat', np.array(a))
            cv2.imshow('mat3', np.array(c))
            cv2.waitKey(0)

    demo()


    # count()