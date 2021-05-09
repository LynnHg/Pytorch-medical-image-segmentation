import os, shutil
import cv2
import numpy as np
import nibabel as nb
from tqdm import tqdm
from PIL import Image

from utils.image_transforms import RandomScaleCrop
from utils.misc import data_rotate
from datasets import mscmr2019
from utils import helpers


palette = mscmr2019.palette
custom_palette = mscmr2019.custom_palette


def nii_to_npy(dataset_dir, save_path, modal_type='C0', to_png=True):
    im_gt_path = ['train45', 'train45_gd']
    # 所有模态数据
    all_modal_im_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[0]))
    all_modal_gt_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[1]))

    # 区分每个模态数据
    im_namelist = [item for item in all_modal_im_namelist if modal_type in item]
    gt_namelist = [item for item in all_modal_gt_namelist if modal_type in item]
    im_namelist.sort(key=lambda x: x.split('_')[0])
    gt_namelist.sort(key=lambda x: x.split('_')[0])

    if os.path.exists(os.path.join(save_path, modal_type)):
        shutil.rmtree(os.path.join(save_path, modal_type))
    else:
        os.makedirs(os.path.join(save_path, modal_type, 'npy', 'Images'))
        os.makedirs(os.path.join(save_path, modal_type, 'npy', 'Labels'))
        if to_png:
            os.makedirs(os.path.join(save_path, modal_type, 'color_png', 'Images'))
            os.makedirs(os.path.join(save_path, modal_type, 'color_png', 'Labels'))

    # 每个模态下的所有病例
    for n in tqdm(range(len(gt_namelist))):
        im = nb.load(os.path.join(dataset_dir, im_gt_path[0], im_namelist[n])).dataobj
        gt = nb.load(os.path.join(dataset_dir, im_gt_path[1], gt_namelist[n])).dataobj
        h, w, c = gt.shape
        # 每一个病例的所有切片
        for i in range(c):
            mask = gt[:, :, i]

            # 跳过gt全黑图
            if np.sum(mask) == 0:
                continue
            npy = im[:, :, i]

            # 保存png格式
            if to_png:
                png = helpers.array_to_img(np.expand_dims(npy, axis=2))
                # mask_png = helpers.array_to_img(np.expand_dims(mask, axis=2))

                mask_png_color = helpers.mask_to_onehot(np.expand_dims(mask, axis=2), palette)
                mask_png_color = helpers.onehot_to_mask(mask_png_color, custom_palette)
                mask_png_color = helpers.array_to_img(mask_png_color)

                file_name = im_namelist[n].split('.')[0] + '{}.png'.format(i)

                png.save(os.path.join(save_path, modal_type, 'color_png', 'Images', file_name))
                # mask_png.save(os.path.join(save_path, 'png', 'Labels', file_name))
                mask_png_color.save(os.path.join(save_path, modal_type, 'color_png', 'Labels', file_name))

            # 保存npy格式
            file_name = im_namelist[n].split('.')[0] + '{}.npy'.format(i)
            np.save(os.path.join(save_path, modal_type, 'npy', 'Images', file_name), npy)
            np.save(os.path.join(save_path, modal_type, 'npy', 'Labels', file_name), mask)

            with open(os.path.join(save_path, modal_type, 'npy', 'all.txt'), 'a') as f:
                f.write(file_name)
                f.write('\n')


# 数据增广
def data_augmentation(dataset_dir, save_path, modal_type='C0', to_png=True):
    save_path_init = dataset_dir
    aug_data_file = ['train1', 'train2', 'train3', 'train4', 'train5']
    angles = [45, 90, 135, 180, 225, 270, 315]
    scale_rate = [0.75, 0.8, 0.9, 1, 1.1, 1.25]
    flip_codes = [1, 0, -1]
    aug_file_name = 'Augdata{}'.format(modal_type)
    for aug_data_file in aug_data_file:
        items = []
        img_path = os.path.join(dataset_dir, modal_type, 'npy')
        data_list = [l.strip('\n') for l in open(os.path.join(
            img_path, '{}.txt'.format(aug_data_file))).readlines()]
        # 获取图像路径
        for it in data_list:
            item = (os.path.join(img_path, 'Images', it), os.path.join(img_path, 'Labels', it))
            items.append(item)

        # 创建增广数据目录
        if os.path.exists(os.path.join(save_path_init, aug_file_name, aug_data_file)):
            # 若该目录已存在，则先删除，用来清空数据
            print('清空原始数据中...')
            shutil.rmtree(os.path.join(save_path_init, aug_file_name, aug_data_file))
            print('原始数据已清空。')

        save_path = os.path.join(save_path_init, aug_file_name, aug_data_file)
        npy_save_path = os.path.join(save_path, 'npy')

        os.makedirs(os.path.join(npy_save_path, 'Images'))
        os.makedirs(os.path.join(npy_save_path, 'Labels'))

        if to_png:
            png_save_path = os.path.join(save_path, 'png')
            os.makedirs(os.path.join(png_save_path, 'Images'))
            os.makedirs(os.path.join(png_save_path, 'Labels'))
        # 加载图像
        for item in tqdm(items):
            img_path, mask_path = item
            file_name = mask_path.split('\\')[-1][:-4] # patient18_C09
            im = np.load(img_path)
            gt = np.load(mask_path)

            # 旋转
            for angle in angles:
                img_ = data_rotate(im, angle)
                gt_ = data_rotate(gt, angle)
                extra = '_rotate{}'.format(angle)

                if to_png:
                    png_file_name = file_name + '{}.png'.format(extra)
                    png_im = helpers.array_to_img(np.expand_dims(img_, axis=2))
                    png_gt = helpers.array_to_img(np.expand_dims(gt_, axis=2))
                    png_im.save(os.path.join(png_save_path, 'Images', png_file_name))
                    png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                npy_file_name = file_name + '{}.npy'.format(extra)
                np.save(os.path.join(npy_save_path, 'Images', npy_file_name), img_)
                np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)

                # 三个模态只需保存任意一次名字

                with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                    f.write(npy_file_name)
                    f.write('\n')
            # 随机缩放
            for sr in scale_rate:
                SR = RandomScaleCrop(256, 256, scale_rate=sr)
                extra = '_scale{}'.format(sr)
                img_, gt_ = SR(Image.fromarray(im), Image.fromarray(gt))
                img_, gt_ = np.array(img_), np.array(gt_)

                if to_png:
                    png_file_name = file_name + '{}.png'.format(extra)
                    png_im = helpers.array_to_img(np.expand_dims(img_, axis=2))
                    png_gt = helpers.array_to_img(np.expand_dims(gt_, axis=2))
                    png_im.save(os.path.join(png_save_path, 'Images', png_file_name))
                    png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                npy_file_name = file_name + '{}.npy'.format(extra)
                np.save(os.path.join(npy_save_path, 'Images', npy_file_name), img_)
                np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)
                with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                    f.write(npy_file_name)
                    f.write('\n')

            # 翻转
            for code in flip_codes:
                img_ = cv2.flip(im, code)
                gt_ = cv2.flip(gt, code)
                if to_png:
                    png_file_name = file_name + '_flip{}.png'.format(code)
                    png_im = helpers.array_to_img(np.expand_dims(img_, axis=2))
                    png_gt = helpers.array_to_img(np.expand_dims(gt_, axis=2))
                    png_im.save(os.path.join(png_save_path, 'Images', png_file_name))
                    png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                npy_file_name = file_name + '_flip{}.npy'.format(code)
                np.save(os.path.join(npy_save_path, 'Images', npy_file_name), img_)
                np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)

                with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                    f.write(npy_file_name)
                    f.write('\n')


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    # 1.3d nii 图像转为 2d 切片
    types = ['C0', 'LGE', 'T2']
    # nii_to_npy(r'E:\Datasets\MSCMR2019', r'../media/Datasets/MSCMR2019', modal_type=types[2])


    # 3. 数据集增广
    data_augmentation(r'E:\Python_WorkSpace\Pytorch-medical-image-segmentation\media\Datasets\MSCMR2019',
                      r'E:\Python_WorkSpace\Pytorch-medical-image-segmentation\media\Datasets', modal_type=types[2])


