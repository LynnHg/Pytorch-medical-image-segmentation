import os
import cv2
import torch
import shutil
import utils.image_transforms as image_transforms
from torch.utils.data import DataLoader
import utils.transforms as extended_transforms
from tqdm import tqdm
from datasets import mscmr2019 as mscmr2019
from utils.loss import *
from hausdorff import hausdorff_distance


type = "baseline"  # lynn baseline
model_type = "attunet"  # lynn baseline

if type == "baseline":
    if model_type == "siamese":
        from networks.siamese_unet import Baseline
    elif model_type == "mcie":
        from networks.baseline import Baseline
    elif model_type == "unet":
        from networks.unet import Baseline
    elif model_type == "fcn":
        from networks.fcn import Baseline
    elif model_type == "segnet":
        from networks.segnet import Baseline
    elif model_type == "attunet":
        from networks.attunet import Baseline
else:
    from networks.baseline import Baseline

fold = 2  # 1 2 3 4 5
augdata = "C0"
root_path = '../'
val_path = os.path.join(root_path, 'media/Datasets/MSCMR2019', augdata, 'npy')

input_transform = extended_transforms.NpyToTensor()
center_crop = image_transforms.CenterCrop(160)
target_transform = extended_transforms.MaskToTensor()

val_set = mscmr2019.MSCMR2019(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, roi_crop=None,
                                  center_crop=center_crop,
                                  target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

palette = mscmr2019.palette
palette_color = mscmr2019.palette_color
num_classes = mscmr2019.num_classes

net = Baseline(num_classes=num_classes).cuda()
net.load_state_dict(torch.load("../checkpoint/attunet_fold{}_dice_{}_722426.pth".format(fold, augdata)))
net.eval()


def auto_val(net):
    # 效果展示图片数
    dices = 0
    # 所有切片的各个类别指标总和
    class_dices = np.array([0] * (num_classes - 1), dtype=np.float)
    class_jaccards = np.array([0] * (num_classes - 1), dtype=np.float)
    class_hsdfs = np.array([0] * (num_classes - 1), dtype=np.float)

    save_path = './results'
    if os.path.exists(save_path):
        # 若该目录已存在，则先删除，用来清空数据
        shutil.rmtree(os.path.join(save_path))
    img_path = os.path.join(save_path, 'images')
    pred_path = os.path.join(save_path, 'pred')
    color_pred_path = os.path.join(save_path, 'color_pred')
    gt_path = os.path.join(save_path, 'gt')
    os.makedirs(img_path)
    os.makedirs(pred_path)
    os.makedirs(color_pred_path)
    os.makedirs(gt_path)

    # 存放每个切片的指标数组
    val_dice_arr = []
    val_lesion_dice_arr = []
    val_jaccard_arr = []
    val_lesion_jaccard_arr = []
    val_hsdf_arr = []
    val_lesion_hsdf_arr = []
    for slice, (input, mask, mask_copy, file_name) in tqdm(enumerate(val_loader, 1)):
        # init_size = (init_size[0].item(), init_size[1].item())
        file_name = file_name[0].split('.')[0]

        X = input.cuda()
        if type == "lynn":
            pred, _ = net(X)
        elif type == "baseline":
            pred = net(X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()
        # pred[pred < 0.5] = 0
        # pred[pred > 0.5] = 1

        # 原图
        m1 = np.array(input.squeeze())
        m1 = helpers.array_to_img(np.expand_dims(m1, 2))

        # gt
        gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), palette)
        gt = helpers.array_to_img(gt)

        # pred
        save_pred = helpers.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), palette)
        save_pred_png = helpers.array_to_img(save_pred)
        save_pred_color = helpers.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), palette_color)
        save_pred_png_color = helpers.array_to_img(save_pred_color)

        # 保存预测结果
        # png格式
        m1.save(os.path.join(img_path, file_name + '.png'))
        gt.save(os.path.join(gt_path, file_name + '.png'))
        save_pred_png.save(os.path.join(pred_path, file_name + '.png'))
        save_pred_png_color.save(os.path.join(color_pred_path, file_name + '.png'))

        class_dice = []
        class_jaccard = []
        class_hsdf = []
        for i in range(1, num_classes):
            class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_jaccard.append(jaccardv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_hsdf.append(hausdorff_distance(np.array(pred[0, i:i + 1, :].squeeze()), np.array(mask[0, i:i + 1, :].squeeze()), distance='manhattan'))

        # dice 指标
        val_dice_arr.append(class_dice) # 所有切片的各类别dice [[0.9, 0.9, 0.8], [0.7, 0.8, 0.6], ...]
        class_dices += np.array(class_dice) # 所有切片的各类别dice总和 # eg. [72.29878073 83.99182165 62.14122799]
        # jaccard 指标
        val_jaccard_arr.append(class_jaccard)
        class_jaccards += np.array(class_jaccard)
        # hsdf 指标
        val_hsdf_arr.append(class_hsdf)
        class_hsdfs += np.array(class_hsdf)
        # print('{}, mean_dice: {:.4} - dice_lvm: {:.4} - dice_lv: {:.4} - dice_rv: {:.4}'
        #       .format(slice, sum(class_dice) / 3,  class_dice[0], class_dice[1], class_dice[2]))
    # dice 指标
    dice_slice_arr = np.array(val_dice_arr)
    dice_slice_allclass_mean_arr = np.mean(dice_slice_arr, axis=1)  # 存放所有切片的类别平均dice
    dice_slice_class_mean_arr = np.mean(dice_slice_arr, axis=0)  # 存放所有切片的类别dice
    dice_mean = np.mean(dice_slice_allclass_mean_arr)
    dice_slice_mean_std = np.std(dice_slice_allclass_mean_arr)  # 所有切片的类别平均dice的方差
    dice_slice_std_lvm = np.std(dice_slice_arr[:, 0:1].squeeze())
    dice_slice_std_lv = np.std(dice_slice_arr[:, 1:2].squeeze())
    dice_slice_std_rv = np.std(dice_slice_arr[:, 2:3].squeeze())

    # jaccard 指标
    jaccard_slice_arr = np.array(val_jaccard_arr)
    jaccard_slice_allclass_mean_arr = np.mean(jaccard_slice_arr, axis=1)
    jaccard_slice_class_mean_arr = np.mean(jaccard_slice_arr, axis=0)
    jaccard_mean = np.mean(jaccard_slice_allclass_mean_arr)
    jaccard_slice_mean_std = np.std(jaccard_slice_allclass_mean_arr)
    jaccard_slice_std_lvm = np.std(jaccard_slice_arr[:, 0:1].squeeze())
    jaccard_slice_std_lv = np.std(jaccard_slice_arr[:, 1:2].squeeze())
    jaccard_slice_std_rv = np.std(jaccard_slice_arr[:, 2:3].squeeze())

    # hsdf 指标
    hsdf_slice_arr = np.array(val_hsdf_arr)
    hsdf_slice_allclass_mean_arr = np.mean(hsdf_slice_arr, axis=1)  # 存放所有切片的类别平均dice
    hsdf_slice_class_mean_arr = np.mean(hsdf_slice_arr, axis=0)  # 存放所有切片的类别dice
    hsdf_mean = np.mean(hsdf_slice_allclass_mean_arr)
    hsdf_slice_mean_std = np.std(hsdf_slice_allclass_mean_arr)  # 所有切片的类别平均dice的方差
    hsdf_slice_std_lvm = np.std(hsdf_slice_arr[:, 0:1].squeeze())
    hsdf_slice_std_lv = np.std(hsdf_slice_arr[:, 1:2].squeeze())
    hsdf_slice_std_rv = np.std(hsdf_slice_arr[:, 2:3].squeeze())


    print('mean_dice: {:.3}±{:.3} - dice_lvm: {:.3}±{:.3} - dice_lv: {:.3}±{:.3} - dice_rv: {:.3}±{:.3}'
          .format(dice_mean, dice_slice_mean_std,
                  dice_slice_class_mean_arr[0], dice_slice_std_lvm,
                  dice_slice_class_mean_arr[1], dice_slice_std_lv,
                  dice_slice_class_mean_arr[2], dice_slice_std_rv))
    print('mean_jaccard: {:.3}±{:.3} - jaccard_lvm: {:.3}±{:.3} - jaccard_lv: {:.3}±{:.3} - jaccard_rv: {:.3}±{:.3}'
          .format(jaccard_mean, jaccard_slice_mean_std,
                  jaccard_slice_class_mean_arr[0], jaccard_slice_std_lvm,
                  jaccard_slice_class_mean_arr[1], jaccard_slice_std_lv,
                  jaccard_slice_class_mean_arr[2], jaccard_slice_std_rv))
    print('mean_hsdf: {:.3}±{:.3} - hsdf_lvm: {:.3}±{:.3} - hsdf_lv: {:.3}±{:.3} - hsdf_rv: {:.3}±{:.3}'
          .format(hsdf_mean, hsdf_slice_mean_std,
                  hsdf_slice_class_mean_arr[0], hsdf_slice_std_lvm,
                  hsdf_slice_class_mean_arr[1], hsdf_slice_std_lv,
                  hsdf_slice_class_mean_arr[2], hsdf_slice_std_rv))

if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    # val(net)
    auto_val(net)