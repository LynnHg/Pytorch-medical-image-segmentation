# 通用包
import time
import os
import torch
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

# 用户包
from datasets import mscmr2019
import utils.image_transforms as joint_transforms
import utils.transforms as extended_transforms
from utils.loss import *
from utils.metrics import diceCoeffv2
from utils import misc
from utils.pytorchtools import EarlyStopping

# 超参设置
crop_size = 160
batch_size = 6
n_epoch = 300
lr_scheduler_eps = 1e-3
lr_scheduler_patience = 10
early_stop_patience = 12
initial_lr = 1e-4
threshold_lr = 1e-6
weight_decay = 1e-5
optimizer_type = 'adam'  # adam, sgd
scheduler_type = 'no'  # ReduceLR, StepLR, poly
label_smoothing = 0.01
aux_loss = False
gamma = 0.5
alpha = 0.85
model_number = random.randint(1, 1e6)


def main(Baseline, fold, loss_name, train_path, val_path):
    # 定义网络
    net = Baseline(num_classes=mscmr2019.num_classes).cuda()

    # 数据预处理，加载
    center_crop = joint_transforms.CenterCrop(crop_size)
    input_transform = extended_transforms.NpyToTensor()
    target_transform = extended_transforms.MaskToTensor()

    train_set = mscmr2019.MSCMR2019(train_path, 'train', fold, joint_transform=None, roi_crop=None,
                                    center_crop=center_crop,
                                    transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_set = mscmr2019.MSCMR2019(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, roi_crop=None,
                                  center_crop=center_crop,
                                  target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 定义损失函数
    if loss_name == 'dice':
        criterion = SoftDiceLoss(mscmr2019.num_classes).cuda()

    # 定义早停机制
    early_stopping = EarlyStopping(early_stop_patience, verbose=True, delta=lr_scheduler_eps,
                                   path=os.path.join(root_path, 'checkpoint', '{}.pth'.format(model_name)))

    # 定义优化器
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    train(train_loader, val_loader, net, criterion, optimizer, None, None, early_stopping, n_epoch, 0)


def train(train_loader, val_loader, net, criterion, optimizer, scheduler, warm_scheduler, early_stopping, num_epoches,
          iters):
    for epoch in range(1, num_epoches + 1):
        st = time.time()
        train_class_dices = np.array([0] * (mscmr2019.num_classes - 1), dtype=np.float)
        val_class_dices = np.array([0] * (mscmr2019.num_classes - 1), dtype=np.float)
        val_dice_arr = []
        train_losses = []
        val_losses = []

        # 训练模型
        net.train()
        for batch, (input, mask, _, __) in enumerate(train_loader, 1):
            X1 = input.cuda()
            y = mask.cuda()
            optimizer.zero_grad()
            output = net(X1)
            output = torch.sigmoid(output)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            iters += 1
            train_losses.append(loss.item())

            class_dice = []
            for i in range(1, mscmr2019.num_classes):
                cur_dice = diceCoeffv2(output[:, i:i + 1, :], y[:, i:i + 1, :]).cpu().item()
                class_dice.append(cur_dice)

            mean_dice = sum(class_dice) / len(class_dice)
            train_class_dices += np.array(class_dice)
            string_print = 'epoch: {} - iters: {} - loss: {:.4} - mean: {:.4} - lvm: {:.4}- lv: {:.4} - rv: {:.4} - time: {:.2}' \
                .format(epoch, iters, loss.data.cpu(), mean_dice, class_dice[0], class_dice[1], class_dice[2], time.time() - st)
            misc.log(string_print)
            st = time.time()

        train_loss = np.average(train_losses)
        train_class_dices = train_class_dices / batch
        train_mean_dice = train_class_dices.sum() / train_class_dices.size

        writer.add_scalar('main_loss', train_loss, epoch)
        writer.add_scalar('main_dice', train_mean_dice, epoch)

        print('epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4} - dice_lvm: {:.4} - dice_lv: {:.4} - dice_rv: {:.4}'.format(
                epoch, num_epoches, train_loss, train_mean_dice, train_class_dices[0], train_class_dices[1], train_class_dices[2]))

        # 验证模型
        net.eval()
        for val_batch, (input, mask, _, __) in enumerate(val_loader, 1):
            val_X1 = input.cuda()
            val_y = mask.cuda()
            pred = net(val_X1)
            pred = torch.sigmoid(pred)
            val_loss = criterion(pred, val_y)
            val_losses.append(val_loss.item())
            pred = pred.cpu().detach()
            val_class_dice = []
            for i in range(1, mscmr2019.num_classes):
                val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

            val_dice_arr.append(val_class_dice)
            val_class_dices += np.array(val_class_dice)

        val_loss = np.average(val_losses)

        val_dice_arr = np.array(val_dice_arr)
        std = (np.std(val_dice_arr[:, 1:2]) + np.std(val_dice_arr[:, 2:3]) + np.std(val_dice_arr[:, 3:4])) / mscmr2019.num_classes
        val_class_dices = val_class_dices / val_batch

        val_mean_dice = val_class_dices.sum() / val_class_dices.size
        organ_mean_dice = (val_class_dices[0] + val_class_dices[1] + val_class_dices[2]) / mscmr2019.num_classes

        val_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        val_writer.add_scalar('main_loss', val_loss, epoch)
        val_writer.add_scalar('main_dice', val_mean_dice, epoch)
        val_writer.add_scalar('lesion_dice', organ_mean_dice, epoch)

        print('val_loss: {:.4} - val_mean_dice: {:.4} - mean: {:.4}±{:.3} - lvm: {:.4}- lv: {:.4} - rv: {:.4}'
            .format(val_loss, val_mean_dice, organ_mean_dice, std, val_class_dices[0], val_class_dices[1], val_class_dices[2]))
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        early_stopping(organ_mean_dice, net, epoch)
        if early_stopping.early_stop or optimizer.param_groups[0]['lr'] < threshold_lr:
            print("Early stopping")
            # 结束模型训练
            break

    print('----------------------------------------------------------')
    print('save epoch {}'.format(early_stopping.save_epoch))
    print('stoped epoch {}'.format(epoch))
    print('----------------------------------------------------------')


if __name__ == '__main__':
    # 5折交叉验证训练模型
    augdata = ["C0", "LGE", "T2"]
    folds = [1, 2, 3, 4, 5]  # 1, 2, 3, 4, 5
    dataset_name = 'MSCMR2019'

    model_type = "unet"
    if model_type == "unet":
        from networks.unet import Baseline
    elif model_type == "fcn":
        from networks.fcn import Baseline
    elif model_type == "segnet":
        from networks.segnet import Baseline
    elif model_type == "attunet":
        from networks.attunet import Baseline

    root_path = '../'

    for modal in augdata:
        for fold in folds:
            loss_name = 'dice'  # dice, bce, wbce, dual, wdual
            reduction = '' + modal
            model_name = '{}_fold{}_{}_{}_{}'.format(model_type, fold, loss_name, reduction, model_number)

            writer = SummaryWriter(
                os.path.join(root_path, 'log/{}/train'.format(dataset_name),
                             model_name + '_{}fold'.format(fold) + str(int(time.time()))))
            val_writer = SummaryWriter(os.path.join(
                os.path.join(root_path, 'log/{}/val'.format(dataset_name), model_name) + '_{}fold'.format(fold) + str(
                    int(time.time()))))

            train_path = os.path.join(root_path, 'media/Datasets/{}/Augdata{}'.format(dataset_name, modal),
                                      'train{}'.format(fold),
                                      'npy')
            val_path = os.path.join(root_path, 'media/Datasets/{}'.format(dataset_name), modal, 'npy')

            main(Baseline=Baseline, fold=fold, loss_name=loss_name, train_path=train_path, val_path=val_path)

