import torch
import torch.nn as nn
import numpy as np
import math
import scipy.spatial
import scipy.ndimage.morphology

"""
True Positive （真正， TP）预测为正的正样本
True Negative（真负 , TN）预测为负的负样本 
False Positive （假正， FP）预测为正的负样本
False Negative（假负 , FN）预测为负的正样本
"""


def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(predict, 1)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(predict, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


def diceCoeff(pred, gt, smooth=1e-5, ):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    score = (2 * intersection + smooth) / (unionset + smooth)

    return score.sum() / N

def diceFlat(pred, gt, smooth=1e-5):
    intersection = ((pred * gt).sum()).item()

    unionset = (pred.sum() + gt.sum()).item()
    score = (2 * intersection + smooth) / (unionset + smooth)
    return score


def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N


def diceCoeffv3(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    # 转为float，以防long类型之间相除结果为0
    score = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

    return score.sum() / N


def jaccard(pred, gt, eps=1e-5):
    """TP / (TP + FP + FN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp.float() + eps) / ((tp + fp + fn).float() + eps)
    return score.sum() / N


def jaccardFlat(pred, gt, eps=1e-5):
    pred_flat = pred.squeeze()
    gt_flat = gt.squeeze()
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))
    score = (tp.float() + eps) / ((tp + fp + fn).float() + eps)
    return score


def jaccardv2(pred, gt, eps=1e-5):
    """TP / (TP + FP + FN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp

    score = (tp + eps).float() / (tp + fp + fn + eps).float()
    return score.sum() / N


def tversky(pred, gt, eps=1e-5, alpha=0.7):
    """TP / (TP + (1-alpha) * FP + alpha * FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (tp + eps) / (tp + (1 - alpha) * fp + alpha * fn + eps)
    return score.sum() / N


def accuracy(pred, gt, eps=1e-5):
    """(TP + TN) / (TP + FP + FN + TN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = ((tp + tn).float() + eps) / ((tp + fp + tn + fn).float() + eps)

    return score.sum() / N


def precision(pred, gt, eps=1e-5):
    """TP / (TP + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))

    score = (tp.float() + eps) / ((tp + fp).float() + eps)

    return score.sum() / N


def sensitivity(pred, gt, eps=1e-5):
    """TP / (TP + FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp.float() + eps) / ((tp + fn).float() + eps)

    return score.sum() / N


def specificity(pred, gt, eps=1e-5):
    """TN / (TN + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))

    score = (tn.float() + eps) / ((fp + tn).float() + eps)

    return score.sum() / N


def recall(pred, gt, eps=1e-5):
    return sensitivity(pred, gt)


class Surface(object):
    # The edge points of the mask object.
    __mask_edge_points = None
    # The edge points of the reference object.
    __reference_edge_points = None
    # The nearest neighbours distances between mask and reference edge points.
    __mask_reference_nn = None
    # The nearest neighbours distances between reference and mask edge points.
    __reference_mask_nn = None
    # Distances of the two objects surface points.
    __distance_matrix = None

    def __init__(self, mask, reference, physical_voxel_spacing=[1, 1, 1], mask_offset=[0, 0, 0],
                 reference_offset=[0, 0, 0], connectivity=1):
        self.connectivity = connectivity
        # compute edge images
        mask_edge_image = self.compute_contour(mask)
        reference_edge_image = self.compute_contour(reference)
        mask_pts = mask_edge_image.nonzero()
        mask_edge_points = list(zip(mask_pts[0], mask_pts[1], mask_pts[2]))
        reference_pts = reference_edge_image.nonzero()
        reference_edge_points = list(zip(reference_pts[0], reference_pts[1], reference_pts[2]))
        # check if there is actually an object present
        if 0 >= len(mask_edge_points):
            raise Exception('The mask image does not seem to contain an object.')
        if 0 >= len(reference_edge_points):
            raise Exception('The reference image does not seem to contain an object.')
        # add offsets to the voxels positions and multiply with physical voxel spacing
        # to get the real positions in millimeters
        physical_voxel_spacing = scipy.array(physical_voxel_spacing)
        mask_edge_points = scipy.array(mask_edge_points, dtype='float64')
        mask_edge_points += scipy.array(mask_offset)
        mask_edge_points *= physical_voxel_spacing
        reference_edge_points = scipy.array(reference_edge_points, dtype='float64')
        reference_edge_points += scipy.array(reference_offset)
        reference_edge_points *= physical_voxel_spacing
        # set member vars
        self.__mask_edge_points = mask_edge_points
        self.__reference_edge_points = reference_edge_points

    def get_maximum_symmetric_surface_distance(self):
        # Get the maximum of the nearest neighbour distances
        A_B_distance = self.get_mask_reference_nn().max()
        B_A_distance = self.get_reference_mask_nn().max()
        # compute result and return
        return min(A_B_distance, B_A_distance)

    def get_root_mean_square_symmetric_surface_distance(self):
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())
        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()
        # square the distances
        A_B_distances_sqrt = A_B_distances * A_B_distances
        B_A_distances_sqrt = B_A_distances * B_A_distances
        # sum the minimal distances
        A_B_distances_sum = A_B_distances_sqrt.sum()
        B_A_distances_sum = B_A_distances_sqrt.sum()
        # compute result and return
        return math.sqrt(1. / (mask_surface_size + reference_surface_sice)) * math.sqrt(
            A_B_distances_sum + B_A_distances_sum)

    def get_average_symmetric_surface_distance(self):
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())
        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()
        # sum the minimal distances
        A_B_distances = A_B_distances.sum()
        B_A_distances = B_A_distances.sum()
        # compute result and return
        return 1. / (mask_surface_size + reference_surface_sice) * (A_B_distances + B_A_distances)

    def get_mask_reference_nn(self):
        # Note: see note for @see get_reference_mask_nn
        if None == self.__mask_reference_nn:
            tree = scipy.spatial.cKDTree(self.get_mask_edge_points())
            self.__mask_reference_nn, _ = tree.query(self.get_reference_edge_points())
        return self.__mask_reference_nn

    def get_reference_mask_nn(self):
        if self.__reference_mask_nn is None:
            tree = scipy.spatial.cKDTree(self.get_reference_edge_points())
            self.__reference_mask_nn, _ = tree.query(self.get_mask_edge_points())
        return self.__reference_mask_nn

    def get_mask_edge_points(self):
        return self.__mask_edge_points

    def get_reference_edge_points(self):
        return self.__reference_edge_points

    def compute_contour(self, array):
        footprint = scipy.ndimage.morphology.generate_binary_structure(array.ndim, self.connectivity)
        # create an erode version of the array
        erode_array = scipy.ndimage.morphology.binary_erosion(array, footprint)
        array = array.astype(bool)
        # xor the erode_array with the original and return
        return array ^ erode_array


if __name__ == '__main__':
    # shape = torch.Size([2, 3, 4, 4])
    # 模拟batch_size = 2
    '''
    1 0 0= bladder
    0 1 0 = tumor
    0 0 1= background 
    '''
    pred = torch.Tensor([[
        [[0, 1, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 1, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 0, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 1, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])

    gt = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])

    dice1 = diceCoeff(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    dice2 = jaccard(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    dice3 = diceCoeffv3(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    print(dice1, dice2, dice3)
