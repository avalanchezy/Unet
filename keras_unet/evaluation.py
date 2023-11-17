#!/usr/bin/python3
import numpy as np
from tqdm import tqdm 
from medpy.metric.binary import hd, assd
from tensorflow.keras.utils import to_categorical
import numpy as np

def evaluate_segmentation(y_true, y_pred, voxel_spacing = [1., 1.]):
    """Compute Dices, Hausdorff distances and ASSD  (Average symmetric surface distance) between the predicted segmentation and the groundtruth"""
    # Preprocess the segmentation
    prediction = np.argmax(y_pred, axis=-1)
    prediction_bin = to_categorical(prediction)
    if prediction_bin.shape[2] < 6:
        layer = np.zeros((prediction_bin.shape[0], prediction_bin.shape[1], 6 - prediction_bin.shape[2]))
        prediction_bin = np.concatenate((prediction_bin, layer), axis=2)

    dice_num = 0
    dice_den = 0
    hausdorff = [0]
    assds = [None]
    dice = [0]

    for i in range(1, y_true.shape[2]):
        pred = prediction_bin[:,:,i]
        gt = y_true[:,:,i]

        # Transform into boolean array
        pred = (pred == 1)
        gt = (gt == 1)

        if (np.sum(pred) > 0) and (np.sum(gt) > 0):  # If the structure is predicted on at least one pixel
            h = hd(pred, gt, voxelspacing=voxel_spacing)
            a = assd(pred, gt, voxelspacing=voxel_spacing)
            dice_num += np.sum(pred[gt]) * 2.0
            dice_den += np.sum(pred[pred]) + np.sum(gt[gt])
            d = np.sum(pred[gt]) * 2.0 / (np.sum(pred[pred]) + np.sum(gt[gt]))

        else:
            h = None
            a = None
            d = None

        dice.append(d)
        hausdorff.append(h)
        assds.append(a)

    if dice_den>0:
        dice[0] = dice_num/dice_den
        hausdorff[0] = max(i for i in hausdorff if i is not None)
        assds[0] = sum(i for i in assds if i is not None) / sum(i is not None for i in assds)
    else:
        dice[0]=None
        hausdorff[0] = None
        assds[0] = None

    return dice, hausdorff, assds


def evaluate_set(set_y_true, set_y_pred, voxel_spacing = [1.0, 1.0]):
    s_d=np.zeros(set_y_true.shape[-1], dtype=np.float32)
    s_h=np.zeros(set_y_true.shape[-1], dtype=np.float32)
    s_a=np.zeros(set_y_true.shape[-1], dtype=np.float32)
    valid_lab=np.zeros(set_y_true.shape[-1], dtype=np.float32)

    for ind in tqdm(range(set_y_true.shape[0])):
        name =str(ind)
        if np.sum(set_y_true[ind, :, :, 1:])!=0:
            [d, h, a] = evaluate_segmentation(y_true=set_y_true[ind, :, :, :],
                                                  y_pred=set_y_pred[ind, :, :, :],
                                                  voxel_spacing=voxel_spacing)
            for lab in range(set_y_true.shape[-1]):
                if d[lab] is not None:
                    s_d[lab] += d[lab]
                    s_h[lab] += h[lab]
                    s_a[lab] += a[lab]
                    valid_lab[lab] += 1
    return s_d/valid_lab, s_h/valid_lab, s_a/valid_lab, valid_lab
