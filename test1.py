import torch
import torch.nn.functional as F

import numpy as np
import os, argparse, time
import imageio

from model.SEINet_EfficientNetB7 import SEINet
from utils.data import test_dataset

# ================= Metrics ==================
def Fmeasure(pred, gt):
    beta2 = 0.3
    pred_bin = (pred >= 0.5).astype(np.float32)
    gt = gt.astype(np.float32)

    tp = np.sum(pred_bin * gt)
    precision = tp / (np.sum(pred_bin) + 1e-8)
    recall = tp / (np.sum(gt) + 1e-8)
    f = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return f

def MAE_metric(pred, gt):
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))

def S_measure(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = gt.astype(np.float32)
    alpha = 0.5
    y = np.mean(gt)
    if y == 0:
        return 1 - np.mean(pred)
    elif y == 1:
        return np.mean(pred)
    else:
        Q = alpha * np.mean((pred - np.mean(pred)) * (gt - y)) / (np.std(pred) * np.std(gt) + 1e-8) + \
            (1 - alpha) * (1 - np.mean(np.abs(pred - gt)))
        return Q

def E_measure(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = gt.astype(np.float32)
    return 1 - np.mean(np.abs(pred - gt))

# ============================================

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = './datasets/'

model = SEINet()
model.load_state_dict(torch.load('/kaggle/input/seinet/pytorch/default/2/SEINet_EfficientNetB7_EORSSD.pth'))

model.cuda()
model.eval()

test_datasets = ['EORSSD']

for dataset in test_datasets:
    save_path = './results/EfficientNetB7/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = '/kaggle/input/eorssd/test-images/'
    gt_root = '/kaggle/input/eorssd/test-labels/'
    print("Dataset:", dataset)

    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0

    # Metric storage
    F_list, S_list, E_list, MAE_list = [], [], [], []

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        time_start = time.time()
        res, s1_sig, e1, s2, s2_sig, e2, s3, s3_sig, e3, s4, s4_sig, e4 = model(image)
        time_end = time.time()
        time_sum += (time_end - time_start)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # Save prediction
        imageio.imsave(save_path + name, (res * 255).astype(np.uint8))

        # Metrics
        F_list.append(Fmeasure(res, gt))
        S_list.append(S_measure(res, gt))
        E_list.append(E_measure(res, gt))
        MAE_list.append(MAE_metric(res, gt))

        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))

    # === Final averaged metrics ===
    print("\n=== Evaluation Results on {} ===".format(dataset))
    print("F-measure: {:.4f}".format(np.mean(F_list)))
    print("S-measure: {:.4f}".format(np.mean(S_list)))
    print("E-measure: {:.4f}".format(np.mean(E_list)))
    print("MAE: {:.4f}".format(np.mean(MAE_list)))
