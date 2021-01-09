import numpy as np
import cv2
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import os
import copy
import re
import logging as logger

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


def find_score(far, vr, target=1e-4):
    l = 0
    u = far.size - 1
    e = -1
    while u - l > 1:
        mid = (l + u) // 2
        if far[mid] == target:
            if target != 0:
                return vr[mid]
            else:
                e = mid
                break
        elif far[mid] < target:
            u = mid
        else:
            l = mid
    if target == 0:
        i = e
        while i >= 0:
            if far[i] != 0:
                break
            i -= 1
        if i >= 0:
            return vr[i + 1]
        else:
            return vr[u]
    if target != 0 and far[l] / target >= 8:
        return 0.0
    nearest_point = (target - far[l]) / (far[u] - far[l]) * (vr[u] - vr[l]) + vr[l]
    return nearest_point


def compute_roc(score, label, num_thresholds=1000, show_sample_hist=False):
    pos_dist = score[label == 1]
    neg_dist = score[label == 0]

    num_pos_samples = pos_dist.size
    num_neg_samples = neg_dist.size
    data_max = np.max(score)
    data_min = np.min(score)
    unit = (data_max - data_min) * 1.0 / num_thresholds
    threshold = data_min + (data_max - data_min) * np.array(range(1, num_thresholds + 1)) / num_thresholds
    new_interval = threshold - unit / 2.0 + 2e-6
    new_interval = np.append(new_interval, np.array(new_interval[-1] + unit))
    P = np.triu(np.ones(num_thresholds))

    pos_hist, dummy = np.histogram(pos_dist, new_interval)
    neg_hist, dummy2 = np.histogram(neg_dist, new_interval)
    pos_mat = pos_hist[:, np.newaxis]
    neg_mat = neg_hist[:, np.newaxis]

    assert pos_hist.size == neg_hist.size == num_thresholds
    far = np.dot(P, neg_mat) / num_neg_samples
    far = np.squeeze(far)
    vr = np.dot(P, pos_mat) / num_pos_samples
    vr = np.squeeze(vr)
    if show_sample_hist is False:
        return far, vr, threshold
    else:
        return far, vr, threshold, pos_hist, neg_hist


def test_lfw(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 600: (i + 1) * 600]
        test_score = score[i * 600: (i + 1) * 600]
        if i == 0:
            train_label = mask[600:]
            train_score = score[600:]
        elif i == 9:
            train_label = mask[:5400]
            train_score = score[:5400]
        else:
            train_label_1 = mask[:i * 600]
            train_label_2 = mask[(i + 1) * 600:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 600]
            train_score_2 = score[(i + 1) * 600:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(600):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 600
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean, std


def load_image(filename, color=True, mean=127.5, std=128.0):
    """
    Load an image && convert it to gray-scale or BGR image as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as ile False
        loads as intensity (if image is already gray-scale
    mean: pre-process, default is minus 127.5, divided by 128.0
    std: pre-process.

    Returns
    -------
    image : an image with type np.uint8 in range [0,255]
        of size (3 x H x W ) in BGR or
        of size (1 x H x W ) in gray-scale, if order == CHW
        else return H X W X 3 in BGR or H X W X 1 in gray-scale
    """
    order = 'CHW'
    assert order.upper() in ['CHW', 'HWC']
    if not os.path.exists(filename):
        raise Exception('%s does not exist.' % filename)

    flags = cv2.IMREAD_COLOR
    if color is False:
        flags = cv2.IMREAD_GRAYSCALE
    python_version = sys.version_info.major
    if python_version == 2:
        img = cv2.imread(filename, flags)
    elif python_version == 3:
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        raise Exception('Unknown python version.')
    if img.ndim == 2:
        assert color is False
        img = img[:, :, np.newaxis]
    if order.upper() == 'CHW':
        img = (img.transpose((2, 0, 1)) - mean) / std
    else:
        img = (img - mean) / std
    return img.astype(np.float32)


class LFWDataset(Dataset):
    def __init__(self, lfw_filename, lfw_root_dir):
        self.lfw_file_list = []
        with open(lfw_filename, 'r') as fin:
            for line in fin:
                l = line.rstrip().lstrip()
                if len(l) > 0:
                    self.lfw_file_list.append(l)
        self.lfw_root_dir = lfw_root_dir

    def __len__(self):
        return len(self.lfw_file_list)

    def __getitem__(self, index):
        filename = self.lfw_file_list[index]
        image = load_image(os.path.join(self.lfw_root_dir, filename))
        image = torch.from_numpy(image)
        return image, filename


def validate(model, lfw_test_pairs, test_data_loader, device):
    model.eval()
    filename2feat = {}
    with torch.no_grad():
        for batch_idx, (image, filenames) in enumerate(test_data_loader):
            image = image.to(device)
            feature = model(image).cpu().numpy()

            for fid, e in enumerate(filenames):
                assert e not in filename2feat
                filename2feat[e] = copy.deepcopy(feature[fid])

    score_list = []
    label_list = []
    for pairs in lfw_test_pairs:
        feat1 = filename2feat[pairs[0]]
        feat2 = filename2feat[pairs[1]]
        dist = np.dot(feat1, feat2) / np.sqrt(np.dot(feat1, feat1) * np.dot(feat2, feat2))
        score_list.append(dist)
        label_list.append(pairs[2])
    score = np.array(score_list)
    label = np.array(label_list)
    lfw_acc, lfw_std = test_lfw(label, score)
    model.train()
    return lfw_acc


def prepare_validate_data():
    # Common settings
    lfw_filename = './lfw_test/lfw_image_list.txt'
    lfw_root_dir = './lfw_test/lfw_cropped_images'
    lfw_pairs_file = './lfw_test/lfw_test_pairs.txt'
    lfw_test_pairs = []
    pat = re.compile(r'(\S+)\s+(\S+)\s+(\S+)')
    with open(lfw_pairs_file, 'r') as infile:
        for line in infile:
            l = line.rstrip()
            l = l.lstrip()
            if len(l) > 0:
                obj = pat.search(l)
                if obj:
                    file1 = obj.group(1)
                    file2 = obj.group(2)
                    label = int(obj.group(3))
                    lfw_test_pairs.append([file1, file2, label])
                else:
                    raise Exception('Cannot parse line %s, expected format: file1 file2 image_label' % l)
    test_data_loader = DataLoader(LFWDataset(lfw_filename, lfw_root_dir), batch_size=100, num_workers=4)
    return lfw_test_pairs, test_data_loader
