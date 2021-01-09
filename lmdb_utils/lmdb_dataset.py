import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import lmdb
from caffe_pb2 import Datum
import random


class LMDBDataset(Dataset):
    def __init__(self, source_lmdbs, source_files):
        assert isinstance(source_files, list) or isinstance(source_files, tuple)
        assert isinstance(source_lmdbs, list) or isinstance(source_lmdbs, tuple)
        assert len(source_lmdbs) == len(source_files)
        assert len(source_files) > 0

        self.envs = []
        self.txns = []
        for lmdb_path in source_lmdbs:
            self.envs.append(lmdb.open(lmdb_path, max_readers=4, readonly=True, lock=False, readahead=False, meminit=False))
            self.txns.append(self.envs[-1].begin(write=False))

        self.train_list = []

        last_label = 0
        max_label = -1
        for db_id, lmdb_train_file in enumerate(source_files):
            with open(lmdb_train_file, 'r') as infile:
                for line in infile:
                    l = line.rstrip().lstrip()
                    if len(l) > 0:
                        lmdb_key, label = l.split(' ')
                        self.train_list.append([lmdb_key, int(label) + last_label, db_id])
                        max_label = max(max_label, int(label) + last_label)
            max_label += 1
            last_label = max_label


    def close(self):
        for i in range(len(self.txns)):
            self.txns[i].abort()
        for j in range(len(self.envs)):
            self.envs[j].close()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        lmdb_key, label, db_id = self.train_list[index]
        datum = Datum()
        raw_byte_buffer = self.txns[db_id].get(lmdb_key.encode('utf-8'))
        datum.ParseFromString(raw_byte_buffer)
        cv_img = cv2.imdecode(np.frombuffer(datum.data, dtype=np.uint8), -1)
        if random.random() < 0.5:
            cv_img = cv2.flip(cv_img, 1)

        if cv_img.ndim == 2:
            rows = cv_img.shape[0]
            cols = cv_img.shape[1]
            buf = np.zeros((3, rows, cols), dtype=np.uint8)
            buf[0] = buf[1] = buf[2] = cv_img
            input_tensor = (torch.from_numpy(buf) - 127.5) * 0.0078125
        else:
            assert cv_img.ndim == 3
            cv_img = np.transpose(cv_img, (2, 0, 1))
            input_tensor = (torch.from_numpy(cv_img) - 127.5) * 0.0078125

        return input_tensor, label
