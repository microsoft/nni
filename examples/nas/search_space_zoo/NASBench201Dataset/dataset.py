import copy
import json
import logging
import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Nb201Dataset(Dataset):
    """Modified from NAS-Bench-201 API"""
    NUM_OP_CANDIDATES = 5
    MEAN = {
        "cifar10-valid": 0.8365059831110725,
        "cifar10": 0.8704242346666666,
        "cifar100": 0.6125050707032396,
        "imagenet-16-120": 0.3376692831133674
    }
    STD = {
        "cifar10-valid": 0.12823611180535768,
        "cifar10": 0.12939587274278727,
        "cifar100": 0.12148740127516067,
        "imagenet-16-120": 0.09261513033735617
    }

    @staticmethod
    def get_available_datasets():
        return ["cifar10-valid", "cifar10", "cifar100", "imagenet-16-120"]

    @staticmethod
    def get_available_splits():
        with open("data/nb201/available_splits.json") as f:
            return json.load(f)

    def __init__(self, dataset, split, data_dir="data/nb201",
                 acc_normalize=True, ops_onehot=True, input_dtype=np.float32):
        self.dataset = dataset
        self.acc_normalize = acc_normalize
        self.ops_onehot = ops_onehot
        self.input_dtype = input_dtype
        with h5py.File(os.path.join(data_dir, "nb201.hdf5"), mode="r") as f:
            self.matrix = f["matrix"][()]
            self.metrics = f[dataset][()]
        with open(os.path.join(data_dir, "splits.pkl"), "rb") as f:
            pkl = pickle.load(f)
            if split not in pkl:
                raise KeyError("'%s' not in splits. Available splits are: %s" % (split, pkl.keys()))
            self.samples = pkl[split]
        self.seed = 0

    def normalize(self, num):
        return (num - self.MEAN[self.dataset]) / self.STD[self.dataset]

    def denormalize(self, num):
        return num * self.STD[self.dataset] + self.MEAN[self.dataset]

    def get_all_metrics(self, split=None, average=False):
        if split == "val":
            metrics_index = 3
        elif split == "test":
            metrics_index = 5
        else:
            metrics_index = [3, 5]
        data = self.metrics[self.samples, :, :][:, :, metrics_index]
        data = data.reshape((data.shape[0], -1))
        if average:
            data = np.mean(data, axis=1)
        return data

    def __getitem__(self, index):
        return self.retrieve_by_arch_id(self.samples[index])

    def retrieve_by_arch_id(self, arch_id):
        matrix = self.matrix[arch_id]
        num_nodes = matrix.shape[0]
        if self.ops_onehot:
            onehot_matrix = np.zeros((num_nodes, num_nodes, self.NUM_OP_CANDIDATES), dtype=self.input_dtype)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    onehot_matrix[i, j, matrix[i][j]] = 1
            matrix = onehot_matrix
        val_acc = self.metrics[arch_id, self.seed, 3]
        test_acc = self.metrics[arch_id, self.seed, 5]
        if self.acc_normalize:
            val_acc, test_acc = self.normalize(val_acc), self.normalize(test_acc)
        return {
            "matrix": matrix,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "flops": self.metrics[arch_id, self.seed, 6],
            "params": self.metrics[arch_id, self.seed, 7],
            "latency": self.metrics[arch_id, self.seed, 8]
        }

    def __len__(self):
        return len(self.samples)
