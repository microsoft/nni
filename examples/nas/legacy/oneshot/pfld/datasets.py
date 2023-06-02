# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import cv2
import os

import numpy as np

from torch.utils import data


class PFLDDatasets(data.Dataset):
    """ Dataset to manage the data loading, augmentation and generation. """

    def __init__(self, file_list, transforms=None, data_root="", img_size=112):
        """
        Parameters
        ----------
        file_list : list
            a list of file path and annotations
        transforms : function
            function for data augmentation
        data_root : str
            the root path of dataset
        img_size : int
            the size of image height or width
        """
        self.line = None
        self.path = None
        self.img_size = img_size
        self.land = None
        self.angle = None
        self.data_root = data_root
        self.transforms = transforms
        with open(file_list, "r") as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        """ Get the data sample and labels with the index. """
        self.line = self.lines[index].strip().split()
        # load image
        if self.data_root:
            self.img = cv2.imread(os.path.join(self.data_root, self.line[0]))
        else:
            self.img = cv2.imread(self.line[0])
        # resize
        self.img = cv2.resize(self.img, (self.img_size, self.img_size))
        # obtain gt labels
        self.land = np.asarray(self.line[1: (106 * 2 + 1)], dtype=np.float32)
        self.angle = np.asarray(self.line[(106 * 2 + 1):], dtype=np.float32)

        # augmentation
        if self.transforms:
            self.img = self.transforms(self.img)

        return self.img, self.land, self.angle

    def __len__(self):
        """ Get the size of dataset. """
        return len(self.lines)
