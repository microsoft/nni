# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import cv2
import os
import random

import numpy as np

from torch.utils import data


def flip(img, annotation):
    """ Flip the image. """
    img = np.fliplr(img).copy()
    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[(4 + 1)::2]

    for i in range(len(landmark_x)):
        landmark_x[i] = w - landmark_x[i]

    new_annotation = list()
    new_annotation.append(x_min)
    new_annotation.append(y_min)
    new_annotation.append(x_max)
    new_annotation.append(y_max)

    for i in range(len(landmark_x)):
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return img, new_annotation


def channel_shuffle(img, annotation):
    """ Channel shuffle. """
    if img.shape[2] == 3:
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]

    return img, annotation


def random_noise(img, annotation, limit=[0, 0.2], p=0.5):
    """ Add the random noise to the image. """
    if random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_brightness(img, annotation, brightness=0.3):
    """ Change the brightness randomly. """
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * img
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_contrast(img, annotation, contrast=0.3):
    """ Change the contrast randomly. """
    coef = np.array([[[0.114, 0.587, 0.299]]])
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_saturation(img, annotation, saturation=0.5):
    """ Change the saturation randomly. """
    coef = np.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_hue(image, annotation, hue=0.5):
    """ Change the hue randomly. """
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image, annotation


def scale(img, annotation):
    """ Resize the image randomly. """
    f_xy = np.random.uniform(-0.4, 0.8)
    origin_h, origin_w = img.shape[:2]

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[(4 + 1)::2]

    h, w = int(origin_h * f_xy), int(origin_w * f_xy)
    image = cv2.resize(img, (h, w)).astype(np.uint8)

    new_annotation = list()
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * f_xy
        new_annotation.append(bbox[i])

    for i in range(len(landmark_x)):
        landmark_x[i] = landmark_x[i] * f_xy
        landmark_y[i] = landmark_y[i] * f_xy
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return image, new_annotation


def rotate(img, annotation, alpha=30):
    """ Rotate the image. """
    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[(4 + 1)::2]
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    new_point_x = list()
    new_point_y = list()
    for (x, y) in zip(landmark_x, landmark_y):
        point_x = rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2]
        point_y = rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]
        new_point_x.append(point_x)
        new_point_y.append(point_y)

    new_annotation = list()
    new_annotation.append(min(new_point_x))
    new_annotation.append(min(new_point_y))
    new_annotation.append(max(new_point_x))
    new_annotation.append(max(new_point_y))

    for (x, y) in zip(landmark_x, landmark_y):
        annotation = rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2]
        new_annotation.append(annotation)
        annotation = rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]
        new_annotation.append(annotation)

    return img_rotated, new_annotation


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
