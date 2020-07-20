from __future__ import print_function
import os
import os.path
import errno
import torch
import numpy as np
import sys
import cv2
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(class_file):
    with open(class_file) as r:
        classes = map(lambda s: s.strip(), r.readlines())

    # classes.sort()
    # class_to_idx = {classes[i]: i for i in range(len(classes))}

    class_to_idx = {iclass: i for i, iclass in enumerate(classes)}

    return classes, class_to_idx


def loadPILImage(path):
    trans_img = Image.open(path).convert('RGB')
    return trans_img

def loadCVImage(path):
    img = cv2.imread(path,  cv2.IMREAD_COLOR)
    trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(trans_img.astype('uint8'), 'RGB')

def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                imgfnames = sorted(os.listdir(cls_fpath))[:250]
                for imgname in imgfnames:
                    if is_image_file(imgname):
                        path = os.path.join(cls_fpath, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
    elif dirname == 'val':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                imgfnames = sorted(os.listdir(cls_fpath))[250:350]
                for imgname in imgfnames:
                    if is_image_file(imgname):
                        path = os.path.join(cls_fpath, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)

    return images

class NewImageNet(data.Dataset):

    base_folder = 'new_dataset'
    def __init__(self, root, train=True,
                 target_transform=None,
                 test=False, loader='opencv'):
        self.root = os.path.expanduser(root)
        if train:
            self.transform = transforms.Compose([
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.loader = loader

        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'info.txt'))
        # self.classes = classes

        if self.train:
            dirname = 'train'
        else:
            dirname = 'val'

        self.class_to_idx = class_to_idx
        self.idx_to_class = dict()
        for idx, key in enumerate(class_to_idx.keys()):
            self.idx_to_class[idx] = key

        self.data_info = make_dataset(self.root, self.base_folder, dirname, class_to_idx)

        if len(self.data_info) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img_path, target) where target is index of the target class.
        """

        img_path, target = self.data_info[index][0], self.data_info[index][1]

        if self.loader == 'pil':
            img = loadPILImage(img_path)
        else:
            img = loadCVImage(img_path)

        if self.transform is not None:
            result_img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return result_img, target

    def __len__(self):
        return len(self.data_info)


def get_newimagenet(dir, batch_size):
    train_data = NewImageNet(root=dir, train=True)
    test_data = NewImageNet(root=dir, train=False)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True, num_workers=16)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=16)

    return [train_loader, test_loader], [train_sampler, test_sampler]


