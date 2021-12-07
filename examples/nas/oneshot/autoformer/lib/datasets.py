import os
import cv2
import json
import torch
import scipy
import scipy.io as sio
from skimage import io

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class Flowers(ImageFolder):
    def __init__(self, root, train=True, transform=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        label_path = os.path.join(root, 'imagelabels.mat')
        split_path = os.path.join(root, 'setid.mat')

        print('Dataset Flowers is trained with resolution 224!')

        # labels
        labels = sio.loadmat(label_path)['labels'][0]
        self.img_to_label = dict()
        for i in range(len(labels)):
            self.img_to_label[i] = labels[i]

        splits = sio.loadmat(split_path)
        self.trnid, self.valid, self.tstid = sorted(splits['trnid'][0].tolist()), \
                                             sorted(splits['valid'][0].tolist()), \
                                             sorted(splits['tstid'][0].tolist())
        if train:
            self.imgs = self.trnid + self.valid
        else:
            self.imgs = self.tstid

        self.samples = []
        for item in self.imgs:
            self.samples.append((os.path.join(root, 'jpg', "image_{:05d}.jpg".format(item)), self.img_to_label[item-1]-1))

class Cars196(ImageFolder, datasets.CIFAR10):
    base_folder_devkit = 'devkit'
    base_folder_trainims = 'cars_train'
    base_folder_testims = 'cars_test'

    filename_testanno = 'cars_test_annos.mat'
    filename_trainanno = 'cars_train_annos.mat'

    base_folder = 'cars_train'
    train_list = [
        ['00001.jpg', '8df595812fee3ca9a215e1ad4b0fb0c4'],
        ['00002.jpg', '4b9e5efcc3612378ec63a22f618b5028']
    ]
    test_list = []
    num_training_classes = 98 # 196/2

    def __init__(self, root, train=False, transform=None, target_transform=None, **kwargs):
        self.root = root
        self.transform = transform

        self.target_transform = target_transform
        self.loader = default_loader
        print('Dataset Cars196 is trained with resolution 224!')

        self.samples = []
        self.nb_classes = 196

        if train:
            labels = \
            sio.loadmat(os.path.join(self.root, self.base_folder_devkit, self.filename_trainanno))['annotations'][0]
            for item in labels:
                img_name = item[-1].tolist()[0]
                label = int(item[4]) - 1
                self.samples.append((os.path.join(self.root, self.base_folder_trainims, img_name), label))
        else:
            labels = \
            sio.loadmat(os.path.join(self.root, 'cars_test_annos_withlabels.mat'))['annotations'][0]
            for item in labels:
                img_name = item[-1].tolist()[0]
                label = int(item[-2]) - 1
                self.samples.append((os.path.join(self.root, self.base_folder_testims, img_name), label))

class Pets(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        train_list_path = os.path.join(self.dataset_root, 'annotations', 'trainval.txt')
        test_list_path = os.path.join(self.dataset_root, 'annotations', 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, 'images', "{}.jpg".format(img_name)), label-1))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, 'images', "{}.jpg".format(img_name)), label-1))

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CARS':
        dataset = Cars196(args.data_path, train=is_train, transform=transform)
        nb_classes = 196
    elif args.data_set == 'PETS':
        dataset = Pets(args.data_path, train=is_train, transform=transform)
        nb_classes = 37
    elif args.data_set == 'FLOWERS':
        dataset = Flowers(args.data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'EVO_IMNET':
        root = os.path.join(args.data_path, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
