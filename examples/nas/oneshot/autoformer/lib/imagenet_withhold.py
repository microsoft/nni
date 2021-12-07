
from PIL import Image
import io

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageNet_Withhold(Dataset):
    def __init__(self, data_root, ann_file='', transform=None, train=True, task ='train'):
        super(ImageNet_Withhold, self).__init__()
        ann_file = ann_file + '/' + 'val_true.txt'
        train_split  = (task == 'train' or  task == 'val')
        self.data_root = data_root + '/'+ ('train' if train_split else 'val')

        self.data = []
        self.nb_classes = 0
        folders = {}
        cnt = 0
        self.z = ZipReader()
        # if train:
        #     for member in self.tarfile.getmembers():
        #         print(member)
        # self.tarfile = tarfile.open(self.data_root)

        f = open(ann_file)
        prefix =  'data/sdb/imagenet'+'/'+ ('train' if train_split else 'val') + '/'
        for line in f:
            tmp = line.strip().split('\t')[0]
            class_pic = tmp.split('/')
            class_tmp = class_pic[0]
            pic = class_pic[1]

            if class_tmp in folders:
                # print(self.tarfile.getmember(('train/' if train else 'val/') + tmp[0] + '.JPEG'))
                self.data.append((class_tmp + '.zip', prefix + tmp + '.JPEG', folders[class_tmp]))
            else:
                folders[class_tmp] = cnt
                cnt += 1
                self.data.append((class_tmp + '.zip', prefix + tmp + '.JPEG',folders[class_tmp]))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if transform is not None:
            self.transforms = transform
        else:
            if train:
                self.transforms = transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])


        self.nb_classes = cnt
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # print('extract_file', time.time()-start_time)
        iob = self.z.read(self.data_root + '/' + self.data[idx][0], self.data[idx][1])
        iob = io.BytesIO(iob)
        img = Image.open(iob).convert('RGB')
        target = self.data[idx][2]
        if self.transforms is not None:
            img = self.transforms(img)
        # print('open', time.time()-start_time)
        return img, target
