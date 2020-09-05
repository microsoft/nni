import math
import os

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch.utils.data
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class HybridTrainPipe(Pipeline):
    def __init__(self, args, batch_size, image_size, rank, world_size):
        device_id = torch.cuda.current_device()
        enable_gpu = args.enable_gpu_dataloader
        super(HybridTrainPipe, self).__init__(batch_size, args.num_threads, device_id, seed=args.seed + rank)
        self.input = ops.FileReader(file_root=os.path.join(args.imagenet_dir, "train"),
                                    shard_id=rank, num_shards=world_size,
                                    random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed" if enable_gpu else "cpu")
        self.res = ops.RandomResizedCrop(device="gpu" if enable_gpu else "cpu",
                                         size=image_size,
                                         mag_filter=types.DALIInterpType.INTERP_LINEAR,
                                         min_filter=types.DALIInterpType.INTERP_TRIANGULAR)
        self.brightness = ops.BrightnessContrast(device="gpu" if enable_gpu else "cpu")
        self.hsv = ops.Hsv(device="gpu" if enable_gpu else "cpu")
        self.rng_0400 = ops.Uniform(range=[0.6, 1.4])

        self.cmnp = ops.CropMirrorNormalize(device="gpu" if enable_gpu else "cpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=MEAN,
                                            std=STD)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.brightness(images, brightness=self.rng_0400(), contrast=self.rng_0400())
        images = self.hsv(images, saturation=self.rng_0400())
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, args, batch_size, image_size, rank, world_size):
        device_id = torch.cuda.current_device()
        enable_gpu = args.enable_gpu_dataloader
        super(HybridValPipe, self).__init__(batch_size, args.num_threads, device_id, seed=args.seed + rank)
        self.input = ops.FileReader(file_root=os.path.join(args.imagenet_dir, "val"),
                                    shard_id=rank, num_shards=world_size,
                                    random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed" if enable_gpu else "cpu")
        self.res = ops.Resize(device="gpu" if enable_gpu else "cpu",
                              resize_shorter=math.ceil(image_size / 0.875),
                              mag_filter=types.DALIInterpType.INTERP_LINEAR,
                              min_filter=types.DALIInterpType.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu" if enable_gpu else "cpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(image_size, image_size),
                                            mean=MEAN,
                                            std=STD)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class _ClassificationWrapper:
    def __init__(self, pipeline, batch_size, world_size, drop_last, infinite):
        self.infinite = infinite
        self.num_samples = pipeline.epoch_size("Reader") // world_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.loader = DALIClassificationIterator(pipeline, size=self.num_samples,
                                                 fill_last_batch=False, auto_reset=True)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                data = next(self.loader)
                images, labels = data[0]["data"].cuda(), data[0]["label"].view(-1).long().cuda()
                if self.drop_last and labels.size(0) < self.batch_size:
                    continue
                return images, labels
            except StopIteration:
                if self.infinite:
                    continue
                raise StopIteration

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


def imagenet_loader_gpu(args, split, batch_size=None, image_size=224, infinite=False, distributed=True):
    if distributed:
        rank, world_size = args.rank, args.world_size
    else:
        rank, world_size = 0, 1
    if batch_size is None:
        batch_size = args.batch_size
    if split == "train":
        pipeline = HybridTrainPipe(args, batch_size, image_size, rank, world_size)
    elif split == "val":
        pipeline = HybridValPipe(args, batch_size, image_size, rank, world_size)
    else:
        raise AssertionError
    pipeline.build()
    return _ClassificationWrapper(pipeline, batch_size, world_size, split == "train", infinite)
    #return _ClassificationWrapper(pipeline, batch_size, world_size, True, infinite)




from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

class ImageNet(Dataset):

    def __init__(self, root, file_list, train=True, image_size=224,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.root = os.path.expanduser(root)
        traindir = os.path.join(self.root, 'train')
        valdir = os.path.join(self.root, 'val')
        self.train = train
        self.directory = self.train and traindir or valdir
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.samples = []
        self.loader = default_loader
        with open(file_list, "r") as f:
            for line in f.readlines():
                filename, label = line.strip().split()
                label = int(label)
                self.samples.append((filename, label))
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(self.image_size / 0.875), int(self.image_size / 0.875))),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(os.path.join(self.directory, path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)
    
def torch_dataloader(args, train=True):
    dataset = ImageNet(args.imagenet_dir, args.file_list, train=True)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_threads, pin_memory=True)
    return dataloader