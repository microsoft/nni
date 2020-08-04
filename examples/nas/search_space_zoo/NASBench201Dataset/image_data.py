import json

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from .downsampled_imagenet import ImageNet16


def get_datasets(configs):
    if configs.dataset.startswith("cifar100"):
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif configs.dataset.startswith("cifar10"):  # cifar10 is a prefix of cifar100
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif configs.dataset.startswith('imagenet-16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
    else:
        raise NotImplementedError

    normalization = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if configs.dataset.startswith("cifar10") or configs.dataset.startswith("cifar100"):
        augmentation = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
    elif configs.dataset.startswith("imagenet-16"):
        augmentation = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2)]
    train_transform = transforms.Compose(augmentation + normalization)
    test_transform = transforms.Compose(normalization)

    if configs.dataset.startswith("cifar100"):
        train_data = datasets.CIFAR100("data/cifar100", train=True, transform=train_transform)
        valid_data = datasets.CIFAR100("data/cifar100", train=False, transform=test_transform)
        assert len(train_data) == 50000 and len(valid_data) == 10000
    elif configs.dataset.startswith("cifar10"):
        train_data = datasets.CIFAR10("data/cifar10", train=True, transform=train_transform)
        valid_data = datasets.CIFAR10("data/cifar10", train=False, transform=test_transform)
        assert len(train_data) == 50000 and len(valid_data) == 10000
    elif configs.dataset.startswith("imagenet-16"):
        num_classes = int(configs.dataset.split("-")[-1])
        train_data = ImageNet16("data/imagenet16", train=True, transform=train_transform, num_classes=num_classes)
        valid_data = ImageNet16("data/imagenet16", train=False, transform=test_transform, num_classes=num_classes)
        assert len(train_data) == 151700 and len(valid_data) == 6000
    return train_data, valid_data


def load_split(config_file_path, split_names):
    with open(config_file_path, "r") as f:
        data = json.load(f)
    result = []
    for name in split_names:
        _, arr = data[name]
        result.append(list(map(int, arr)))
    return result


def get_dataloader(configs):
    train_data, valid_data = get_datasets(configs)
    split_path = "data/nb201/split-{}.txt".format(configs.dataset)
    kwargs = {"batch_size": configs.batch_size, "num_workers": configs.num_threads}
    if configs.dataset == "cifar10-valid":
        train_split, valid_split = load_split(split_path, ["train", "valid"])
        train_loader = DataLoader(train_data, sampler=SubsetRandomSampler(train_split), drop_last=True, **kwargs)
        valid_loader = DataLoader(train_data, sampler=SubsetRandomSampler(valid_split), **kwargs)
        test_loader = DataLoader(valid_data, **kwargs)
    elif configs.dataset == "cifar10":
        train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **kwargs)
        valid_loader = DataLoader(valid_data, **kwargs)
        test_loader = DataLoader(valid_data, **kwargs)
    else:
        valid_split, test_split = load_split(split_path, ["xvalid", "xtest"])
        train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **kwargs)
        valid_loader = DataLoader(valid_data, sampler=SubsetRandomSampler(valid_split), **kwargs)
        test_loader = DataLoader(valid_data, sampler=SubsetRandomSampler(test_split), **kwargs)
    return train_loader, valid_loader, test_loader
