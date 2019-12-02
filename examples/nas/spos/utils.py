import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageNet


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def spos_to_bgr_tensor(pic):
    """Modified from `to_tensor`"""
    if not isinstance(pic, Image.Image):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img[:, :, [2, 1, 0]].transpose(0, 1).transpose(0, 2).contiguous()
    return img.float() if isinstance(img, torch.ByteTensor) else img


def get_imagenet(imagenet_root, spos_pre):
    if spos_pre:
        postprocess = [
            transforms.Lambda(lambda img: spos_to_bgr_tensor(img))
        ]
    else:
        postprocess = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(0.5),
    ] + postprocess)
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ] + postprocess)
    train_dataset = ImageNet(imagenet_root, split="train", transform=train_transform)
    valid_dataset = ImageNet(imagenet_root, split="val", transform=valid_transform)
    return train_dataset, valid_dataset


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def accuracy(output, target, topk=(1, 5)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res
