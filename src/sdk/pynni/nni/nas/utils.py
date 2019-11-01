import re
from collections import OrderedDict

import torch

_counter = 0


def global_mutable_counting():
    global _counter
    _counter += 1
    return _counter


def to_snake_case(camel_case):
    return re.sub('(?!^)([A-Z]+)', r'_\1', camel_case).lower()


def auto_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeterGroup(object):

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __str__(self):
        return "  ".join(str(v) for _, v in self.meters.items())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
