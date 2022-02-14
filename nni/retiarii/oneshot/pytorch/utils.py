# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
import nni.retiarii.nn.pytorch as nn
from nni.nas.pytorch.mutables import InputChoice, LayerChoice

_logger = logging.getLogger(__name__)


def to_device(obj, device):
    """
    Move a tensor, tuple, list, or dict onto device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def to_list(arr):
    if torch.is_tensor(arr):
        return arr.cpu().numpy().tolist()
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return arr


class AverageMeterGroup:
    """
    Average meter group for multiple average meters.
    """

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        """
        Update the meter group with a dict of metrics.
        Non-exist average meters will be automatically created.
        """
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        """
        Return a summary string of group data.
        """
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """
    Computes and stores the average and current value.

    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.

        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def _replace_module_with_type(root_module, init_fn, type_name, modules):
    if modules is None:
        modules = []
    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, type_name):
                setattr(m, name, init_fn(child))
                modules.append((child.key, getattr(m, name)))
            else:
                apply(child)

    apply(root_module)
    return modules


def replace_layer_choice(root_module, init_fn, modules=None):
    """
    Replace layer choice modules with modules that are initiated with init_fn.

    Parameters
    ----------
    root_module : nn.Module
        Root module to traverse.
    init_fn : Callable
        Initializing function.
    modules : dict, optional
        Update the replaced modules into the dict and check duplicate if provided.

    Returns
    -------
    List[Tuple[str, nn.Module]]
        A list from layer choice keys (names) and replaced modules.
    """
    return _replace_module_with_type(root_module, init_fn, (LayerChoice, nn.LayerChoice), modules)


def replace_input_choice(root_module, init_fn, modules=None):
    """
    Replace input choice modules with modules that are initiated with init_fn.

    Parameters
    ----------
    root_module : nn.Module
        Root module to traverse.
    init_fn : Callable
        Initializing function.
    modules : dict, optional
        Update the replaced modules into the dict and check duplicate if provided.

    Returns
    -------
    List[Tuple[str, nn.Module]]
        A list from layer choice keys (names) and replaced modules.
    """
    return _replace_module_with_type(root_module, init_fn, (InputChoice, nn.InputChoice), modules)


class InterleavedTrainValDataLoader(DataLoader):
    """
    Dataloader that yields both train data and validation data in a batch, with an order of (train_batch, val_batch). The shorter
    one will be upsampled (repeated) to the length of the longer one, and the tail of the last repeat will be dropped. This enables
    users to train both model parameters and architecture parameters in parallel in an epoch.

    Some NAS algorithms, i.e. DARTS and Proxyless, require this type of dataloader.

    Parameters
    ----------
    train_data : DataLoader
        training dataloader
    val_data : DataLoader
        validation dataloader

    Example
    --------
    Fit your dataloaders into a parallel one.
    >>> para_loader = InterleavedTrainValDataLoader(train_dataloader, val_dataloader)
    Then you can use the ``para_loader`` as a normal training loader.
    """
    def __init__(self, train_dataloader, val_dataloader):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.equal_len = len(train_dataloader) == len(val_dataloader)
        self.train_longer = len(train_dataloader) > len(val_dataloader)
        super().__init__(None)

    def __iter__(self):
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)
        return self

    def __next__(self):
        try:
            train_batch = next(self.train_iter)
        except StopIteration:
            # training data is used up
            if self.equal_len or self.train_longer:
                # if training is the longger one or equal, stop iteration
                raise StopIteration()
            # if training is the shorter one, upsample it
            self.train_iter = iter(self.train_loader)
            train_batch = next(self.train_iter)

        try:
            val_batch = next(self.val_iter)
        except StopIteration:
            # validation data is used up
            if not self.train_longer:
                # if validation is the longger one (the equal condition is
                # covered above), stop iteration
                raise StopIteration()
            # if validation is the shorter one, upsample it
            self.val_iter = iter(self.val_loader)
            val_batch = next(self.val_iter)

        return train_batch, val_batch

    def __len__(self) -> int:
        return max(len(self.train_loader), len(self.val_loader))


class ConcatenateTrainValDataLoader(DataLoader):
    """
    Dataloader that yields validation data after training data in an epoch. You will get a batch with the form of (batch, source) in the
    training step, where ``source`` is a string which is either 'train' or 'val', indicating which dataloader the batch comes from. This
    enables users to train model parameters first in an epoch, and then train architecture parameters.

    Some NAS algorithms, i.e. ENAS, may require this type of dataloader.

    Parameters
    ----------
    train_data : DataLoader
        training dataloader
    val_data : DataLoader
        validation dataloader

    Warnings
    ----------
    If you set ``limit_train_batches`` of the trainer, the validation batches may be skipped.
    Consider downsampling the train dataset and the validation dataset instead if you want to shorten the length of data.

    Example
    --------
    Fit your dataloaders into a concatenated one.
    >>> concat_loader = ConcatenateTrainValDataLoader(train_dataloader, val_datalodaer)
    Then you can use the ``concat_loader`` as a normal training loader.
    """
    def __init__(self, train_dataloader, val_dataloader):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        super().__init__(None)

    def __iter__(self):
        self.cur_iter = iter(self.train_loader)
        self.source = 'train'
        return self

    def __next__(self):
        try:
            batch = next(self.cur_iter)
        except StopIteration:
            # training data is used up, change to validation data
            if self.source == 'train':
                self.cur_iter = iter(self.val_loader)
                self.source = 'val'
                return next(self)
            raise StopIteration()
        else:
            return batch, self.source

    def __len__(self):
        return len(self.train_loader) + len(self.val_loader)
