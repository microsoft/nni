# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import OrderedDict

import numpy as np
import torch
import nni.retiarii.nn.pytorch as nn
from nni.nas.pytorch.mutables import InputChoice, LayerChoice
from torch.utils.data import DataLoader, Dataset
from nni.retiarii.evaluator.pytorch.lightning import DataLoader as nniDataLoader

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

def replace_module_with_type(root_module, init_fn, type_name, modules):

    return _replace_module_with_type(root_module, init_fn, type_name, modules)

class ParallelTrainValDataset(Dataset):
    ''' Dataset used by NAS algorithms. Some NAS algs may require both train_batch
        and val_bath at the training_step. 
        In this implement, the shorter one is upsampled to match the longer one.
    '''
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_dataset = train_data.dataset if isinstance(train_data, DataLoader) else train_data
        self.val_dataset = val_data.dataset if isinstance(val_data, DataLoader) else val_data
        if len(self.train_dataset) / len(self.val_dataset) > 1.5 or len(self.val_dataset) / len(self.train_dataset) > 1.5 :
            logging.warning(f'The length difference betweeen the training dataset({len(self.train_dataset)})' \
                f' and the validation dataset({len(self.val_dataset)}) is too large. The shorter one will be upsampled.')
    
    def __getitem__(self, index):
        # The shorter one is upsampled
        return self.train_dataset[index % len(self.train_dataset)], self.val_dataset[index % len(self.val_dataset)]
    
    def __len__(self):
        return max(len(self.train_dataset), len(self.val_dataset))

class ConcatenateTrainValDataset(Dataset):
    ''' Dataset used by NAS algorithms. Some NAS algs may require both train_batch
        and val_bath at the training_step. 
        In this implement, the validation batches come after the train ones.
    '''
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_dataset = train_data if isinstance(train_data, Dataset) else train_data.dataset
        self.val_dataset = val_data if isinstance(val_data, Dataset) else val_data.dataset
        
    def __getitem__(self, index):
        if index < len(self.train_dataset):
            return (self.train_dataset[index], True)
        return (self.val_dataset[index - len(self.train_dataset)], False)
    
    def __len__(self):
        return len(self.train_dataset)+ len(self.val_dataset)

class ParallelTrainValDataLoader(DataLoader):
    def __init__(self, train_dataloader, val_dataloader):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.equal_len = len(train_dataloader) == len(val_dataloader)
        self.train_longger = len(train_dataloader) > len(val_dataloader)
        super().__init__(None)
    
    def __iter__(self):
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)
        return self
    
    def __next__(self):
        try:
            train_batch = next(self.train_iter)
        except StopIteration:
            if self.equal_len or self.train_longger:
                raise StopIteration()
            # val is the longger one
            self.train_iter = iter(self.train_loader)
            train_batch = next(self.train_iter)
        try:
            val_batch = next(self.val_iter)
        except StopIteration:
            if not self.train_longger:
                raise StopIteration()
            self.val_iter = iter(self.val_loader)
            val_batch = next(self.val_iter)
        return train_batch, val_batch
    
    def __len__(self) -> int:
        return max(len(self.train_loader), len(self.val_loader))

class ConcatenateTrainValDataLoader(DataLoader):
    def __init__(self, train_dataloader, val_dataloader):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        super().__init__(None)
    
    def __iter__(self):
        self.cur_iter = iter(self.train_loader)
        self.is_train = True
        return self
    
    def __next__(self):
        try:
            batch = next(self.cur_iter)
        except StopIteration:
            if self.is_train:
                self.cur_iter = iter(self.val_loader)
                self.is_train = False
                return next(self)
            raise StopIteration()
        else:
            return batch, self.is_train

    def __len__(self):
        return len(self.train_loader) + len(self.val_loader)

def get_parallel_dataloader(train_dataset, val_dataset, batch_size = 1, shuffle = False, num_workers = 1):
    parallel_dataset = ParallelTrainValDataset(train_dataset, val_dataset)
    return nniDataLoader(parallel_dataset, batch_size, shuffle, num_workers=num_workers)

def get_concatenate_dataloader(train_dataset, val_dataset, batch_size = 1, shuffle = False, num_workers = 1):
    concat_dataset = ConcatenateTrainValDataset(train_dataset, val_dataset)
    return nniDataLoader(concat_dataset, batch_size, shuffle, num_workers=num_workers)

