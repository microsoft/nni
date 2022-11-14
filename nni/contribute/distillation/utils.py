# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import pickle
from typing import List

import numpy
import torch


def _to_tensor(sample):
    if isinstance(sample, torch.Tensor):
        return sample
    elif isinstance(sample, numpy.ndarray):
        return torch.from_numpy(sample)
    else:
        return torch.tensor(sample)


def _default_labels_split_fn(labels: torch.Tensor):
    if isinstance(labels, torch.Tensor):
        return [sl.squeeze(0) for sl in torch.split(labels.detach().cpu(), 1)]
    else:
        raise NotImplementedError(f'Only support split tensor, please customize split function for {type(labels)}.')


def _default_labels_collate_fn(labels: List):
    labels = list(map(_to_tensor, labels))
    return torch.stack(labels)


def pickle_dump(obj, file_path, protocol=4):
    dump_path = Path(file_path).absolute()
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open(mode='wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def pickle_load(file_path):
    load_path = Path(file_path).absolute()
    assert load_path.exists(), f'{file_path} is not exist.'
    with load_path.open(mode='rb') as f:
        return pickle.load(f)
