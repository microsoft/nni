# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import abc
from pathlib import Path
import pickle
from typing import Dict, List, Union

import numpy
import torch
import torch.nn.functional as F


_DISTIL_DATA = Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]


def _to_cpu_detach(data: _DISTIL_DATA) -> _DISTIL_DATA:
    if isinstance(data, torch.Tensor):
        return data.cpu().detach()
    elif isinstance(data, abc.Sequence) and not isinstance(data, str):
        return [_to_cpu_detach(d) for d in data]
    elif isinstance(data, abc.Mapping):
        return {k: _to_cpu_detach(v) for k, v in data.items()}
    else:
        return data


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


# def kl_div_distil(output: torch.Tensor, target: torch.Tensor, temperature: float = 2., reduction: str = 'batchmean'):
#     return F.kl_div(F.log_softmax(output / temperature, dim=-1), F.softmax(target / temperature, dim=-1), reduction=reduction) * (temperature ** 2)
