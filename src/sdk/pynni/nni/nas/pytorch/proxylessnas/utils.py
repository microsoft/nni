# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

def detach_variable(inputs):
    """
    Detach variables

    Parameters
    ----------
    inputs : pytorch tensors
        pytorch tensors
    """
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    """
    Parameters
    ----------
    pred : pytorch tensor
        predicted value
    target : pytorch tensor
        label
    label_smoothing : float
        the degree of label smoothing

    Returns
    -------
    pytorch tensor
        cross entropy
    """
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k

    Parameters
    ----------
    output : pytorch tensor
        output, e.g., predicted value
    target : pytorch tensor
        label
    topk : tuple
        specify top1 and top5

    Returns
    -------
    list
        accuracy of top1 and top5
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
