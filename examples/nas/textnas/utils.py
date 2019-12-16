import torch


def get_length(mask):
    length = torch.sum(mask, 1)
    length = length.long()
    return length
