import torch

def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc1": (predicted == target).sum().item() / batch_size}