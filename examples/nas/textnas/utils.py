import torch
import torch.nn as nn

INF = 1E10
EPS = 1E-12


def get_length(mask):
    length = torch.sum(mask, 1)
    length = length.long()
    return length


class GlobalAvgPool(nn.Module):
    def forward(self, x, mask):
        x = torch.sum(x, 2)
        length = torch.sum(mask, 1, keepdim=True).float()
        length += torch.eq(length, 0.0).float() * EPS
        length = length.repeat(1, x.size()[1])
        x /= length
        return x


class GlobalMaxPool(nn.Module):
    def forward(self, x, mask):
        mask = torch.eq(mask.float(), 0.0).long()
        mask = torch.unsqueeze(mask, dim=1).repeat(1, x.size()[1], 1)
        mask *= -INF
        x += mask
        x, _ = torch.max(x + mask, 2)
        return x
