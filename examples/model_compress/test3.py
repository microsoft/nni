import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    n = Net()
    conv = getattr(n, 'conv')
    conv.weight.data.fill_(0.5)
    conv.bias.data.fill_(0.1)
    #---------------
    index = torch.tensor([1])
    conv2 = getattr(n, 'conv')
    new_conv2 = torch.nn.Conv2d(in_channels=1,
                               out_channels=conv2.out_channels,
                               kernel_size=conv2.kernel_size,
                               stride=conv2.stride,
                               padding=conv2.padding,
                               dilation=conv2.dilation,
                               groups=1, # currently only support groups is 1
                               bias=conv2.bias is not None,
                               padding_mode=conv2.padding_mode)
    tmp_weight_data = tmp_bias_data = None
    print('before select: ', conv2.weight.data.size(), conv2.weight.data)
    tmp_weight_data = torch.index_select(conv2.weight.data, 1, index)
    print('after select: ', tmp_weight_data.size(), tmp_weight_data)
    new_conv2.weight.data.copy_(tmp_weight_data)
    if conv2.bias is not None:
        new_conv2.bias.data.copy_(conv2.bias.data)
    setattr(n, 'conv', new_conv2)
    #----------------
    dummy_input = torch.zeros([1, 1, 3, 3])
    out = n(dummy_input)
    print(out)
