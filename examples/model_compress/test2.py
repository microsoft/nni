import torch
import torch.nn as nn

torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(4, 8, 3)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 4, 5)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        #print(out)
        out = self.relu(out)
        #print(out)
        out = self.conv2(out)
        #print(out)
        return out

def add_masks(model):
    bn = getattr(model, 'bn')
    print(bn)
    #print('before: ', bn.weight.data)
    bn.weight.data = bn.weight.data * torch.tensor([0,1,0,1,0,1,0,1])
    #print('after', bn.weight.data)
    bn.bias.data = bn.bias.data * torch.tensor([0,1,0,1,0,1,0,1])

def model_speedup(model):
    index = torch.tensor([1,3,5,7])
    #-----
    conv = getattr(model, 'conv')
    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                               out_channels=4,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=1, # currently only support groups is 1
                               bias=conv.bias is not None,
                               padding_mode=conv.padding_mode)
    tmp_weight_data = tmp_bias_data = None
    tmp_weight_data = torch.index_select(conv.weight.data, 0, index)
    if conv.bias is not None:
        tmp_bias_data = torch.index_select(conv.bias.data, 0, index)
    new_conv.weight.data.copy_(tmp_weight_data)
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data if tmp_bias_data is None else tmp_bias_data)
    setattr(model, 'conv', new_conv)
    #-------
    norm = getattr(model, 'bn')
    new_norm = torch.nn.BatchNorm2d(num_features=4,
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, index)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, index)
    if norm.track_running_stats:
        new_norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, index)
        new_norm.running_var.data = torch.index_select(norm.running_var.data, 0, index)
    setattr(model, 'bn', new_norm)
    #---------
    conv2 = getattr(model, 'conv2')
    new_conv2 = torch.nn.Conv2d(in_channels=4,
                               out_channels=conv2.out_channels,
                               kernel_size=conv2.kernel_size,
                               stride=conv2.stride,
                               padding=conv2.padding,
                               dilation=conv2.dilation,
                               groups=1, # currently only support groups is 1
                               bias=conv2.bias is not None,
                               padding_mode=conv2.padding_mode)
    tmp_weight_data = tmp_bias_data = None
    #print('before select: ', conv2.weight.data.size(), conv2.weight.data)
    tmp_weight_data = torch.index_select(conv2.weight.data, 1, index)
    #print('after select: ', tmp_weight_data.size(), tmp_weight_data)
    new_conv2.weight.data.copy_(tmp_weight_data)
    if conv2.bias is not None:
        new_conv2.bias.data.copy_(conv2.bias.data)
    setattr(model, 'conv2', new_conv2)
    #----------

if __name__ == '__main__':
    n = Net()
    n.train()
    dummy_input = torch.randn(6, 4, 16, 16)
    mask_flag = True
    if mask_flag:
        add_masks(n)
    else:
        conv = getattr(n, 'conv')
        #print('conv before: ', conv.weight.data.size(), conv.weight.data)
        model_speedup(n)
        bn = getattr(n, 'bn')
        #print('bn: ', bn.weight.data)
        conv = getattr(n, 'conv')
        #print('conv after: ', conv.weight.data.size(), conv.weight.data)
    out = n(dummy_input)
    print(out.size(), out)

    '''example_weight = torch.rand(1, 1, 3, 3)
    example_forward_input = torch.rand(1, 1, 3, 3)

    # Trace a specific method and construct `ScriptModule` with
    # a single `forward` method
    module = torch.jit.trace(n.forward, example_forward_input)

    # Trace a module (implicitly traces `forward`) and construct a
    # `ScriptModule` with a single `forward` method
    module = torch.jit.trace(n, example_forward_input)
    print(module.graph)
    print(torch._C._jit_pass_inline(module.graph))
    print(module.graph)'''
