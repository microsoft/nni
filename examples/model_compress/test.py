import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    n = Net()
    example_weight = torch.rand(1, 1, 3, 3)
    example_forward_input = torch.rand(1, 1, 3, 3)

    # Trace a specific method and construct `ScriptModule` with
    # a single `forward` method
    module = torch.jit.trace(n.forward, example_forward_input)

    # Trace a module (implicitly traces `forward`) and construct a
    # `ScriptModule` with a single `forward` method
    module = torch.jit.trace(n, example_forward_input)
    print(module.graph)
    print(torch._C._jit_pass_inline(module.graph))
    print(module.graph)
