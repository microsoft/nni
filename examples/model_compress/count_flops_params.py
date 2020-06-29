#%%
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from nni.compression.torch import FPGMPruner
from nni.compression.torch import apply_compression_results, ModelSpeedup

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

config_list = [{
            'sparsity': 0.5,
            'op_types': ['Conv2d']
        }]

model = NaiveModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
pruner = FPGMPruner(model, config_list, optimizer)
model = pruner.compress()

### torchprofile
# inputs = torch.randn(1, 1, 28, 28)
# from torchprofile import profile_macs
# macs = profile_macs(model, inputs, reduction=None)


# ### thop
# from thop import profile
# macs, params = profile(model, inputs=(inputs, ), verbose=False)

# ### torchscope
# from torchscope import scope
# scope(model, input_size=(1, 28, 28))


### counter
from nni.compression.torch.utils.counter import count_flops_params
flops, params = count_flops_params(model, (1, 1, 28, 28))
print(flops, params)
