import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


@model_wrapper
class Model(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, 3)
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(10, 10, 3),
            nn.MaxPool2d(3)
        ])
        self.conv3 = nn.LayerChoice([
            nn.Identity(),
            nn.Conv2d(10, 10, 1)
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


@model_wrapper
class ModelInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.LayerChoice([
            nn.Linear(10, 10),
            nn.Linear(10, 10, bias=False)
        ])
        self.net2 = nn.LayerChoice([
            nn.Linear(10, 10),
            nn.Linear(10, 10, bias=False)
        ])

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


@model_wrapper
class ModelNested(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = ModelInner()
        self.fc2 = nn.LayerChoice([
            nn.Linear(10, 10),
            nn.Linear(10, 10, bias=False)
        ])
        self.fc3 = ModelInner()

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


def test_model_wrapper():
    model = Model(3)
    assert model.trace_symbol == Model.__wrapped__
    assert model.trace_kwargs == {'in_channels': 3}
    assert model.conv2.label == 'model_1'
    assert model.conv3.label == 'model_2'
    assert model(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 1])

    model = Model(4)
    assert model.trace_symbol == Model.__wrapped__
    assert model.conv2.label == 'model_1'  # not changed


def test_model_wrapper_nested():
    model = ModelNested()
    assert model.fc1.net1.label == 'model_1_1'
    assert model.fc1.net2.label == 'model_1_2'
    assert model.fc2.label == 'model_2'
    assert model.fc3.net1.label == 'model_3_1'
    assert model.fc3.net2.label == 'model_3_2'


if __name__ == '__main__':
    test_model_wrapper_nested()
