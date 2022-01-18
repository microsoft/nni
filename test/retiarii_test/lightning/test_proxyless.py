from nni.retiarii.nn.pytorch.component import Repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.retiarii.evaluator.pytorch.lightning import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from nni.retiarii.nn.pytorch import LayerChoice
import pytorch_lightning as pl

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        self.dropout1 = LayerChoice([
            nn.Dropout(.25),
            nn.Dropout(.5),
            nn.Dropout(.75)
        ])
        self.dropout2 = nn.Dropout(0.5)
        self.fc = LayerChoice([
            nn.Sequential(
                nn.Linear(9216, 64),
                nn.ReLU(),
                self.dropout2,
                nn.Linear(64, 10),
            ),
            nn.Sequential(
                nn.Linear(9216, 128),
                nn.ReLU(),
                self.dropout2,
                nn.Linear(128, 10),
            ),
            nn.Sequential(
                nn.Linear(9216, 256),
                nn.ReLU(),
                self.dropout2,
                nn.Linear(256, 10),
            )
        ])
        self.rpfc = Repeat(
            nn.Linear(10, 10),
            [1, 2]
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc(x)
        x = self.rpfc(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.CrossEntropyLoss()
        return loss(output, y)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).argmax(1, True)
        acc = pred.eq(y.view_as(pred)).sum().item()/ len(y)
        return acc
    
    def configure_optimizers(self):
        optim = torch.optim.Adagrad(self.parameters(), 1, 0)
        return optim

base_model = Net()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST('data/mnist', train = True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, 32, shuffle= True)
valid_dataset = MNIST('data/mnist', train = False, download=True, transform=transform)
valid_loader = DataLoader(valid_dataset, 64, shuffle= True)

def test_proxyless():
    from nni.retiarii.evaluator.pytorch.lightning import Classification, Regression
    cls = Classification(train_dataloader=train_loader, val_dataloaders=valid_loader, **{'max_epochs':1})
    cls.module.set_model(base_model)
    from nni.retiarii.oneshot.pytorch.differentiable import ProxylessModel
    proxyless_model = ProxylessModel(cls.module)
    from nni.retiarii.oneshot.pytorch.utils import ParallelTrainValDataLoader
    para_loader = ParallelTrainValDataLoader(cls.train_dataloader, cls.val_dataloaders)
    cls.trainer.fit(proxyless_model, para_loader)

if __name__ == '__main__':
    test_proxyless()