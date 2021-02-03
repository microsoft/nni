import nni.retiarii.trainer.pytorch.lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from nni.retiarii import blackbox_module as bm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x


def test_lightning_trainer():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = bm(MNIST)(root='data/mnist', train=False, download=True, transform=transform)
    test_dataset = bm(MNIST)(root='data/mnist', train=True, download=True, transform=transform)
    lightning = pl.Lightning(pl.SupervisedLearning(), pl.Trainer(max_epochs=10, limit_val_batches=0.0),
                             train_dataloader=DataLoader(train_dataset, batch_size=100),
                             val_dataloaders=DataLoader(test_dataset, batch_size=100))
    lightning._execute(MNISTModel)


if __name__ == '__main__':
    test_lightning_trainer()