# This file will be deleted, will push an example to replace it.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.datasets import MNIST

from nni.experimental.distillation.preprocessor import Preprocessor
from nni.experimental.distillation.uid_dataset import AugmentationDataset, IndexedDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()

    for data, label in dataloader:
        optimizer.zero_grad()
        data, label = data.cuda(), label.cuda()
        logits = model(data)
        loss = F.cross_entropy(logits.softmax(-1), label)
        loss.backward()
        optimizer.step()


def ditil_train(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()

    for soft_label, (data, label) in dataloader:
        optimizer.zero_grad()
        data, label, soft_label = data.cuda(), label.cuda(), soft_label.cuda()
        logits = model(data)
        loss = F.cross_entropy(logits.softmax(-1), soft_label)
        loss.backward()
        optimizer.step()


def evaluate(model: torch.nn.Module, dataloader: DataLoader):
    model.eval()

    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y.view_as(preds)).sum().item()
    return correct / len(dataloader.dataset)


def labels_split_fn(labels: torch.Tensor):
    if isinstance(labels, torch.Tensor):
        values, indices = labels.detach().topk(2, -1)
        return list(zip(values.cpu().split(1), indices.cpu().split(1)))
    else:
        raise NotImplementedError('Only support split tensor, please customize split function.')


def labels_collate_fn(labels: list):
    values, indices = list(zip(*labels))
    values, indices = torch.cat(values), torch.cat(indices)
    pad_val = (1 - values.sum(-1)) / 8
    logits = pad_val.unsqueeze(-1) + torch.zeros(len(labels), 10)
    logits.scatter_(-1, indices, values)
    return logits


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomResizedCrop((28, 28), (0.8, 1.0)),
        transforms.RandomRotation(30)
    ])

    def transform_fn(sample):
        if isinstance(sample, (tuple, list)):
            return transform(sample[0]), sample[1]
        else:
            return transform(sample)

    mnist_train_dataset = MNIST(root='data/mnist', train=True, download=True, transform=transform)
    meta_dataset = MNIST(root='data/mnist', train=True, download=True)
    mnist_test_dataset = MNIST(root='data/mnist', train=False, transform=transform)
    mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=32, shuffle=True)
    mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=32)

    lenet = Net().cuda()

    # optimizer = Adam(lenet.parameters(), lr=1e-3)
    # for _ in range(3):
    #     start = time.time()
    #     train(lenet, mnist_train_dataloader, optimizer)
    #     print(time.time() - start)
    #     print(evaluate(lenet, mnist_test_dataloader))

    ########################################################

    def create_dataloader(dataset: Dataset):
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def predict(batch):
        lenet.eval()
        data, _ = batch[0].cuda(), batch[1].cuda()
        return lenet(data).softmax(-1)

    aug_dataset = AugmentationDataset(IndexedDataset(meta_dataset), transform=transform_fn)

    preprocessor = Preprocessor(predict, aug_dataset, create_dataloader, cache_mode='pickle',
                                labels_split_fn=labels_split_fn,
                                labels_collate_fn=labels_collate_fn
                               )

    start = time.time()
    preprocessor.preprocess_labels()
    print(time.time() - start)

    # preprocessor.save_checkpoint('/home/ningshang/nni/playground/distil_storage')

    # preprocessor = Preprocessor(predict, mnist_train_dataset, create_dataloader,
    #                             checkpoint_folder='/home/ningshang/nni/playground/distil_storage')

    relabeled_train_dataloader = preprocessor.create_replay_dataloader()

    lenet = Net().cuda()

    optimizer = Adam(lenet.parameters(), lr=1e-3)
    for _ in range(3):
        start = time.time()
        ditil_train(lenet, relabeled_train_dataloader, optimizer)
        print(time.time() - start)
        print(evaluate(lenet, mnist_test_dataloader))
