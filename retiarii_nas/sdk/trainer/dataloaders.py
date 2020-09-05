from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def mnist_dataloader(batch_size=32, num_workers=4, shuffle=True, train=True):
    def dataloader(trainer):
        dataset = MNIST("./data/mnist", train=train, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return loader
    return dataloader
