from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, ImageNet
from sdk.trainer.fake_dataloader import FakeMNISTDataLoader, FakeDataLoader
import torch

import collections
import threading

def mnist_dataloader(batch_size=32, num_workers=4, shuffle=True, train=True):
    def dataloader(trainer):
        dataset = MNIST("./data/mnist", train=train, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return loader
    return dataloader

def fake_mnist_dataloader(input_size, batch_size=32, num_workers=4, shuffle=True, train=True):
    def dataloader(trainer):
        dataset = MNIST("./data/mnist", train=train, download=True, transform=transforms.ToTensor())
        loader = FakeMNISTDataLoader(dataset, input_size, batch_size) 
        return loader
    return dataloader

def fake_imagenet_dataloader(imagenet_root, input_size, batch_size=32, train=True):
    split = 'train' if train else 'val'
    def dataloader(trainer):
        dataset = ImageNet(imagenet_root, split=split, transform=transforms.ToTensor())
        loader = FakeDataLoader(dataset, input_size, batch_size) 
        return loader
    return dataloader
    

class BERTDataLoader(object):
    def __init__(self, dataset, batch_size, num_workers, **kwargs):
        self.num_workers = num_workers
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)
        
        from transformers import BertModel, BertTokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_queue = 32
        self.input_size = (batch_size, 64, 768)
        self.batch_size = batch_size
        self.length = len(dataset) // batch_size

    def __iter__(self):
        self.counter = 0
        self.shared_queue = collections.deque()
        return self._process_gen()

    def __len__(self):
        return len(self.dataloader)

    def _data_preprocess(self, text, label):
        text = torch.tensor([self.tokenizer.encode(t, max_length=64, pad_to_max_length=True, truncation=True) for t in text]).cuda()
        mask = text > 0
        with torch.no_grad():
            output, _ = self.model(text)
        return output, mask.float(), label.cuda()

    def _process_gen(self):
        for text, label in self.dataloader:
            yield self._data_preprocess(text, label)

    def _process(self):
        for text, label in self.dataloader:
            while len(self.shared_queue) >= self.max_queue:
                time.sleep(1)
            data = self._data_preprocess(text, label)            
            self.shared_queue.append(data)

    def __next__(self):
        self.counter += 1
        if self.counter >= len(self):
            raise StopIteration
        while not self.shared_queue:
            time.sleep(0.1)
        text, masks, labels = self.shared_queue.popleft()
        masks = masks.float()
        print("mask", masks.type())
        return text, masks, labels

def bert_dataloader(dataset, batch_size, num_workers, **kwargs):
    def dataloader(trainer):
        loader = BERTDataLoader(dataset, batch_size, num_workers, **kwargs)
        return loader
    return dataloader
