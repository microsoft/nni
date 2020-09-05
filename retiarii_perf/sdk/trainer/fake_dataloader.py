import torch

class FakeMNISTDataLoader:
    """
    Try to be as close to DataLoader as possible.
    Assume format: (images, labels)
    Input size needs to be known in advance.
    """

    def __init__(self, dataset, input_size, batch_size):
        self.input_size = input_size
        self.batch_size = batch_size
        self.length = len(dataset) // batch_size

    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        self.counter += 1
        images = torch.zeros(self.input_size, dtype=torch.float, device="cuda")
        labels = torch.zeros(self.batch_size, dtype=torch.long, device="cuda")
        return images, labels

class FakeDataLoader:
    """
    Try to be as close to DataLoader as possible.
    Assume format: (images, labels)
    Input size needs to be known in advance.
    """

    def __init__(self, dataset, input_size, batch_size, has_mask = False, mask_size = []):
        self.input_size = input_size
        self.batch_size = batch_size
        self.length = len(dataset) // batch_size
        self.has_mask = has_mask
        self.mask_size = mask_size

    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        self.counter += 1
        images = torch.zeros(self.input_size, dtype=torch.float)
        if self.has_mask:
            mask = torch.ones(self.mask_size, dtype=torch.float)
        labels = torch.zeros(self.batch_size, dtype=torch.long)
        if self.has_mask:
            return images, mask, labels
        else:
            return images, labels