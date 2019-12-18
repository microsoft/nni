import torch
from torchtext import data, datasets, vocab


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)

train, val, test = datasets.SST.splits(TEXT, LABEL, root="data")

TEXT.build_vocab(train, vectors=vocab.GloVe(cache="data"))
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=128, device=device)


class IteratorWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        data = next(self.iterator)
        text, length = data.text
        max_length = text.size(1)
        label = data.label
        bs = label.size(0)
        mask = torch.arange(max_length, device=length.device).unsqueeze(0).repeat(bs, 1)
        mask = mask < length.unsqueeze(-1).repeat(1, max_length)
        return (text, mask), label


train_iter = IteratorWrapper(train_iter)
val_iter = IteratorWrapper(val_iter)
test_iter = IteratorWrapper(test_iter)
print(len(train_iter), len(val_iter), len(test_iter))

# for i, data in enumerate(train_iter):
#     print(data)
#     if i >= 5:
#         break
