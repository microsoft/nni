import torch
from torchtext import data, datasets, vocab


TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)

train, val, test = datasets.SST.splits(TEXT, LABEL, root="data")

TEXT.build_vocab(train, vectors=vocab.GloVe(cache="data"))
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=4, device=torch.device("cuda"))

for i, data in enumerate(train_iter):
    print(data)
    print(data.text)
    print(data.label)
    if i >= 5:
        break
