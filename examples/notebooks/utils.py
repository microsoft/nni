import os
import torch
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from torchvision.datasets.utils import download_url

class TitanicDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool = True):
        filename = 'train.csv' if train else 'eval.csv'
        if not os.path.exists(os.path.join(root, filename)):
            download_url(os.path.join('https://storage.googleapis.com/tf-datasets/titanic/', filename), root, filename)

        df = pd.read_csv(os.path.join(root, filename))
        object_colunmns = df.select_dtypes(include='object').columns.values
        for idx in df.columns:
            if idx in object_colunmns:
                df[idx] = LabelEncoder().fit_transform(df[idx])
           
        self.x = df.iloc[:, 1:].values
        self.y = df.iloc[:, 0].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx]), self.y[idx]

def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc1": (predicted == target).sum().item() / batch_size}