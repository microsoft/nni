"""
A modification of the original GraphNAS implementation based on nni.
See https://github.com/GraphNAS/GraphNAS. 
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import nni

from model.pyg_gnn import GraphNet


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_accuracy(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

def main():
    fix_seed(123)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='Citeseer',
                        help="Dataset, one of {Cora, Citeseer, Pubmed}")
    parser.add_argument('--epochs',
                        type=str,
                        default='300',
                        help="epochs")
    args = parser.parse_args()

    dataset = Planetoid('.', args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    num_label = int(max(data.y)) + 1
    arc = nni.get_next_parameter()
    arc[-1] = num_label
    model = GraphNet(arc, num_feat=data.x.shape[1], num_label=num_label, batch_normal=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    min_val_loss = float('inf')
    max_val_acc = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()

        # training
        logits = model(data.x, data.edge_index)
        logits = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        logits = model(data.x, data.edge_index)
        logits = F.log_softmax(logits, dim=1)
        val_loss = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
        val_acc = compute_accuracy(logits, data.y, data.val_mask)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            max_val_acc = val_acc

    nni.report_final_result(max_val_acc)

if __name__ == '__main__':
    main()
