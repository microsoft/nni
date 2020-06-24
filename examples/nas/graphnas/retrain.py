import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import torch_geometric.transforms as T
import pickle
import logging
from torch_geometric.datasets import Planetoid
from nni.nas.pytorch.darts import DartsTrainer
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
    parser.add_argument('--arc_save_path',
                        type=str,
                        default='./arcs.pkl',
                        help="Architecture save path")
    args = parser.parse_args()

    dataset = Planetoid('.', args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    num_label = int(max(data.y)) + 1

    f = open(args.arc_save_path, 'rb')
    arcs = pickle.load(f)
    f.close()

    best_arc = None
    total_max_val_acc = 0
    total_max_test_acc = 0

    for arc in arcs:
        arc[-1] = num_label
        model = GraphNet(arc, num_feat=data.x.shape[1], num_label=num_label, batch_normal=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        min_val_loss = float('inf')
        max_val_acc = 0
        max_test_acc = 0
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
            test_acc = compute_accuracy(logits, data.y, data.test_mask)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                max_val_acc = val_acc
                max_test_acc = test_acc

        print(f'Current architecture: {arc} | Val acc: {max_val_acc} | Test acc: {max_test_acc}')

        if max_val_acc > total_max_val_acc:
            total_max_val_acc = max_val_acc
            total_max_test_acc = max_test_acc
            best_arc = arc

    print('*' * 80)
    print(f'Best architecture: {best_arc} | Val acc: {total_max_val_acc} | Test acc: {total_max_test_acc}')
    print('*' * 80)

if __name__ == '__main__':
    main()
