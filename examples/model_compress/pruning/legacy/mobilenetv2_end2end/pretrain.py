# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from time import gmtime, strftime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_validation(model, valid_dataloader):
    model.eval()
    
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(valid_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds= model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)

    valid_loss = np.array(loss_list).mean()
    valid_acc = np.array(acc_list).mean()
    
    return valid_loss, valid_acc


def run_pretrain(args):
    print(args)
    torch.set_num_threads(args.n_workers)
    
    model_type = 'mobilenet_v2_torchhub'   
    pretrained = True                      # load imagenet weight
    experiment_dir = 'pretrained_{}'.format(model_type) if args.experiment_dir is None else args.experiment_dir
    os.mkdir(experiment_dir)
    checkpoint = None
    input_size = 224
    n_classes = 120
    
    log = open(experiment_dir + '/pretrain.log', 'w')
    
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    model = model.to(device)
    print(model)
    # count_flops(model, device=device)

    train_dataset = TrainDataset('./data/stanford-dogs/Processed/train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = EvalDataset('./data/stanford-dogs/Processed/valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    for epoch in range(args.n_epochs):
        print('Start training epoch {}'.format(epoch))
        loss_list = []

        # train
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            
        # validation
        valid_loss, valid_acc = run_validation(model, valid_dataloader)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))
        log.write('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}\n'.format
                  (epoch, train_loss, valid_loss, valid_acc))
        
        # save
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), experiment_dir + '/checkpoint_best.pt')

    log.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Example code for pruning MobileNetV2')

    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='directory containing the pretrained model')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_best.pt',
                         help='checkpoint of the pretrained model')
    
    # finetuning parameters
    parser.add_argument('--n_workers', type=int, default=16,
                        help='number of threads')
    parser.add_argument('--n_epochs', type=int, default=180,
                        help='number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training and inference')

    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    args = parse_args()
    run_pretrain(args)
