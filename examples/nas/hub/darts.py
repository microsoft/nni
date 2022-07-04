# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Reproduction of experiments in `DARTS paper <https://arxiv.org/abs/1806.09055>`__.
"""

import argparse

from nni.retiarii.evaluator.pytorch import Lightning, ClassificationModule, AccuracyWithLogits
from nni.retiarii.hub.pytorch import DARTS
from nni.retiarii import fixed_arch

class AuxLossClassificationModule(ClassificationModule):
    def training_step(self, batch, batch_idx):
        """"""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss


def search():
    model_space = DARTS(16, 8, 'cifar')


def train(arch):
    with fixed_arch(arch):
        model = DARTS(36, 20, 'cifar', auxiliary_loss=True)

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['search', 'train', 'test', 'search_train'], default='search_train')
    parser.add

    



if __name__ == '__main__':
    main()
