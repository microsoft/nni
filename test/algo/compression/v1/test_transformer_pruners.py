# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math
import sys
import unittest
from unittest import TestCase, main

from nni.algorithms.compression.pytorch.pruning import TransformerHeadPruner

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from ut.sdk.models.pytorch_models.transformer import TransformerEncoder


def validate_sparsity(wrapper, sparsity, bias=False):
    masks = [wrapper.weight_mask]
    if bias and wrapper.bias_mask is not None:
        masks.append(wrapper.bias_mask)
    for m in masks:
        actual_sparsity = (m == 0).sum().item() / m.numel()
        msg = 'actual sparsity: {:.2f}, target sparsity: {:.2f}'.format(actual_sparsity, sparsity)
        assert math.isclose(actual_sparsity, sparsity, abs_tol=0.1), msg


class Model(nn.Module):
    """
    A binary classifier using a transformer encoder for contextual embedding.
    """
    def __init__(self, n_layer, hidden_dim, n_head):
        super(Model, self).__init__()
        self.embedding = TransformerEncoder(vocab_size=100, hidden_dim=hidden_dim, n_layers=n_layer, n_heads=n_head)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        raw_output = self.embedding(x, mask)
        pooled_output = raw_output[0]
        prediction = F.sigmoid(self.classifier(pooled_output)).squeeze()
        return prediction


def train(model, dataloader, criterion, optimizer):
    model.train()
    device = next(model.parameters()).device
    for _ in range(2):
        y = torch.ones(10).to(device)
        out = model(torch.randint(0, 100, (4, 10)).to(device), torch.ones(10).to(device))
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def dry_run(model):
    device = next(model.parameters()).device
    for _ in range(2):
        y = torch.ones(10).to(device)
        _ = model(torch.randint(0, 100, (4, 10)).to(device), torch.ones(10).to(device))


def head_pruner_tests(criterion, global_sort, use_graph, iterative):
    print("Testing criterion {} with global_sort={} and use_graph={}".format(criterion, global_sort, use_graph))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build config list and arguments
    config_list = [{'sparsity': 0.5, 'op_types': ['Linear']}]

    kwargs = {'ranking_criterion': criterion, 'head_hidden_dim': 64}
    if global_sort:
        kwargs['global_sort'] = True
    else:
        kwargs['global_sort'] = False

    if use_graph:
        attention_name_groups = list(zip(['embedding.layers.{}.self_attn.q_proj'.format(i) for i in range(6)],
                                         ['embedding.layers.{}.self_attn.k_proj'.format(i) for i in range(6)],
                                         ['embedding.layers.{}.self_attn.v_proj'.format(i) for i in range(6)],
                                         ['embedding.layers.{}.self_attn.output_proj'.format(i) for i in range(6)]))
        kwargs['attention_name_groups'] = attention_name_groups
    else:
        dummy_input = (torch.randint(0, 100, (10, 32)).to(device), torch.ones(32).to(device))
        kwargs['dummy_input'] = dummy_input

    if iterative:
        kwargs['num_iterations'] = 2
        kwargs['epochs_per_iteration'] = 1

    n_layers = 6
    n_heads = 8
    hidden_dim = 512
    model = Model(n_layers, hidden_dim, n_heads)
    model.to(device)
    kwargs['optimizer'] = torch.optim.SGD(model.parameters(), lr=0.001)

    def trainer(model, optimizer, criterion, epoch):
        return train(model, None, criterion, optimizer)
    kwargs['trainer'] = trainer
    kwargs['criterion'] = nn.BCELoss()

    def forward_runner(model):
        return dry_run(model)
    kwargs['forward_runner'] = forward_runner

    # create pruner and call compress()
    pruner = TransformerHeadPruner(model, config_list, **kwargs)
    pruner.compress()

    # test model and mask export
    pruner.export_model('./model_tmp.pth', './mask_tmp.pth', device=device)
    dummy_input = (torch.randint(0, 100, (10, 32)).to(device), torch.ones(32).to(device))
    pruner.export_model('./model_tmp.pth', './mask_tmp.pth', './onnx_tmp.pth',
                        dummy_input=dummy_input, opset_version=10)

    # validate sparsity
    if not global_sort:
        for wrapper in pruner.modules_wrapper:
            validate_sparsity(wrapper, wrapper.config['sparsity'])


class PrunerTestCase(TestCase):
    def test_head_pruner(self):
        for criterion in ["l1_weight", "l2_weight", "l1_activation", "l2_activation", "taylorfo"]:
            for global_sort in [False, True]:
                for use_graph in [False, True]:
                    for iterative in [False, True]:
                        head_pruner_tests(criterion, global_sort, use_graph, iterative)

        file_paths = ['./model_tmp.pth', './mask_tmp.pth', './onnx_tmp.pth', './search_history.csv',
                      './search_result.json']
        for f in file_paths:
            if os.path.exists(f):
                os.remove(f)


if __name__ == '__main__':
    main()
