# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn

from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch import mutables

from op_libs.textnas_ops import ConvBN, LinearCombine, AvgPool, MaxPool, RNN, Attention, BatchNorm, Bool
from op_libs.textnas_utils import GlobalMaxPool, GlobalAvgPool

import collections
from transformers import BertModel, BertTokenizer

class SharedDataLoader(object):
    def __init__(self, dataset, rank, once, batch_size, num_workers, **kwargs):
        self.num_workers = num_workers
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, **kwargs)
        if rank == 0 or not once:
            self.model = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_queue = 32
        self.once = once
        self.input_size = (batch_size, 64, 768)
        self.batch_size = batch_size
        self.rank = rank
        self.length = len(dataset) // batch_size

    def __iter__(self):
        self.counter = 0
        self.shared_queue = collections.deque()
        if self.rank == 0 or not self.once:
            if self.num_workers == 0:
                return self._process_gen()
            else:
                threading.Thread(target=self._process).start()
        return self

    def __len__(self):
        return len(self.dataloader)

    def _data_preprocess(self, text, label):
        text = torch.tensor([self.tokenizer.encode(t, max_length=64, pad_to_max_length=True) for t in text]).cuda()
        mask = text > 0
        with torch.no_grad():
            output, _ = self.model(text)
        return output, mask, label.cuda()

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
        if not self.once:
            while not self.shared_queue:
                time.sleep(0.1)
            return self.shared_queue.popleft()
        if self.rank == 0:
            while not self.shared_queue:
                time.sleep(0.1)
            text, masks, labels = self.shared_queue.popleft()
            masks = masks.float()
        else:
            text = torch.zeros(self.input_size, dtype=torch.float, device="cuda")
            labels = torch.zeros(self.batch_size, dtype=torch.long, device="cuda")
            masks = torch.zeros(self.input_size[:2], dtype=torch.float, device="cuda")
        torch.distributed.broadcast(text, 0)
        torch.distributed.broadcast(labels, 0)
        torch.distributed.broadcast(masks, 0)
        masks = masks.bool()
        return text, masks, labels

class WrapperOp(nn.Module):
    def __init__(self, op_choice, input_args):
        super(WrapperOp, self).__init__()
        self.op_choice = op_choice
        self.input_args = input_args
        self.op = None

        def conv_shortcut(kernel_size, hidden_units, cnn_keep_prob):
            return ConvBN(kernel_size, hidden_units, hidden_units,
                          cnn_keep_prob, False, True)
        
        if op_choice == 'conv_shortcut1':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'conv_shortcut3':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'conv_shortcut5':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'conv_shortcut7':
            self.op = conv_shortcut(*input_args)
        elif op_choice == 'AvgPool':
            self.op = AvgPool(3, False, True)
        elif op_choice == 'MaxPool':
            self.op = MaxPool(3, False, True)
        elif op_choice == 'RNN':
            self.op = RNN(*input_args)
        elif op_choice == 'Attention':
            self.op = Attention(*input_args)
        else:
            raise

    def forward(self, prec, mask):
        return self.op(prec, mask)

class Layer(mutables.MutableScope):
    def __init__(self, key, prev_keys, hidden_units, choose_from_k, cnn_keep_prob, lstm_keep_prob, att_keep_prob, att_mask):
        super(Layer, self).__init__(key)

        self.n_candidates = len(prev_keys)
        if self.n_candidates:
            self.prec = mutables.InputChoice(choose_from=prev_keys[-choose_from_k:], n_chosen=1)
            #===self.prec = 1
        else:
            # first layer, skip input choice
            self.prec = None
        self.op = mutables.LayerChoice([
            #conv_shortcut(1),
            WrapperOp('conv_shortcut1', [1, hidden_units, cnn_keep_prob]),
            #conv_shortcut(3),
            WrapperOp('conv_shortcut3', [3, hidden_units, cnn_keep_prob]),
            #conv_shortcut(5),
            WrapperOp('conv_shortcut5', [5, hidden_units, cnn_keep_prob]),
            #conv_shortcut(7),
            WrapperOp('conv_shortcut7', [7, hidden_units, cnn_keep_prob]),
            #AvgPool(3, False, True),
            WrapperOp('AvgPool', [3, False, True]),
            #MaxPool(3, False, True),
            WrapperOp('MaxPool', [3, False, True]),
            #RNN(hidden_units, lstm_keep_prob),
            WrapperOp('RNN', [hidden_units, lstm_keep_prob]),
            #Attention(hidden_units, 4, att_keep_prob, att_mask)
            WrapperOp('Attention', [hidden_units, 4, att_keep_prob, att_mask])
        ])
        #self.op = conv_shortcut(1)
        #self.op = Attention(hidden_units, 4, att_keep_prob, att_mask)
        #self.op = RNN(hidden_units, lstm_keep_prob)
        #self.op = WrapperOp('RNN', [hidden_units, lstm_keep_prob])
        #self.op = WrapperOp('Attention', [hidden_units, 4, att_keep_prob, att_mask])
        #self.op = WrapperOp('MaxPool', [3, False, True])
        #self.op = WrapperOp('AvgPool', [3, False, True])
        #self.op = WrapperOp('conv_shortcut7', [7, hidden_units, cnn_keep_prob])
        #self.op = WrapperOp('conv_shortcut5', [5, hidden_units, cnn_keep_prob])
        #self.op = WrapperOp('conv_shortcut3', [3, hidden_units, cnn_keep_prob])
        #self.op = WrapperOp('conv_shortcut1', [1, hidden_units, cnn_keep_prob])
        if self.n_candidates:
            self.skipconnect = mutables.InputChoice(choose_from=prev_keys)
            #===self.skipconnect = 1
        else:
            self.skipconnect = None
        self.bn = BatchNorm(hidden_units, False, True)

    def forward(self, last_layer, prev_layers, mask):
        # pass an extra last_layer to deal with layer 0 (prev_layers is empty)
        if self.prec is None:
            prec = last_layer
        else:
            prec = self.prec(prev_layers[-self.prec.n_candidates:])  # skip first
            #===x = min(len(prev_layers), self.prec_n_candidates)
            #===prec = prev_layers[-x]  # skip first
        out = self.op(prec, mask)
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[-self.skipconnect.n_candidates:])
            #===connection = prev_layers[-self.skip_n_candidates]
            if connection is not None:
                out = out + connection
        out = self.bn(out, mask)
        return out


class Model(nn.Module):
    def __init__(self, embedding_dim=768, hidden_units=256, num_layers=24, num_classes=5, choose_from_k=5,
                 lstm_keep_prob=0.5, cnn_keep_prob=0.5, att_keep_prob=0.5, att_mask=True,
                 embed_keep_prob=0.5, final_output_keep_prob=1.0, global_pool="avg"):
        super(Model, self).__init__()

        # self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        #from sdk.custom_ops_torch import Bert
        #self.bert = Bert("bert-base-uncased", 64, True, True)
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.breakpoint = nn.Identity()
        self.Bool = Bool()
        self.init_conv = ConvBN(1, embedding_dim, hidden_units, cnn_keep_prob, False, True)

        self.layers = nn.ModuleList()
        candidate_keys_pool = []
        for layer_id in range(self.num_layers):
            k = "layer_{}".format(layer_id)
            self.layers.append(Layer(k, candidate_keys_pool, hidden_units, choose_from_k,
                                     cnn_keep_prob, lstm_keep_prob, att_keep_prob, att_mask))
            candidate_keys_pool.append(k)

        self.linear_combine = LinearCombine(self.num_layers)
        self.linear_out = nn.Linear(self.hidden_units, self.num_classes)

        self.embed_dropout = nn.Dropout(p=1 - embed_keep_prob)
        self.output_dropout = nn.Dropout(p=1 - final_output_keep_prob)

        assert global_pool in ["max", "avg"]
        if global_pool == "max":
            self.global_pool = GlobalMaxPool()
        elif global_pool == "avg":
            self.global_pool = GlobalAvgPool()

    def forward(self, text, mask):
        # sent_ids, mask = inputs
        # seq = self.embedding(sent_ids.long())
        #text, mask = self.bert(inputs)
        breakpoint_text = self.breakpoint(text)
        seq = self.embed_dropout(breakpoint_text)

        seq = torch.transpose(seq, 1, 2)  # from (N, L, C) -> (N, C, L)
        
        bool_mask = self.Bool(mask)
        x = self.init_conv(seq, bool_mask)
        prev_layers = []

        for layer in self.layers:
            x = layer(x, prev_layers, bool_mask)
            prev_layers.append(x)

        x = self.linear_combine(torch.stack(prev_layers))
        x = self.global_pool(x, bool_mask)
        x = self.output_dropout(x)
        x = self.linear_out(x)
        return x

#====================Training approach

import sdk
from sdk.mutators.builtin_mutators import ModuleMutator
from sdk.trainer.dataloaders import BERTDataLoader
import sdk.trainer.sst_dataset
import json

class ModelTrain(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrain, self).__init__()
        self.device = torch.device(device)
        train_dataset, _, test_dataset = sst_dataset.read_data_sst(train_with_valid=True)
        self.train_loader = BERTDataLoader(train_dataset, 256, 0, shuffle=True, drop_last=True)
        self.test_loader = BERTDataLoader(test_dataset, 256, 0)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

from sdk.custom_ops_torch import FixedInputChoice

def apply_fixed_input_choice(model, fixed_arc_json):
    with open(fixed_arc_json) as f:
        fixed_arc = json.load(f)
    def apply(m):
        for name, module in m.named_children():
            if isinstance(module, mutables.InputChoice):
                n_candidates = getattr(m, name).n_candidates
                setattr(m, name, FixedInputChoice(fixed_arc[module.key]))
                getattr(m, name).n_candidates = n_candidates
            else:
                apply(module)
    apply(model)
#====================Experiment config

# mnasnet0_5
base_model = Model()

apply_fixed_architecture(base_model, "final_arc.json")
apply_fixed_input_choice(base_model, "final_arc.json")


exp = sdk.create_experiment('textnas', base_model)
exp.specify_training(ModelTrain)

exp.specify_mutators([])
exp.specify_strategy('naive.strategy.main', 'naive.strategy.RandomSampler')
run_config = {
    'authorName': 'nas',
    'experimentName': 'nas',
    'trialConcurrency': 1,
    'maxExecDuration': '24h',
    'maxTrialNum': 999,
    'trainingServicePlatform': 'local',
    'searchSpacePath': 'empty.json',
    'useAnnotation': False
} # nni experiment config
pre_run_config = {
    'name' : f'textnas',
    'x_shape' : [128, 64, 768],
    'x_dtype' : 'torch.float32',
    'y_shape' : [128],
    "y_dtype" : "torch.int64",
    "mask" : True
}
exp.run(run_config, pre_run_config=pre_run_config)