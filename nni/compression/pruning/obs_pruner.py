from __future__ import annotations

from collections import defaultdict
from functools import partial
import math
from typing import Dict, List
import torch

from nni.contrib.compression.base.compressor import Pruner
from nni.contrib.compression.base.wrapper import ModuleWrapper
from nni.contrib.compression.utils import Evaluator, ForwardHook
from nni.contrib.compression.pruning.tools.utils import is_active_target


class OBSPruner(Pruner):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict],
                 evaluator: Evaluator | None = None, existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers)
        self.sample_info = defaultdict(dict)
        self.hessian_info = defaultdict(dict)
        self.hooks = list()
        for module_name, ts in self._target_spaces.items():
            for target_name, target_space in ts.items():
                if is_active_target(target_space):
                    self.sample_info[module_name][target_name] = 0
                    self.hessian_info[module_name][target_name] = None

    def add_batch(self, inputs: torch.Tensor, module_name: str):
        """
        Parameters
        ----------
        inputs
            A minibatch of data, the first dimention should be the batch dim,
            the last dimention should be the hidden dim.
            Only support tensor format like (batch_size, ..., input_hidden_dim).
        """
        inputs = inputs.data.reshape(-1, inputs.shape[-1])
        hessian_info = self.hessian_info[module_name]
        for target_name in hessian_info.keys():
            sample_number = self.sample_info[module_name][target_name]
            inp = inputs.T.float() * math.sqrt(2 / (sample_number + inputs.shape[0]))
            hessian = inp.matmul(inp.T)
            if hessian_info[target_name] is None:
                hessian_info[target_name] = hessian
            else:
                hessian_info[target_name] *= sample_number / (sample_number + inputs.shape[0])
                hessian_info[target_name] += hessian
            self.sample_info[module_name][target_name] += inputs.shape[0]

    def generate_masks(self) -> Dict[str, Dict[str, torch.Tensor]]:
        global_blocksize = 128
        masks = defaultdict(dict)
        for module_name, hi in self.hessian_info.items():
            for target_name, hessian in hi.items():
                target_space = self._target_spaces[module_name][target_name]
                block_size = global_blocksize
                sparse_ratio = target_space.sparse_ratio
                if target_space.max_sparse_ratio:
                    sparse_ratio = min(sparse_ratio, target_space.max_sparse_ratio)
                if target_space.min_sparse_ratio:
                    sparse_ratio = max(sparse_ratio, target_space.min_sparse_ratio)

                weight = target_space.target.data.flatten(1).float()
                dead = torch.diag(hessian) == 0
                hessian[dead, dead] = 1
                weight[:, dead] = 0.
                mask = torch.ones_like(weight)

                damp = 0.01 * torch.mean(torch.diag(hessian))
                diag = torch.arange(weight.shape[1], device=hessian.device)
                hessian[diag] += damp

                hessian_inv = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
                hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)

                for col_num in range(0, weight.shape[1], block_size):
                    start, end = col_num, min(col_num + block_size, weight.shape[1])

                    sub_weight = weight[:, start: end].clone()
                    sub_sparse_weight = torch.zeros_like(sub_weight)
                    sub_error = torch.zeros_like(sub_weight)
                    sub_hessian_inv = hessian_inv[start: end, start: end]

                    # generate sub mask
                    metric = (sub_weight ** 2 / (torch.diag(sub_hessian_inv).reshape(1, -1)) ** 2).flatten()
                    pruned_number = int(sparse_ratio * sub_weight.numel())
                    indices = torch.topk(metric, pruned_number, largest=False)[1]
                    sub_mask = torch.ones_like(metric)
                    sub_mask[indices] = 0.
                    sub_mask = sub_mask.reshape_as(sub_weight)

                    # update sub weight with error
                    for j in range(end - start):
                        w, d = sub_weight[:, j], sub_hessian_inv[j, j]
                        q = w.clone()
                        q[sub_mask[:, j] == 0] = 0.

                        sub_sparse_weight[:, j] = q
                        sub_error[:, j] = (w - q) / d
                        sub_weight[:, j:] -= sub_error[:, j].unsqueeze(1).matmul(sub_hessian_inv[j, j:].unsqueeze(0))

                    # update whole weight
                    weight[:, start: end] = sub_sparse_weight
                    weight[:, end:] -= sub_error.matmul(hessian_inv[start: end, end:])
                    mask[:, start: end] = sub_mask

                target_space.target.data = weight.reshape_as(target_space.target.data).to(target_space.target.data.dtype)
                masks[module_name][target_name] = mask

        return masks

    def _single_compress(self, max_steps: None, max_epochs: None):
        return self._fusion_compress(max_steps, max_epochs)

    def _register_add_batch(self, evaluator: Evaluator):
        def collector(buffer: List, module_name):
            def collect_hessian(module, inp, output):
                inp = inp if isinstance(inp, torch.Tensor) else inp[0]
                self.add_batch(inp, module_name)
            return collect_hessian

        for module_name, hi in self.hessian_info.items():
            wrapper = self._module_wrappers[module_name]
            hook = ForwardHook(wrapper.module, wrapper.name, partial(collector, module_name=module_name))
            self.hooks.append(hook)
        evaluator.register_hooks(self.hooks)

    def _register_trigger(self, evaluator: Evaluator):
        pass

    def _fuse_preprocess(self, evaluator: Evaluator):
        self._register_add_batch(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator):
        masks = self.generate_masks()
        self.update_masks(masks)
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()


from pathlib import Path

import numpy as np

import torch
from torch.utils.data import ConcatDataset

import nni
from nni.contrib.compression import TransformersEvaluator

from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertForSequenceClassification, EvalPrediction
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


task_name = 'mnli'


def build_model(pretrained_model_name_or_path: str, task_name: str):
    is_regression = task_name == 'stsb'
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
    return model


def prepare_datasets(task_name: str, tokenizer: BertTokenizerFast, cache_dir: str):
    task_to_keys = {
        'cola': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'qqp': ('question1', 'question2'),
        'rte': ('sentence1', 'sentence2'),
        'sst2': ('sentence', None),
        'stsb': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # used to preprocess the raw data
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if 'label' in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result['labels'] = examples['label']
        return result

    raw_datasets = load_dataset('glue', task_name, cache_dir=cache_dir)
    for key in list(raw_datasets.keys()):
        if 'test' in key:
            raw_datasets.pop(key)

    processed_datasets = raw_datasets.map(preprocess_function, batched=True,
                                          remove_columns=raw_datasets['train'].column_names)

    train_dataset = processed_datasets['train']
    if task_name == 'mnli':
        validation_datasets = {
            'validation_matched': processed_datasets['validation_matched'],
            'validation_mismatched': processed_datasets['validation_mismatched']
        }
    else:
        validation_datasets = {
            'validation': processed_datasets['validation']
        }

    return train_dataset, validation_datasets


def prepare_traced_trainer(model, task_name, load_best_model_at_end=False):
    is_regression = task_name == 'stsb'
    metric = load_metric('glue', task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        result['default'] = result.get('f1', result.get('accuracy', 0.))
        return result

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_dataset, validation_datasets = prepare_datasets(task_name, tokenizer, None)
    merged_validation_dataset = ConcatDataset([d for d in validation_datasets.values()])
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(output_dir='./output/trainer',
                                      do_train=True,
                                      do_eval=True,
                                      evaluation_strategy='steps',
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32,
                                      num_train_epochs=3,
                                      dataloader_num_workers=12,
                                      learning_rate=3e-5,
                                      save_strategy='steps',
                                      save_total_limit=1,
                                      metric_for_best_model='default',
                                      load_best_model_at_end=load_best_model_at_end,
                                      disable_tqdm=True,
                                      optim='adamw_torch',
                                      seed=1024)
    trainer = nni.trace(Trainer)(model=model,
                                 args=training_args,
                                 data_collator=data_collator,
                                 train_dataset=train_dataset,
                                 eval_dataset=merged_validation_dataset,
                                 tokenizer=tokenizer,
                                 compute_metrics=compute_metrics,)
    return trainer


model = build_model('bert-base-uncased', 'mnli')
model.load_state_dict(torch.load('./mnli.bin'))

config_list = [{
    'op_types': ['Linear'],
    'op_names_re': ['bert\.encoder\.*'],
    'sparse_ratio': 0.6
}]

trainer = prepare_traced_trainer(model, 'mnli')
evaluator = TransformersEvaluator(trainer)
pruner = OBSPruner(model, config_list, evaluator)
_, masks = pruner.compress(100, None)

for module_name, ms in masks.items():
    for target_name, mask in ms.items():
        print(module_name, target_name, (1 - mask.sum() / mask.numel()).item())

trainer = prepare_traced_trainer(model, 'mnli')
metric = trainer.evaluate()
print(metric)


model = build_model('bert-base-uncased', 'mnli')
model.load_state_dict(torch.load('./mnli.bin'))

from nni.contrib.compression.pruning import LevelPruner
pruner = LevelPruner(model, config_list)
_, masks = pruner.compress(None, None)

for module_name, ms in masks.items():
    for target_name, mask in ms.items():
        print(module_name, target_name, (1 - mask.sum() / mask.numel()).item())

trainer = prepare_traced_trainer(model, 'mnli')
metric = trainer.evaluate()
print(metric)
