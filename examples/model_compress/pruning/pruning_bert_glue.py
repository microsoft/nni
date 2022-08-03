# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import functools
from pathlib import Path
import sys
import time

from typing import Callable

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# huggingface part
from datasets import load_metric, load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    set_seed
)

import nni
from nni.compression.pytorch import TorchEvaluator
from nni.compression.pytorch.pruning import MovementPruner, TaylorFOWeightPruner
from nni.compression.pytorch.speedup import ModelSpeedup

pretrained_model_name_or_path = 'bert-base-uncased'
task_name = 'mnli'
experiment_id = 'exp_id'
log_dir = Path(f'./pruning_log/{pretrained_model_name_or_path}/{task_name}/{experiment_id}')
model_dir = Path(f'./models/{pretrained_model_name_or_path}/{task_name}')

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(cache_dir='./data', batch_size=32):
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
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
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)

    train_dataset = processed_datasets['train']
    validate_dataset = processed_datasets['validation_matched' if task_name == 'mnli' else 'validation']
    validate_dataset2 = processed_datasets['validation_mismatched'] if task_name == 'mnli' else None

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
    validate_dataloader1 = DataLoader(validate_dataset, collate_fn=data_collator, batch_size=batch_size)
    validate_dataloader2 = DataLoader(validate_dataset2, collate_fn=data_collator, batch_size=batch_size) if task_name == 'mnli' else None

    return train_dataloader, validate_dataloader1, validate_dataloader2


def prepare_model():
    is_regression = task_name == 'stsb'
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)
    return BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)


train_dataloader, validate_dataloader1, validate_dataloader2 = prepare_data()


def training(model: BertForSequenceClassification, optimizer: torch.optim.Optimizer, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, max_steps: int | None = None, max_epochs: int | None = None,
             save_best_model: bool = False, save_path: str | None = None, log_path: str | None = Path(log_dir) / 'training.log', teacher_model: torch.nn.Module | None = None):
    model.train()
    if teacher_model:
        teacher_model.eval()
    current_step = 0
    best_result = 0

    for current_epoch in range(max_epochs if max_epochs else 1):
        for batch in train_dataloader:
            batch.to(device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            if teacher_model:
                distil_loss = F.kl_div(F.log_softmax(outputs.logits / 2, dim=-1), F.softmax(teacher_model(**batch).logits / 2, dim=-1), reduction='batchmean') * (2 ** 2)
                loss = 0.1 * loss + 0.9 * distil_loss
            loss = criterion(loss, None)
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            current_step += 1
            if current_step % 1000 == 0 or current_step % len(train_dataloader) == 0:
                result = evaluation(model)
                with (log_path).open('a+') as f:
                    f.write('[{}] Epoch {}, Step {}: {}\n'.format(time.asctime(time.localtime(time.time())), current_epoch, current_step, result))
                if save_best_model and best_result < result[0]:
                    assert save_path is not None
                    torch.save(model.state_dict(), save_path)
                    best_result = result[0]
            if max_steps and current_step >= max_steps:
                return


def evaluation(model):
    model.eval()
    is_regression = task_name == 'stsb'
    metric = load_metric('glue', task_name)

    for batch in validate_dataloader1:
        batch.to(device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=predictions,
            references=batch['labels'],
        )
    result = metric.compute()

    if validate_dataloader2:
        for batch in validate_dataloader2:
            batch.to(device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=predictions,
                references=batch['labels'],
            )
        result = {'matched': result, 'mismatched': metric.compute()}
        return (result['matched']['accuracy'] + result['mismatched']['accuracy']) / 2, result

    return result.get('f1', result.get('accuracy', None)), result


# using huggingface native loss
def fake_criterion(outputs, targets):
    return outputs


if __name__ == '__main__':
    set_seed(1024)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model = prepare_model().to(device)

    lr = 3e-5
    steps_per_epoch = len(train_dataloader)

    # finetuning model
    finetuned_model_state_path = Path(model_dir) / 'finetuned_model_state.pth'
    if finetuned_model_state_path.exists():
        pretrained_model.load_state_dict(torch.load(finetuned_model_state_path))
    else:
        optimizer = Adam(pretrained_model.parameters(), lr=lr, eps=1e-8)

        def lr_lambda(current_step: int):
            return max(0.0, float(3 * steps_per_epoch - current_step) / float(3 * steps_per_epoch))

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        training(pretrained_model, optimizer, fake_criterion, lr_scheduler=lr_scheduler, max_epochs=3, save_best_model=True, save_path=finetuned_model_state_path)
    finetuned_model = pretrained_model

    # get teacher model
    teacher_model = prepare_model().to(device)
    teacher_model.load_state_dict(torch.load(finetuned_model_state_path))

    # block movement pruning attention
    # make sure you have used nni.trace to wrap the optimizer class before initialize
    total_epochs = 4
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = 1 * steps_per_epoch
    cooldown_steps = 1 * steps_per_epoch

    movement_training = functools.partial(training, log_path=log_dir / 'movement_pruning.log')

    traced_optimizer = nni.trace(Adam)(finetuned_model.parameters(), lr=3e-5, eps=1e-8)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / warmup_steps
        return max(0.0, float(total_steps - current_step) / float(total_steps - warmup_steps))

    traced_scheduler = nni.trace(LambdaLR)(traced_optimizer, lr_lambda)
    evaluator = TorchEvaluator(movement_training, traced_optimizer, fake_criterion, traced_scheduler)
    config_list = [{'op_types': ['Linear'], 'op_partial_names': ['bert.encoder.layer.{}.'.format(i) for i in range(12)], 'sparsity': 0.1}]
    pruner = MovementPruner(model=finetuned_model,
                            config_list=config_list,
                            evaluator=evaluator,
                            training_epochs=total_epochs,
                            warm_up_step=warmup_steps,
                            cool_down_beginning_step=total_steps - cooldown_steps,
                            regular_scale=10,
                            movement_mode='soft',
                            sparse_granularity='auto')
    simulated_pruning_model, attention_masks = pruner.compress()
    pruner.show_pruned_weights()

    torch.save(attention_masks, Path(log_dir) / 'attention_masks.pth')
    del pruner
    del finetuned_model

    # pruning empty head & create config_list for FFN pruning
    attention_pruned_model = prepare_model().to(device)
    attention_pruned_model.load_state_dict(torch.load(finetuned_model_state_path))
    ffn_config_list = []
    attention_masks = torch.load(Path(log_dir) / 'attention_masks.pth')

    layer_count = 0
    module_list = []
    for i in range(0, 12):
        prefix = f'bert.encoder.layer.{i}.'
        value_mask: torch.Tensor = attention_masks[prefix + 'attention.self.value']['weight']
        head_mask = (value_mask.reshape(12, -1).sum(-1) == 0.)
        head_idx = torch.arange(len(head_mask))[head_mask].long().tolist()
        print(f'{i}: {len(head_idx)} {head_idx}')
        if len(head_idx) != 12:
            attention_pruned_model.bert.encoder.layer[i].attention.prune_heads(head_idx)
            module_list.append(attention_pruned_model.bert.encoder.layer[i])
            sparsity = 1 - (1 - len(head_idx) / 12) * 0.5
            sparsity_per_iter = 1 - (1 - sparsity) ** (1 / 12)
            ffn_config_list.append({'op_names': [f'bert.encoder.layer.{layer_count}.intermediate.dense'], 'sparsity': sparsity_per_iter})
            layer_count += 1

    attention_pruned_model.bert.encoder.layer = torch.nn.ModuleList(module_list)
    print(attention_pruned_model)
    print(ffn_config_list)

    # finetuning speedup model
    total_epochs = 5
    optimizer = Adam(attention_pruned_model.parameters(), lr=lr, eps=1e-8)

    def lr_lambda(current_step: int):
        return max(0.0, float(total_epochs * steps_per_epoch - current_step) / float(total_epochs * steps_per_epoch))

    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    at_model_save_path = log_dir / 'attention_pruned_model_state.pth'
    training(attention_pruned_model, optimizer, fake_criterion, lr_scheduler=lr_scheduler, max_epochs=total_epochs, save_best_model=True, save_path=at_model_save_path, teacher_model=teacher_model)

    attention_pruned_model.load_state_dict(torch.load(at_model_save_path))

    # pruning FFN with TaylorFOWeightPruner

    distil_training = functools.partial(training, teacher_model=teacher_model, log_path=log_dir / 'taylor_pruning.log')
    traced_optimizer = nni.trace(Adam)(attention_pruned_model.parameters(), lr=3e-5, eps=1e-8)
    evaluator = TorchEvaluator(distil_training, traced_optimizer, fake_criterion)

    attention_pruned_model.train()
    current_step = 0
    best_result = 0
    total_epochs = 4
    init_lr = 3e-5

    for current_epoch in range(total_epochs):
        for batch in train_dataloader:
            # pruning 12 times
            if current_step % 2000 == 0 and current_step < 24000:
                check_point = attention_pruned_model.state_dict()
                pruner = TaylorFOWeightPruner(attention_pruned_model, ffn_config_list, evaluator, 1000)
                _, ffn_masks = pruner.compress()
                renamed_ffn_masks = {}
                for model_name, targets_mask in ffn_masks.items():
                    renamed_ffn_masks[model_name.split('bert.encoder.')[1]] = targets_mask
                pruner._unwrap_model()
                attention_pruned_model.load_state_dict(check_point)
                ModelSpeedup(attention_pruned_model.bert.encoder, torch.rand(8, 128, 768).to(device), renamed_ffn_masks).speedup_model()
                optimizer = Adam(attention_pruned_model.parameters(), lr=init_lr)
            batch.to(device)
            optimizer.zero_grad()
            # manually schedule lr
            for params_group in optimizer.param_groups:
                params_group['lr'] = (1 - current_step / (total_epochs * steps_per_epoch)) * init_lr
            outputs = attention_pruned_model(**batch)
            loss = outputs.loss
            if teacher_model:
                distil_loss = F.kl_div(F.log_softmax(outputs.logits / 2, dim=-1), F.softmax(teacher_model(**batch).logits / 2, dim=-1), reduction='batchmean') * (2 ** 2)
                loss = 0.1 * loss + 0.9 * distil_loss
            loss.backward()
            optimizer.step()
            current_step += 1
            if current_step % 1000 == 0 or current_step % len(train_dataloader) == 0:
                result = evaluation(attention_pruned_model)
                with (log_dir / 'ffn_pruning.log').open('a+') as f:
                    f.write('[{}] Epoch {}, Step {}: {}\n'.format(time.asctime(time.localtime(time.time())), current_epoch, current_step, result))
                if current_step >= 24000 and best_result < result[0]:
                    torch.save(attention_pruned_model, log_dir / 'best_model.pth')
                    best_result = result[0]
                    with (log_dir / 'ffn_pruning.log').open('a+') as f:
                        original_stdout = sys.stdout
                        sys.stdout = f
                        print(attention_pruned_model)
                        sys.stdout = original_stdout
