"""
Pruning Bert on Task MNLI
=========================

Workable Pruning Process
------------------------

Here we show an effective transformer pruning process that NNI team has tried, and users can use NNI to discover better processes.

The entire pruning process can be divided into the following steps:

1. Finetune the pre-trained model on the downstream task. From our experience,
   the final performance of pruning on the finetuned model is better than pruning directly on the pre-trained model.
   At the same time, the finetuned model obtained in this step will also be used as the teacher model for the following
   distillation training.
2. Pruning the attention layer at first. Here we apply block-sparse on attention layer weight,
   and directly prune the head (condense the weight) if the head was fully masked.
   If the head was partially masked, we will not prune it and recover its weight.
3. Retrain the head-pruned model with distillation. Recover the model precision before pruning FFN layer.
4. Pruning the FFN layer. Here we apply the output channels pruning on the 1st FFN layer,
   and the 2nd FFN layer input channels will be pruned due to the pruning of 1st layer output channels.
5. Retrain the final pruned model with distillation.

During the process of pruning transformer, we gained some of the following experiences:

* We using :ref:`movement-pruner` in step 2 and :ref:`taylor-fo-weight-pruner` in step 4. :ref:`movement-pruner` has good performance on attention layers,
  and :ref:`taylor-fo-weight-pruner` method has good performance on FFN layers. These two pruners are all some kinds of gradient-based pruning algorithms,
  we also try weight-based pruning algorithms like :ref:`l1-norm-pruner`, but it doesn't seem to work well in this scenario.
* Distillation is a good way to recover model precision. In terms of results, usually 1~2% improvement in accuracy can be achieved when we prune bert on mnli task.
* It is necessary to gradually increase the sparsity rather than reaching a very high sparsity all at once.

Experiment
----------

The complete pruning process takes about 8 hours on one A100.

Preparation
^^^^^^^^^^^

.. note::

    Please set ``dev_mode`` to ``False`` to run this tutorial. Here ``dev_mode`` is ``True`` by default is for generating documents.

"""

dev_mode = True

# %%
# Some basic setting.

from pathlib import Path
from typing import Callable

pretrained_model_name_or_path = 'bert-base-uncased'
task_name = 'mnli'
experiment_id = 'pruning_bert'

# heads_num and layers_num should align with pretrained_model_name_or_path
heads_num = 12
layers_num = 12

# used to save the experiment log
log_dir = Path(f'./pruning_log/{pretrained_model_name_or_path}/{task_name}/{experiment_id}')
log_dir.mkdir(parents=True, exist_ok=True)

# used to save the finetuned model and share between different experiemnts with same pretrained_model_name_or_path and task_name
model_dir = Path(f'./models/{pretrained_model_name_or_path}/{task_name}')
model_dir.mkdir(parents=True, exist_ok=True)

from transformers import set_seed
set_seed(1024)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# The function used to create dataloaders, note that ``mnli`` has two evaluation dataset.
# If ``teacher_model`` is set, will run all dataset on teacher model to get the ``teacher_logits`` for distillation.

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding

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

def prepare_data(cache_dir='./data', train_batch_size=32, eval_batch_size=32,
                    teacher_model: torch.nn.Module = None):
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    data_collator = DataCollatorWithPadding(tokenizer)

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

    # if has teacher model, add 'teacher_logits' to datasets who has 'labels'.
    # 'teacher_logits' is used for distillation and avoid the double counting.
    if teacher_model:
        teacher_model_training = teacher_model.training
        teacher_model.eval()
        model_device = next(teacher_model.parameters()).device

        def add_teacher_logits(examples):
            result = {k: v for k, v in examples.items()}
            samples = data_collator(result).to(model_device)
            if 'labels' in samples:
                with torch.no_grad():
                    logits = teacher_model(**samples).logits.tolist()
                result['teacher_logits'] = logits
            return result

        processed_datasets = processed_datasets.map(add_teacher_logits, batched=True,
                                                    batch_size=train_batch_size)
        teacher_model.train(teacher_model_training)

    train_dataset = processed_datasets['train']
    validation_dataset = processed_datasets['validation_matched' if task_name == 'mnli' else 'validation']
    validation_dataset2 = processed_datasets['validation_mismatched'] if task_name == 'mnli' else None

    train_dataloader = DataLoader(train_dataset,
                                    shuffle=True,
                                    collate_fn=data_collator,
                                    batch_size=train_batch_size)
    validation_dataloader = DataLoader(validation_dataset,
                                        collate_fn=data_collator,
                                        batch_size=eval_batch_size)
    validation_dataloader2 = DataLoader(validation_dataset2,
                                        collate_fn=data_collator,
                                        batch_size=eval_batch_size) if task_name == 'mnli' else None

    return train_dataloader, validation_dataloader, validation_dataloader2

# %%
# Training function & evaluation function.

import time
import torch.nn.functional as F
from datasets import load_metric

def training(train_dataloader: DataLoader,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                max_steps: int = None, max_epochs: int = None,
                save_best_model: bool = False, save_path: str = None,
                log_path: str = Path(log_dir) / 'training.log',
                distillation: bool = False,
                evaluation_func=None):
    model.train()
    current_step = 0
    best_result = 0

    for current_epoch in range(max_epochs if max_epochs else 1):
        for batch in train_dataloader:
            batch.to(device)
            teacher_logits = batch.pop('teacher_logits', None)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            if distillation:
                assert teacher_logits is not None
                distil_loss = F.kl_div(F.log_softmax(outputs.logits / 2, dim=-1),
                                        F.softmax(teacher_logits / 2, dim=-1), reduction='batchmean') * (2 ** 2)
                loss = 0.1 * loss + 0.9 * distil_loss

            loss = criterion(loss, None)
            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            current_step += 1

            # evaluation for every 1000 steps
            if current_step % 1000 == 0 or current_step % len(train_dataloader) == 0:
                result = evaluation_func(model) if evaluation_func else None
                with (log_path).open('a+') as f:
                    msg = '[{}] Epoch {}, Step {}: {}\n'.format(time.asctime(time.localtime(time.time())), current_epoch, current_step, result)
                    f.write(msg)
                # if it's the best model, save it.
                if save_best_model and best_result < result['default']:
                    assert save_path is not None
                    torch.save(model.state_dict(), save_path)
                    best_result = result['default']

            if max_steps and current_step >= max_steps:
                return

def evaluation(validation_dataloader: DataLoader,
                validation_dataloader2: DataLoader,
                model: torch.nn.Module):
    training = model.training
    model.eval()
    is_regression = task_name == 'stsb'
    metric = load_metric('glue', task_name)

    for batch in validation_dataloader:
        batch.pop('teacher_logits', None)
        batch.to(device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=predictions,
            references=batch['labels'],
        )
    result = metric.compute()

    if validation_dataloader2:
        for batch in validation_dataloader2:
            batch.pop('teacher_logits', None)
            batch.to(device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=predictions,
                references=batch['labels'],
            )
        result = {'matched': result, 'mismatched': metric.compute()}
        result['default'] = (result['matched']['accuracy'] + result['mismatched']['accuracy']) / 2
    else:
        result['default'] = result.get('f1', result.get('accuracy', None))

    model.train(training)
    return result

# using huggingface native loss
def fake_criterion(outputs, targets):
    return outputs


# %%
# Prepare pre-trained model and finetuning on downstream task.

import functools

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertForSequenceClassification

def create_pretrained_model():
    is_regression = task_name == 'stsb'
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)
    return BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)

def create_finetuned_model():
    pretrained_model = create_pretrained_model().to(device)

    train_dataloader, validation_dataloader, validation_dataloader2 = prepare_data()
    evaluation_func = functools.partial(evaluation, validation_dataloader, validation_dataloader2)
    steps_per_epoch = len(train_dataloader)
    training_epochs = 3

    finetuned_model_state_path = Path(model_dir) / 'finetuned_model_state.pth'

    if finetuned_model_state_path.exists():
        pretrained_model.load_state_dict(torch.load(finetuned_model_state_path))
    elif dev_mode:
        pass
    else:
        optimizer = Adam(pretrained_model.parameters(), lr=3e-5, eps=1e-8)

        def lr_lambda(current_step: int):
            return max(0.0, float(training_epochs * steps_per_epoch - current_step) / float(training_epochs * steps_per_epoch))

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        training(train_dataloader, pretrained_model, optimizer, fake_criterion, lr_scheduler=lr_scheduler, max_epochs=training_epochs,
                    save_best_model=True, save_path=finetuned_model_state_path, evaluation_func=evaluation_func)
    return pretrained_model

finetuned_model = create_finetuned_model()

# %%
# Using finetuned model as teacher model to create dataloader.
# Add 'teacher_logits' to dataset, it is used to do the distillation, it can be seen as a kind of data label.

if not dev_mode:
    train_dataloader, validation_dataloader, validation_dataloader2 = prepare_data(teacher_model=finetuned_model)
else:
    train_dataloader, validation_dataloader, validation_dataloader2 = prepare_data()

evaluation_func = functools.partial(evaluation, validation_dataloader, validation_dataloader2)

# %%
# Pruning
# ^^^^^^^
# First, using MovementPruner to prune attention head.

steps_per_epoch = len(train_dataloader)

# Set training steps/epochs for pruning.

if not dev_mode:
    total_epochs = 4
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = 1 * steps_per_epoch
    cooldown_steps = 1 * steps_per_epoch
else:
    total_epochs = 1
    total_steps = 3
    warmup_steps = 1
    cooldown_steps = 1

# Initialize evaluator used by MovementPruner.

import nni
from nni.algorithms.compression.v2.pytorch import TorchEvaluator

movement_training = functools.partial(training, train_dataloader, log_path=log_dir / 'movement_pruning.log',
                                    evaluation_func=evaluation_func)
traced_optimizer = nni.trace(Adam)(finetuned_model.parameters(), lr=3e-5, eps=1e-8)

def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / warmup_steps
    return max(0.0, float(total_steps - current_step) / float(total_steps - warmup_steps))

traced_scheduler = nni.trace(LambdaLR)(traced_optimizer, lr_lambda)
evaluator = TorchEvaluator(movement_training, traced_optimizer, fake_criterion, traced_scheduler)

# Apply block-soft-movement pruning on attention layers.

from nni.compression.pytorch.pruning import MovementPruner

config_list = [{'op_types': ['Linear'], 'op_partial_names': ['bert.encoder.layer.{}.'.format(i) for i in range(layers_num)], 'sparsity': 0.1}]
pruner = MovementPruner(model=finetuned_model,
                        config_list=config_list,
                        evaluator=evaluator,
                        training_epochs=total_epochs,
                        training_steps=total_steps,
                        warm_up_step=warmup_steps,
                        cool_down_beginning_step=total_steps - cooldown_steps,
                        regular_scale=10,
                        movement_mode='soft',
                        sparse_granularity='auto')
_, attention_masks = pruner.compress()
pruner.show_pruned_weights()

torch.save(attention_masks, Path(log_dir) / 'attention_masks.pth')

# %%
# Load a new finetuned model to do the speedup.
# Note that nni speedup don't support replace attention module, so here we manully replace the attention module.
#
# If the head is entire masked, physically prune it and create config_list for FFN pruning.

attention_pruned_model = create_finetuned_model().to(device)
attention_masks = torch.load(Path(log_dir) / 'attention_masks.pth')

ffn_config_list = []
layer_count = 0
module_list = []
for i in range(0, layers_num):
    prefix = f'bert.encoder.layer.{i}.'
    value_mask: torch.Tensor = attention_masks[prefix + 'attention.self.value']['weight']
    head_mask = (value_mask.reshape(heads_num, -1).sum(-1) == 0.)
    head_idx = torch.arange(len(head_mask))[head_mask].long().tolist()
    print(f'layer {i} pruner {len(head_idx)} head: {head_idx}')
    if len(head_idx) != heads_num:
        attention_pruned_model.bert.encoder.layer[i].attention.prune_heads(head_idx)
        module_list.append(attention_pruned_model.bert.encoder.layer[i])
        # The final ffn weight remaining ratio is the half of the attention weight remaining ratio.
        # This is just an empirical configuration, you can use any other method to determine this sparsity.
        sparsity = 1 - (1 - len(head_idx) / heads_num) * 0.5
        # here we use a simple sparsity schedule, we will prune ffn in 12 iterations, each iteration prune `sparsity_per_iter`.
        sparsity_per_iter = 1 - (1 - sparsity) ** (1 / heads_num)
        ffn_config_list.append({'op_names': [f'bert.encoder.layer.{layer_count}.intermediate.dense'], 'sparsity': sparsity_per_iter})
        layer_count += 1

attention_pruned_model.bert.encoder.layer = torch.nn.ModuleList(module_list)

# %%
# Retrain the attention pruned model with distillation.

if not dev_mode:
    total_epochs = 5
    total_steps = None
    distillation = True
else:
    total_epochs = 1
    total_steps = 1
    distillation = False

optimizer = Adam(attention_pruned_model.parameters(), lr=3e-5, eps=1e-8)

def lr_lambda(current_step: int):
    return max(0.0, float(total_epochs * steps_per_epoch - current_step) / float(total_epochs * steps_per_epoch))

lr_scheduler = LambdaLR(optimizer, lr_lambda)
at_model_save_path = log_dir / 'attention_pruned_model_state.pth'
training(train_dataloader, attention_pruned_model, optimizer, fake_criterion, lr_scheduler=lr_scheduler,
         max_epochs=total_epochs, max_steps=total_steps, save_best_model=True, save_path=at_model_save_path,
         distillation=distillation, evaluation_func=evaluation_func)

if not dev_mode:
    attention_pruned_model.load_state_dict(torch.load(at_model_save_path))

# %%
# Iterative pruning FFN with TaylorFOWeightPruner in 12 iterations.
# Finetuning 2000 steps after each iteration, then finetuning 2 epochs after pruning finished.
# 
# NNI will support per-step-pruning-schedule in the future, then can use an pruner to replace the following code.

if not dev_mode:
    total_epochs = 4
    total_steps = None
    taylor_pruner_steps = 1000
    steps_per_iteration = 2000
    total_pruning_steps = 24000
    distillation = True
else:
    total_epochs = 1
    total_steps = 6
    taylor_pruner_steps = 2
    steps_per_iteration = 2
    total_pruning_steps = 4
    distillation = False

from nni.compression.pytorch.pruning import TaylorFOWeightPruner
from nni.compression.pytorch.speedup import ModelSpeedup

distil_training = functools.partial(training, train_dataloader, log_path=log_dir / 'taylor_pruning.log',
                                    distillation=distillation, evaluation_func=evaluation_func)
traced_optimizer = nni.trace(Adam)(attention_pruned_model.parameters(), lr=3e-5, eps=1e-8)
evaluator = TorchEvaluator(distil_training, traced_optimizer, fake_criterion)

current_step = 0
best_result = 0
init_lr = 3e-5

dummy_input = torch.rand(8, 128, 768).to(device)

attention_pruned_model.train()
for current_epoch in range(total_epochs):
    for batch in train_dataloader:
        if total_steps and current_step >= total_steps:
            break
        # pruning 12 times
        if current_step % steps_per_iteration == 0 and current_step < total_pruning_steps:
            check_point = attention_pruned_model.state_dict()
            pruner = TaylorFOWeightPruner(attention_pruned_model, ffn_config_list, evaluator, taylor_pruner_steps)
            _, ffn_masks = pruner.compress()
            renamed_ffn_masks = {}
            # rename the masks keys, because we only speedup the bert.encoder
            for model_name, targets_mask in ffn_masks.items():
                renamed_ffn_masks[model_name.split('bert.encoder.')[1]] = targets_mask
            pruner._unwrap_model()
            attention_pruned_model.load_state_dict(check_point)
            ModelSpeedup(attention_pruned_model.bert.encoder, dummy_input, renamed_ffn_masks).speedup_model()
            optimizer = Adam(attention_pruned_model.parameters(), lr=init_lr)

        batch.to(device)
        teacher_logits = batch.pop('teacher_logits', None)
        optimizer.zero_grad()

        # manually schedule lr
        for params_group in optimizer.param_groups:
            params_group['lr'] = (1 - current_step / (total_epochs * steps_per_epoch)) * init_lr

        outputs = attention_pruned_model(**batch)
        loss = outputs.loss

        # distillation
        if teacher_logits is not None:
            distil_loss = F.kl_div(F.log_softmax(outputs.logits / 2, dim=-1),
                                    F.softmax(teacher_logits / 2, dim=-1), reduction='batchmean') * (2 ** 2)
            loss = 0.1 * loss + 0.9 * distil_loss
        loss.backward()
        optimizer.step()

        current_step += 1
        if current_step % 1000 == 0 or current_step % len(train_dataloader) == 0:
            result = evaluation_func(attention_pruned_model)
            with (log_dir / 'ffn_pruning.log').open('a+') as f:
                msg = '[{}] Epoch {}, Step {}: {}\n'.format(time.asctime(time.localtime(time.time())),
                                                            current_epoch, current_step, result)
                f.write(msg)
            if current_step >= total_pruning_steps and best_result < result['default']:
                torch.save(attention_pruned_model, log_dir / 'best_model.pth')
                best_result = result['default']

# %%
# Result
# ------
# The speedup is test on the entire validation dataset with batch size 32 on A100.
# We test under two pytorch version and found the latency varying widely.
# 
# Setting 1: pytorch 1.12.1
#
# Setting 2: pytorch 1.10.0
# 
# .. list-table:: Prune Bert-base-uncased on MNLI
#     :header-rows: 1
#     :widths: auto
#
#     * - Attention Pruning Method
#       - FFN Pruning Method
#       - Total Sparsity
#       - Accuracy
#       - Acc. Drop
#       - Speedup (S1)
#       - Speedup (S2)
#     * -
#       -
#       - 0%
#       - 84.73 / 84.63
#       - +0.0 / +0.0
#       - 12.56s (x1.00)
#       - 4.05s (x1.00)
#     * - :ref:`movement-pruner` (soft, sparsity=0.1, regular_scale=5)
#       - :ref:`taylor-fo-weight-pruner`
#       - 51.39%
#       - 84.25 / 84.96
#       - -0.48 / +0.33
#       - 6.85s (x1.83)
#       - 2.7s (x1.50)
#     * - :ref:`movement-pruner` (soft, sparsity=0.1, regular_scale=10)
#       - :ref:`taylor-fo-weight-pruner`
#       - 66.67%
#       - 83.98 / 83.75
#       - -0.75 / -0.88
#       - 4.73s (x2.66)
#       - 2.16s (x1.86)
#     * - :ref:`movement-pruner` (soft, sparsity=0.1, regular_scale=20)
#       - :ref:`taylor-fo-weight-pruner`
#       - 77.78%
#       - 83.02 / 83.06
#       - -1.71 / -1.57
#       - 3.35s (x3.75)
#       - 1.72s (x2.35)
#     * - :ref:`movement-pruner` (soft, sparsity=0.1, regular_scale=30)
#       - :ref:`taylor-fo-weight-pruner`
#       - 87.04%
#       - 81.24 / 80.99
#       - -3.49 / -3.64
#       - 2.19s (x5.74)
#       - 1.31s (x3.09)
