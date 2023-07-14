"""
Quantize BERT on Task GLUE
==========================

Here we show an effective transformer simulated quantization process that NNI team has tried, and users can use NNI to discover better process.

We use the BERT model and the trainer pipeline in the Transformers to do some experiments.
The entire quantization process can be divided into the following steps:

1. Use the BERT-base-uncased model and the trainer pipeline in the transformers to fine-tune the model on the downstream task GLUE.
   From our experience, the final performance of quantization on the finetuned model is
   better than quantization directly on the pre-trained model.
2. Use a specific quantizer to quantize the finetuned model on the GLUE.
   Here we apply QAT, LSQ and PTQ quantizers to quantize the BERT model so that 
   we can compare their performance of the quantized BERT on the GLUE.
   Among them, LSQ and QAT are quantization aware training methods, and PTQ is a post-training quantization method.

During the process of quantizing BERT:

* we use the BERT model and the trainer pipeline in the Transformers to do some experiments.
* we use int8 to quantize Linear layers in the BERT.encoder.

Experiment
----------

Preparation
^^^^^^^^^^^

This section is mainly for fine-tuning model on the downstream task GLUE.
If you are familiar with how to finetune BERT on GLUE dataset, you can skip this section.

1. Load the tokenizer and BERT model from Huggingface transformers.
2. Create a trainer instance to fine-tune the BERT model.

.. note::

    Please set ``dev_mode`` to ``False`` to run this tutorial. Here ``dev_mode`` is ``True`` by default is for generating documents.

"""

from pathlib import Path
import argparse

import numpy as np

import torch
from torch.utils.data import ConcatDataset

import nni

from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertForSequenceClassification, EvalPrediction
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


task_name = 'qnli'
finetune_lr = 4e-5
quant_lr = 1e-5
quant_method = 'lsq'
dev_mode = True

if dev_mode:
    quant_max_epochs = 1
    finetune_max_epochs = 1
else:
    quant_max_epochs = 10
    finetune_max_epochs = 10


# %%
# Load the pre-trained model from the transformers

def build_model(pretrained_model_name_or_path: str, task_name: str):
    is_regression = task_name == 'stsb'
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
    return model

# %%
# Create datasets on the specific task GLUE

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

# %%
# Create a trainer instance
#
# .. note::
#
#     Please set ``is_quant`` to ``False`` to fine-tune the BERT model and set ``is_quant`` to ``True``
#     , when you need to create a traced trainer and use ``quant_lr`` for model quantization.

def prepare_traced_trainer(model, load_best_model_at_end=False, is_quant=False):
    is_regression = task_name == 'stsb'
    metric = load_metric('glue', task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        result['default'] = result.get('f1', result.get('accuracy', 0.))
        return result

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_dataset, validation_datasets = prepare_datasets(task_name, tokenizer, '')
    merged_validation_dataset = ConcatDataset([d for d in validation_datasets.values()]) # type: ignore
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(output_dir='./output/trainer',
                                      do_train=True,
                                      do_eval=True,
                                      evaluation_strategy='steps',
                                      per_device_train_batch_size=128, #128,
                                      per_device_eval_batch_size=128, #128,
                                      num_train_epochs=finetune_max_epochs,
                                      dataloader_num_workers=12,
                                      save_strategy='steps',
                                      save_total_limit=1,
                                      metric_for_best_model='default',
                                      greater_is_better=True,
                                      seed=1024,
                                      load_best_model_at_end=load_best_model_at_end,)
    if is_quant:
        training_args.learning_rate = quant_lr
    else:
        training_args.learning_rate = finetune_lr
    trainer = nni.trace(Trainer)(model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=merged_validation_dataset,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        )

    return trainer

# %%
# Create the finetuned model

def build_finetuning_model(state_dict_path: str, is_quant=False):
    model = build_model('bert-base-uncased', task_name)
    if Path(state_dict_path).exists():
        model.load_state_dict(torch.load(state_dict_path))
    else:
        trainer = prepare_traced_trainer(model, True, is_quant)
        trainer.train()
        torch.save(model.state_dict(), state_dict_path)
    return model

# %%
# Quantization
# ^^^^^^^^^^^^
# After fine-tuning the BERT model on the specific task GLUE, a specific quantizer instsance can be created
# to process quantization aware training or post-training quantization with BERT on the GLUE.
#
# The entire quantization process can be devided into the following steps:
#
# 1. Call ``build_finetuning_model`` to load or fine-tune the BERT model on a specific task GLUE
# 2. Call ``prepare_traced_trainer`` and set ``is_quant`` to ``True`` to create a traced trainer instance for model quantization
# 3. Call the TransformersEvaluator to create an evaluator instance
# 4. Use the defined config_list and evaluator to create a quantizer instance
# 5. Define ``max_steps`` or ``max_epochs``. Note that ``max_steps`` and ``max_epochs`` cannot be None at the same time.
# 6. Call ``quantizer.compress(max_steps, max_epochs)`` to execute the simulated quantization process

import nni
from nni.compression.quantization import QATQuantizer, LsqQuantizer, PtqQuantizer
from nni.compression.utils import TransformersEvaluator

def fake_quantize():
    config_list = [{
        'op_types': ['Linear'],
        'op_names_re': ['bert.encoder.layer.{}'.format(i) for i in range(12)],
        'target_names': ['weight', '_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine',
        'granularity': 'default',
    }]

    # create a finetune model
    Path('./output/bert_finetuned/').mkdir(parents=True, exist_ok=True)
    model: torch.nn.Module = build_finetuning_model(f'./output/bert_finetuned/{task_name}.bin', is_quant=False)  # type: ignore
    traced_trainer = prepare_traced_trainer(model, is_quant=False)
    evaluator = TransformersEvaluator(traced_trainer)
    if quant_method == 'lsq':
        quantizer = LsqQuantizer(model, config_list, evaluator)
        model, calibration_config = quantizer.compress(max_steps=None, max_epochs=quant_max_epochs)
    elif quant_method == 'qat':
        quantizer = QATQuantizer(model, config_list, evaluator, 1000)
        model, calibration_config = quantizer.compress(max_steps=None, max_epochs=quant_max_epochs)
    elif quant_method == 'ptq':
        quantizer = PtqQuantizer(model, config_list, evaluator)
        model, calibration_config = quantizer.compress(max_steps=1, max_epochs=None)
    else:
        raise ValueError(f"quantization method {quant_method} is not supported")
    print(calibration_config)
    # evaluate the performance of the fake quantize model
    quantizer.evaluator.bind_model(model, quantizer._get_param_names_map())
    print(quantizer.evaluator.evaluate())

def evaluate():
    model = build_finetuning_model(f'./output/bert_finetuned/{task_name}.bin', is_quant=False)
    trainer = prepare_traced_trainer(model, is_quant=False)
    metrics = trainer.evaluate()
    print(f"Evaluate metrics={metrics}")


skip_exec = True
if not skip_exec:
    fake_quantize()
    evaluate()


# %%
# Result
# ------
# We experimented with PTQ, LSQ, and QAT algorithms on the MNLI, QNLI, QQP and  MRPC datasets respectively on an A100, and the experimental results are as follows.
#
# .. list-table:: Quantize Bert-base-uncased on MNLI, QNLI, MRPC and QQP
#     :header-rows: 1
#     :widths: auto
#
#     * - Quant Method
#       - MNLI
#       - QNLI
#       - MRPC
#       - QQP
#     * - Metrics
#       - ACC
#       - ACC
#       - F1
#       - F1
#     * - Baseline
#       - 85.04%
#       - 91.67%
#       - 87.69%
#       - 88.42%
#     * - LSQ
#       - 84.34%
#       - 91.69%
#       - 89.9%
#       - 88.16%
#     * - QAT
#       - 83.68%
#       - 90.52%
#       - 89.16%
#       - 87.62%
#     * - PTQ
#       - 76.37%
#       - 67.67%
#       - 74.79%
#       - 84.42%
