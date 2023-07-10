from pathlib import Path
import argparse
import sys

import numpy as np

import torch
from torch.utils.data import ConcatDataset

import nni

from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertForSequenceClassification, EvalPrediction
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


task_name = 'mnli'
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
                                      fp16=False,
                                      learning_rate=3e-5,
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
                                      eval_steps=100,
                                      deepspeed="./bert_ds_config.json",
                                      load_best_model_at_end=load_best_model_at_end,)
    # if is_quant:
    #     training_args.learning_rate = quant_lr
    # else:
    #     training_args.learning_rate = finetune_lr
    print(f"=== learning_rate:{training_args.learning_rate} ====")
    trainer = nni.trace(Trainer)(model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=merged_validation_dataset,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        )

    return trainer


def build_finetuning_model(state_dict_path: str, is_quant=False):
    model = build_model('bert-base-uncased', task_name)
    if Path(state_dict_path).exists():
        model.load_state_dict(torch.load(state_dict_path))
        print(f"==== load finetune model directly ====")
    else:
        trainer = prepare_traced_trainer(model, True, is_quant)
        trainer.train()
        torch.save(model.state_dict(), state_dict_path)
    return model


import nni
from nni.compression.quantization import QATQuantizer, LsqQuantizer, PtqQuantizer
from nni.compression.utils import TransformersEvaluator


def fake_quantize():
    config_list = [{
        'op_types': ['Linear'],
        'op_names_re': ['bert.encoder.layer.{}'.format(i) for i in range(12)],
        'target_names': ['weight', '_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'symmetric',
        'granularity': 'default',
    }]

    # create a finetune model
    Path('./output/bert_finetuned/').mkdir(parents=True, exist_ok=True)
    model: torch.nn.Module = build_finetuning_model(f'./output/{task_name}.bin', is_quant=False)  # type: ignore
    traced_trainer = prepare_traced_trainer(model, is_quant=False)
    evaluator = TransformersEvaluator(traced_trainer)
    if quant_method == 'lsq':
        quantizer = LsqQuantizer(model, config_list, evaluator)
        model, calibration_config = quantizer.compress(max_steps=None, max_epochs=quant_max_epochs)
    elif quant_method == 'qat':
        quantizer = QATQuantizer(model, config_list, evaluator, 10)
        model, calibration_config = quantizer.compress(max_steps=None, max_epochs=quant_max_epochs)
    elif quant_method == 'ptq':
        quantizer = PtqQuantizer(model, config_list, evaluator)
        model, calibration_config = quantizer.compress(max_steps=1, max_epochs=None)
    else:
        raise ValueError(f"quantization method {quant_method} is not supported")
    print(calibration_config)
    # evaluate the performance of the fake quantize model
    quantizer.evaluator.bind_model(model, quantizer._get_param_names_map())


def evaluate():
    model = build_finetuning_model(f'./output/{task_name}.bin', is_quant=False)
    trainer = prepare_traced_trainer(model, is_quant=False)
    metrics = trainer.evaluate()
    print(f"Evaluate metrics={metrics}")


fake_quantize()
# evaluate()
