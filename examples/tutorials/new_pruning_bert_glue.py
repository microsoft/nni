from __future__ import annotations

from pathlib import Path

import numpy as np

import torch
from torch.utils.data import ConcatDataset, DataLoader

import nni

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


def prepare_dataloaders(task_name: str, cache_dir: str, train_batch_size: int = 32, eval_batch_size: int = 32):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_dataset, validation_datasets = prepare_datasets(task_name, tokenizer, cache_dir)
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
    validation_dataloaders = {
        val_name: DataLoader(val_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
            for val_name, val_dataset in validation_datasets.items()
    }
    return train_dataloader, validation_dataloaders


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


def build_finetuning_model(task_name: str, state_dict_path: str):
    model = build_model('bert-base-uncased', task_name)
    if Path(state_dict_path).exists():
        model.load_state_dict(torch.load(state_dict_path))
    else:
        trainer = prepare_traced_trainer(model, task_name, True)
        trainer.train()
        torch.save(model.state_dict(), state_dict_path)
    return model


from nni.contrib.compression.distillation import DynamicLayerwiseDistiller, Adaptive1dLayerwiseDistiller
from nni.contrib.compression.utils import TransformersEvaluator

def dynamic_distiller(student_model: BertForSequenceClassification, teacher_model: BertForSequenceClassification,
                      student_trainer: Trainer):
    layer_num = len(student_model.bert.encoder.layer)
    config_list = [{
        'op_names': [f'bert.encoder.layer.{i}'],
        'link': [f'bert.encoder.layer.{j}' for j in range(i, layer_num)],
        'lambda': 0.9,
        'apply_method': 'mse',
    } for i in range(layer_num)]
    config_list.append({
        'op_names': ['classifier'],
        'link': ['classifier'],
        'lambda': 0.9,
        'apply_method': 'kl',
    })

    evaluator = TransformersEvaluator(student_trainer)

    def teacher_predict(batch, teacher_model):
        return teacher_model(**batch)

    return DynamicLayerwiseDistiller(student_model, config_list, evaluator, teacher_model, teacher_predict, origin_loss_lambda=0.1)


def dynamic_distillation(student_model: BertForSequenceClassification, teacher_model: BertForSequenceClassification,
                         max_steps: int | None, max_epochs: int | None):
    student_trainer = prepare_traced_trainer(student_model, task_name, True)

    ori_teacher_device = teacher_model.device
    training = teacher_model.training
    teacher_model.to(student_trainer.args.device).eval()

    distiller = dynamic_distiller(student_model, teacher_model, student_trainer)
    distiller.compress(max_steps, max_epochs)
    distiller.unwrap_model()

    teacher_model.to(ori_teacher_device).train(training)


def adapt_distiller(student_model: BertForSequenceClassification, teacher_model: BertForSequenceClassification,
                    student_trainer: Trainer):
    layer_num = len(student_model.bert.encoder.layer)
    config_list = [{
        'op_names': [f'bert.encoder.layer.{i}'],
        'lambda': 0.9,
        'apply_method': 'mse',
    } for i in range(layer_num)]
    config_list.append({
        'op_names': ['classifier'],
        'link': ['classifier'],
        'lambda': 0.9,
        'apply_method': 'kl',
    })

    evaluator = TransformersEvaluator(student_trainer)

    def teacher_predict(batch, teacher_model):
        return teacher_model(**batch)

    return Adaptive1dLayerwiseDistiller(student_model, config_list, evaluator, teacher_model, teacher_predict, origin_loss_lambda=0.1)


def adapt_distillation(student_model: BertForSequenceClassification, teacher_model: BertForSequenceClassification,
                       max_steps: int | None, max_epochs: int | None):
    student_trainer = prepare_traced_trainer(student_model, task_name, True)

    ori_teacher_device = teacher_model.device
    training = teacher_model.training
    teacher_model.to(student_trainer.args.device).eval()

    distiller = adapt_distiller(student_model, teacher_model, student_trainer)
    dummy_input = (torch.randint(0, 10000, [8, 128]), torch.randint(0, 2, [8, 128]), torch.randint(0, 2, [8, 128]))
    dummy_input = [_.to(student_trainer.args.device) for _ in dummy_input]
    distiller.track_forward(*dummy_input)

    distiller.compress(max_steps, max_epochs)
    distiller.unwrap_model()

    teacher_model.to(ori_teacher_device).train(training)


from nni.contrib.compression.pruning import MovementPruner
from nni.compression.pytorch.speedup.v2 import ModelSpeedup
from nni.compression.pytorch.speedup.v2.external_replacer import TransformersAttentionReplacer


def pruning_attn():
    Path('./output/bert_finetuned/').mkdir(parents=True, exist_ok=True)
    model = build_finetuning_model(task_name, f'./output/bert_finetuned/{task_name}.bin')
    trainer = prepare_traced_trainer(model, task_name)
    evaluator = TransformersEvaluator(trainer)

    config_list = [{
        'op_types': ['Linear'],
        'op_names_re': ['bert\.encoder\.layer\.[0-9]*\.attention\.*'],
        'sparse_threshold': 0.1,
        'granularity': [64, 64]
    }]

    pruner = MovementPruner(model, config_list, evaluator, warmup_step=9000, cooldown_begin_step=36000, regular_scale=10)
    pruner.compress(None, 4)
    pruner.unwrap_model()

    masks = pruner.get_masks()
    Path('./output/pruning/').mkdir(parents=True, exist_ok=True)
    torch.save(masks, './output/pruning/attn_masks.pth')
    torch.save(model, './output/pruning/attn_masked_model.pth')


def speedup_attn():
    # model = build_finetuning_model(task_name, f'./output/bert_finetuned/{task_name}.bin')
    model = torch.load('./output/pruning/attn_masked_model.pth', map_location='cpu')
    masks = torch.load('./output/pruning/attn_masks.pth', map_location='cpu')
    dummy_input = (torch.randint(0, 10000, [8, 128]), torch.randint(0, 2, [8, 128]), torch.randint(0, 2, [8, 128]))
    replacer = TransformersAttentionReplacer(model)
    ModelSpeedup(model, dummy_input, masks, customized_replacers=[replacer]).speedup_model()

    # finetuning
    teacher_model = build_finetuning_model('mnli', f'./output/bert_finetuned/{task_name}.bin')
    dynamic_distillation(model, teacher_model, None, 3)
    torch.save(model, './output/pruning/attn_pruned_model.pth')


from nni.contrib.compression.pruning import TaylorPruner, AGPPruner
from transformers.models.bert.modeling_bert import BertAttention, BertLayer


def pruning_ffn():
    model: BertForSequenceClassification = torch.load('./output/pruning/attn_pruned_model.pth')
    teacher_model: BertForSequenceClassification = build_finetuning_model('mnli', f'./output/bert_finetuned/{task_name}.bin')
    # create ffn config list, here simply use a linear function related to the number of retained heads to determine the sparse ratio
    config_list = []
    for name, module in model.named_modules():
        if isinstance(module, BertLayer):
            retained_head_num = module.attention.self.num_attention_heads
            ori_head_num = len(module.attention.pruned_heads) + retained_head_num
            ffn_sparse_ratio = 1 - retained_head_num / ori_head_num / 2
            config_list.append({'op_names': [f'{name}.intermediate.dense'], 'sparse_ratio': ffn_sparse_ratio})

    trainer = prepare_traced_trainer(model, task_name)
    teacher_model.eval().to(trainer.args.device)
    distiller = dynamic_distiller(model, teacher_model, trainer)
    taylor_pruner = TaylorPruner.from_compressor(distiller, config_list, 1000)
    agp_pruner = AGPPruner(taylor_pruner, 1000, 36)
    agp_pruner.compress(None, 3)
    agp_pruner.unwrap_model()
    distiller.unwrap_teacher_model()

    masks = agp_pruner.get_masks()
    Path('./output/pruning/').mkdir(parents=True, exist_ok=True)
    torch.save(masks, './output/pruning/ffn_masks.pth')
    torch.save(model, './output/pruning/ffn_masked_model.pth')


def speedup_ffn():
    model = torch.load('./output/pruning/ffn_masked_model.pth', map_location='cpu')
    masks = torch.load('./output/pruning/ffn_masks.pth', map_location='cpu')
    dummy_input = (torch.randint(0, 10000, [8, 128]), torch.randint(0, 2, [8, 128]), torch.randint(0, 2, [8, 128]))
    ModelSpeedup(model, dummy_input, masks).speedup_model()

    # finetuning
    teacher_model = build_finetuning_model('mnli', f'./output/bert_finetuned/{task_name}.bin')
    dynamic_distillation(model, teacher_model, None, 3)
    torch.save(model, './output/pruning/ffn_pruned_model.pth')


from nni.contrib.compression.base.setting import PruningSetting

output_align_setting = {
    '_output_': {
        'align': {
            'module_name': None,
            'target_name': 'weight',
            'dims': [0],
        },
        'apply_method': 'mul',
    }
}
PruningSetting.register('BertAttention', output_align_setting)
PruningSetting.register('BertOutput', output_align_setting)


def pruning_embedding():
    model: BertForSequenceClassification = torch.load('./output/pruning/ffn_pruned_model.pth')
    teacher_model: BertForSequenceClassification = build_finetuning_model('mnli', f'./output/bert_finetuned/{task_name}.bin')

    ori_head_num = 0
    retained_head_num = 0
    for _, module in model.named_modules():
        if isinstance(module, BertLayer):
            retained_head_num += module.attention.self.num_attention_heads
            ori_head_num += len(module.attention.pruned_heads) + retained_head_num
    # sparse_ratio = 1 - retained_head_num / ori_head_num
    sparse_ratio = 0.5
    config_list = [{
        'op_types': ['Embedding'],
        'op_names_re': ['bert\.embeddings.*'],
        'sparse_ratio': sparse_ratio,
        'dependency_group_id': 1,
        'granularity': [-1, 1],
    }, {
        'op_names_re': ['bert\.encoder\.layer\.[0-9]*\.attention$',
                        'bert\.encoder\.layer\.[0-9]*\.output$'],
        'target_names': ['_output_'],
        'target_settings': {
            '_output_': {
                'align': {
                    'module_name': 'bert.embeddings.word_embeddings',
                    'target_name': 'weight',
                    'dims': [1],
                }
            }
        }
    }, {
        'op_names_re': ['bert\.encoder\.layer\.[0-9]*\.attention.output.dense',
                        'bert\.encoder\.layer\.[0-9]*\.output.dense'],
        'target_names': ['weight'],
        'target_settings': {
            'weight': {
                'granularity': 'out_channel',
                'align': {
                    'module_name': 'bert.embeddings.word_embeddings',
                    'target_name': 'weight',
                    'dims': [1],
                }
            }
        }
    }]

    trainer = prepare_traced_trainer(model, task_name)
    teacher_model.eval().to(trainer.args.device)
    distiller = dynamic_distiller(model, teacher_model, trainer)
    taylor_pruner = TaylorPruner.from_compressor(distiller, config_list, 1000)
    agp_pruner = AGPPruner(taylor_pruner, 1000, 36)
    agp_pruner.compress(None, 3)
    agp_pruner.unwrap_model()
    distiller.unwrap_teacher_model()

    masks = agp_pruner.get_masks()
    Path('./output/pruning/').mkdir(parents=True, exist_ok=True)
    torch.save(masks, './output/pruning/embedding_masks.pth')
    torch.save(model, './output/pruning/embedding_masked_model.pth')


def speedup_embedding():
    model = torch.load('./output/pruning/embedding_masked_model.pth', map_location='cpu')
    masks = torch.load('./output/pruning/embedding_masks.pth', map_location='cpu')
    dummy_input = (torch.randint(0, 10000, [8, 128]), torch.randint(0, 2, [8, 128]), torch.randint(0, 2, [8, 128]))
    ModelSpeedup(model, dummy_input, masks).speedup_model()

    # finetuning
    teacher_model = build_finetuning_model('mnli', f'./output/bert_finetuned/{task_name}.bin')
    adapt_distillation(model, teacher_model, None, 4)
    torch.save(model, './output/pruning/embedding_pruned_model.pth')


def evaluate_pruned_model():
    model: BertForSequenceClassification = torch.load('./output/pruning/embedding_pruned_model.pth')
    trainer = prepare_traced_trainer(model, task_name)
    metric = trainer.evaluate()
    pruned_num_params = sum(param.numel() for param in model.parameters()) + sum(buffer.numel() for buffer in model.buffers())

    model = build_finetuning_model(task_name, f'./output/bert_finetuned/{task_name}.bin')
    ori_num_params = sum(param.numel() for param in model.parameters()) + sum(buffer.numel() for buffer in model.buffers())
    print(f'Metric: {metric}\nSparsity: {1 - pruned_num_params / ori_num_params}')

Path('./output/bert_finetuned').mkdir(exist_ok=True, parents=True)
build_finetuning_model(task_name, f'./output/bert_finetuned/{task_name}.bin')

pruning_attn()
speedup_attn()
pruning_ffn()
speedup_ffn()
pruning_embedding()
speedup_embedding()
evaluate_pruned_model()
