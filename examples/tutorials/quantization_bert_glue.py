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


task_name = 'rte' 
finetune_lr = 4e-5
quant_lr = 1e-5
quant_method = 'ptq'
dev_mode = False

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
from nni.contrib.compression.quantization import QATQuantizer, LsqQuantizer, PtqQuantizer
from nni.contrib.compression.utils import TransformersEvaluator

# dummy_input is used for torch2onnx and onnx2trt

# transfer dummy_input type into dict
def transfer_dummy_input(dummy_input,input_names):
    dict_dummy_input = {}
    if isinstance(dummy_input,dict):
        for input_name,input_tensor in dummy_input.items():
            if torch.is_tensor(input_tensor):
                continue
            else:
                dummy_input[input_name] = torch.tensor(input_tensor)
        dict_dummy_input = dummy_input
    elif isinstance(dummy_input,tuple):
        for i in range(len(dummy_input)):
            if torch.is_tensor(dummy_input[i]):
                continue
            else:
                temp_dummy_input = torch.tensor(dummy_input[i])
                dict_dummy_input[input_names[i]] = temp_dummy_input
    elif torch.is_tensor(dummy_input):
        dict_dummy_input[input_names[0]] = dummy_input
    else :
        print('the dummy_input type is not allowed !')
    return dict_dummy_input

dummy_input = ([[101, 11271, 20726, 1010, 1996, 7794, 1997, 1996, 3364, 5696, 20726, 1010, 2038, 2351, 1997, 11192, 4456, 2012, 2287, 4008, 1010, 2429, 2000, 1996, 5696, 20726, 3192, 1012, 102, 5696, 20726, 2018, 2019, 4926, 1012, 102]],[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]],[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
input_names=['input_ids','token_type_ids','attention_mask']
dummy_input = transfer_dummy_input(dummy_input,input_names)

def fake_quantize():
    config_list = [{
        'op_types': ['Linear'],
        'op_names_re': ['bert.encoder.layer.{}'.format(i) for i in range(12)],
        'target_names': ['weight', '_input_','_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'symmetric',#'affine''symmetric'
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

    model.eval()
    model.to('cpu')
    print('quantized torch-model output: ', model(**dummy_input))
    model.to('cuda')
    quantizer.unwrap_model()
    evaluate()

    # Speed up the model with TensorRT
    from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
    engine = ModelSpeedupTensorRT(model, dummy_input=dummy_input, config=calibration_config, onnx_path='bert_rte.onnx',input_names=['input_ids','token_type_ids','attention_mask'],output_names=['output'],
    dynamic_axes={'input_ids' : {1 : 'seq_len'},
                'token_type_ids' : {1 : 'seq_len'},
                'attention_mask' : {1 : 'seq_len'}},
    dynamic_shape_setting ={'min_shape' : (1,18),
                            'opt_shape' : (1,72),
                            'max_shape' : (1,360)})
    engine.compress()
    import time
    start_time = time.time()
    output, time_span = engine.inference(dummy_input)
    infer_time = time.time() - start_time
    print('test dummy_input inference output: ', output)
    print('test dummy_input inference time: ', time_span, infer_time)
    test_Accuracy(engine)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_Accuracy(engine):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    _, validation_datasets = prepare_datasets(task_name, tokenizer, '')
    merged_validation_dataset = ConcatDataset([d for d in validation_datasets.values()]) # type: ignore
    true_cnt = 0
    total_time = 0
    for input_data in merged_validation_dataset:
        for input_name,input_tensor in input_data.items():
            if 'labels' != input_name:
                input_data[input_name] = torch.tensor([input_tensor])
        test_data = {key: input_data[key] for key in list(input_data.keys())[:-1]}
        output, time_span = engine.inference(test_data,reset_context=True)
        total_time += time_span
        prediction = torch.argmax(output,-1)
        if input_data['labels'] == prediction:
            true_cnt +=1
    Accuracy = true_cnt/len(merged_validation_dataset)
    print('inference time: ', total_time /len(merged_validation_dataset))
    print('Accuracy of mode #1: ', Accuracy)

def test_onnx_Accuracy(onnx_model):
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_model)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    _, validation_datasets = prepare_datasets(task_name, tokenizer, '')
    merged_validation_dataset = ConcatDataset([d for d in validation_datasets.values()]) # type: ignore
    true_cnt = 0
    for input_data in merged_validation_dataset:
        for input_name,input_tensor in input_data.items():
            if 'labels' != input_name:
                input_data[input_name] = to_numpy(torch.tensor([input_tensor]))
        test_data = {key: input_data[key] for key in list(input_data.keys())[:-1]}
        output = ort_session.run(None, test_data)
        prediction = np.argmax(output,-1)
        if input_data['labels'] == prediction:
            true_cnt +=1
    Accuracy = true_cnt/len(merged_validation_dataset)
    print('Accuracy of mode #1: ', Accuracy)



def evaluate():
    model = build_finetuning_model(f'./output/bert_finetuned/{task_name}.bin', is_quant=False)
    trainer = prepare_traced_trainer(model, is_quant=False)
    metrics = trainer.evaluate()
    print(f"Evaluate metrics={metrics}")


fake_quantize()
test_onnx_Accuracy('bert_rte.onnx')
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
