import argparse
from bdb import set_trace
import glob
import json
import logging

import os
import random
import math

import numpy as np
import torch
# from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from nni.compression.pytorch.utils import get_module_by_name
from emmental import MaskedBertConfig, MaskedBertForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from emmental.modules.masked_nn import MaskedLinear
import nni
import torch
import sys
import os
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

mask_path = './mask'
weight_path = './weight'

task_name = "qqp"
model_name_or_path = '../training/result/qqp_partial/coarse_0.3/checkpoint-220000/'
data_dir = './glue_data/QQP'
max_seq_length= 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_and_cache_examples(task, tokenizer, evaluate=False):
    
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if task in ["mnli",
                    "mnli-mm"] and args.model_type in ["roberta",
                                                       "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(
                data_dir) if evaluate else processor.get_train_examples(
                data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels)
    return dataset

def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ['qqp']
    eval_outputs_dirs = './tmp'
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=32)

        # Eval!
        # print(f"***** Running evaluation {prefix} *****")
        # print(f"  Num examples = {len(eval_dataset)}")
        # print(f"  Batch size = {args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]}
                
                inputs["token_type_ids"] = batch[2] 
                     # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        
        from scipy.special import softmax

        probs = softmax(preds, axis=-1)
        entropy = np.exp((-probs * np.log(probs)).sum(axis=-1).mean())
        preds = np.argmax(preds, axis=1)
        # import pdb; pdb.set_trace()
        # result = compute_metrics(eval_task, preds, out_label_ids)
        # results.update(result)
        results[eval_task] = (preds == out_label_ids).mean()
        # if entropy is not None:
        #     result["eval_avg_entropy"] = entropy

        # output_eval_file = os.path.join(
        #     eval_output_dir, prefix, "eval_results.txt")

    return results

def apply_mask(model, mask):
    cfglist = []
    for layer in mask:
        # here the sparsity ratio doesn't matter, because we will
        # assign the mask manually
        cfglist.append({'sparsity':0.9, 'op_types':['Linear'], 'op_names':[layer]})
    tmp_pruner = LevelPruner(model, cfglist)
    tmp_pruner.compress()
    
    for name in mask:
        _, module = get_module_by_name(model, name)
        for key in mask[name]:
            # print(module)
            # import ipdb; ipdb.set_trace()
            assert(hasattr(module,key + '_mask'))
            setattr(module, key+'_mask', mask[name][key].cuda())

def train(train_dataset, model, tokenizer, teacher=None, num_train_epochs=10):
    """ Train the model """
    train_sampler = RandomSampler(
        train_dataset) 
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=32)

    t_total = len(train_dataloader) * num_train_epochs
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    weight_decay = 0.0
    warmup_steps=100
    temperature = 2.0
    alpha_distil = 0.9
    alpha_ce =0.1
    max_grad_norm = 1.0
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "lr": learning_rate,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon)

    ######## Quantize addition part
    configure_list = [{
        'quant_types': ['weight', 'output'],
        'quant_bits': {
            'weight': 8,
            'output': 8
        }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
        'op_types':['Linear']
    }]

    quantizer = QAT_Quantizer(model, configure_list, optimizer)
    quantizer.compress()
    ###############################
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total)

    # Train!
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")

    print(f"  Total optimization steps = {t_total}")
    # Distillation
    if teacher is not None:
        print("  Training with distillation")

    global_step = 0

    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(
    #     epochs_trained,
    #     int(num_train_epochs),
    #     desc="Epoch",

    # )
    # Added here for reproducibility

    for _ in range(num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]}
            
            inputs["token_type_ids"] = batch[2] # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(**inputs)
                # print(outputs)

                loss, logits_stu, reps_stu, attentions_stu = outputs[0], outputs[1], outputs[2], outputs[3]
                # import pdb; pdb.set_trace()
                # Distillation loss
                if teacher is not None:
                    if "token_type_ids" not in inputs:
                        inputs["token_type_ids"] = batch[2]
                    with torch.no_grad():
                        outputs_tea = teacher(
                            input_ids=inputs["input_ids"],
                            token_type_ids=inputs["token_type_ids"],
                            attention_mask=inputs["attention_mask"],
                        )
                        logits_tea = outputs_tea.logits
                        reps_tea, attentions_tea = outputs_tea.hidden_states, outputs_tea.attentions

                    teacher_layer_num = len(attentions_tea)
                    student_layer_num = len(attentions_stu)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(
                        teacher_layer_num / student_layer_num)
                    new_attentions_tea = [attentions_tea[i *
                                                         layers_per_block +
                                                         layers_per_block -
                                                         1] for i in range(student_layer_num)]

                    att_loss, rep_loss = 0, 0
                    for student_att, teacher_att in zip(
                            attentions_stu, new_attentions_tea):
                        student_att = torch.where(
                            student_att <= -1e2,
                            torch.zeros_like(student_att).to(device),
                            student_att)
                        teacher_att = torch.where(
                            teacher_att <= -1e2,
                            torch.zeros_like(teacher_att).to(device),
                            teacher_att)

                        tmp_loss = F.mse_loss(
                            student_att, teacher_att, reduction="mean",)
                        att_loss += tmp_loss

                    new_reps_tea = [reps_tea[i * layers_per_block]
                                    for i in range(student_layer_num + 1)]
                    new_reps_stu = reps_stu
                    for i_threp, (student_rep, teacher_rep) in enumerate(
                            zip(new_reps_stu, new_reps_tea)):
                        tmp_loss = F.mse_loss(
                            student_rep, teacher_rep, reduction="mean",)
                        rep_loss += tmp_loss

                    loss_logits = F.kl_div(
                        input=F.log_softmax(logits_stu / temperature, dim=-1),
                        target=F.softmax(logits_tea / temperature, dim=-1),
                        reduction="batchmean",
                    ) * (temperature ** 2)

                    loss_distill = loss_logits + rep_loss + att_loss
                    loss = alpha_distil * loss_distill + alpha_ce * loss




         
            loss.backward()

            tr_loss += loss.item()
            if True:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)

       
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

    
        results = evaluate(model, tokenizer)
        print("Evaluate Accuracy", results)    
        if(results['qqp']>0.8848):
            import ipdb; ipdb.set_trace()       
            break
    return global_step, tr_loss / global_step

# def train(train_dataset, model, tokenizer, teacher=None, num_train_epochs=10):
#     """ Train the model """
#     train_sampler = RandomSampler(
#         train_dataset) 
#     train_dataloader = DataLoader(
#         train_dataset,
#         sampler=train_sampler,
#         batch_size=32)

#     t_total = len(train_dataloader) * num_train_epochs
#     learning_rate = 5e-5
#     adam_epsilon = 1e-8
#     weight_decay = 0.0
#     warmup_steps=100
#     temperature = 2.0
#     alpha_distil = 0.9
#     alpha_ce =0.1
#     max_grad_norm = 1.0
#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]

#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p
#                 for n, p in model.named_parameters()
#                 if "mask_score" not in n and "threshold" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
#             ],
#             "lr": learning_rate,
#             "weight_decay": weight_decay,
#         },
#         {
#             "params": [
#                 p
#                 for n, p in model.named_parameters()
#                 if "mask_score" not in n and "threshold" not in n and p.requires_grad and any(nd in n for nd in no_decay)
#             ],
#             "lr": learning_rate,
#             "weight_decay": 0.0,
#         },
#     ]

#     optimizer = AdamW(
#         optimizer_grouped_parameters,
#         lr=learning_rate,
#         eps=adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=t_total)

#     # Train!
#     print("***** Running training *****")
#     print(f"  Num examples = {len(train_dataset)}")
#     print(f"  Num Epochs = {num_train_epochs}")

#     print(f"  Total optimization steps = {t_total}")
#     # Distillation
#     if teacher is not None:
#         print("  Training with distillation")

#     global_step = 0

#     epochs_trained = 0
#     steps_trained_in_current_epoch = 0
#     # Check if continuing training from a checkpoint

#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     # train_iterator = trange(
#     #     epochs_trained,
#     #     int(num_train_epochs),
#     #     desc="Epoch",

#     # )
#     # Added here for reproducibility

#     for _ in range(num_train_epochs):
#         epoch_iterator = tqdm(train_dataloader, desc="Iteration")
#         for step, batch in enumerate(epoch_iterator):

#             # Skip past any already trained steps if resuming training
#             if steps_trained_in_current_epoch > 0:
#                 steps_trained_in_current_epoch -= 1
#                 continue

#             model.train()
#             batch = tuple(t.to(device) for t in batch)

#             inputs = {
#                 "input_ids": batch[0],
#                 "attention_mask": batch[1],
#                 "labels": batch[3]}
            
#             inputs["token_type_ids"] = batch[2] # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

#             with torch.cuda.amp.autocast(enabled=False):
#                 outputs = model(**inputs)
#                 # print(outputs)

#                 loss, logits_stu, reps_stu, attentions_stu = outputs[0], outputs[1], outputs[2], outputs[3]
#                 # import pdb; pdb.set_trace()
#                 # Distillation loss
#                 if teacher is not None:
#                     if "token_type_ids" not in inputs:
#                         inputs["token_type_ids"] = batch[2]
#                     with torch.no_grad():
#                         outputs_tea = teacher(
#                             input_ids=inputs["input_ids"],
#                             token_type_ids=inputs["token_type_ids"],
#                             attention_mask=inputs["attention_mask"],
#                         )
#                         logits_tea = outputs_tea.logits
#                         reps_tea, attentions_tea = outputs_tea.hidden_states, outputs_tea.attentions

#                     teacher_layer_num = len(attentions_tea)
#                     student_layer_num = len(attentions_stu)
#                     assert teacher_layer_num % student_layer_num == 0
#                     layers_per_block = int(
#                         teacher_layer_num / student_layer_num)
#                     new_attentions_tea = [attentions_tea[i *
#                                                          layers_per_block +
#                                                          layers_per_block -
#                                                          1] for i in range(student_layer_num)]

#                     att_loss, rep_loss = 0, 0
#                     for student_att, teacher_att in zip(
#                             attentions_stu, new_attentions_tea):
#                         student_att = torch.where(
#                             student_att <= -1e2,
#                             torch.zeros_like(student_att).to(device),
#                             student_att)
#                         teacher_att = torch.where(
#                             teacher_att <= -1e2,
#                             torch.zeros_like(teacher_att).to(device),
#                             teacher_att)

#                         tmp_loss = F.mse_loss(
#                             student_att, teacher_att, reduction="mean",)
#                         att_loss += tmp_loss

#                     new_reps_tea = [reps_tea[i * layers_per_block]
#                                     for i in range(student_layer_num + 1)]
#                     new_reps_stu = reps_stu
#                     for i_threp, (student_rep, teacher_rep) in enumerate(
#                             zip(new_reps_stu, new_reps_tea)):
#                         tmp_loss = F.mse_loss(
#                             student_rep, teacher_rep, reduction="mean",)
#                         rep_loss += tmp_loss

#                     loss_logits = F.kl_div(
#                         input=F.log_softmax(logits_stu / temperature, dim=-1),
#                         target=F.softmax(logits_tea / temperature, dim=-1),
#                         reduction="batchmean",
#                     ) * (temperature ** 2)

#                     loss_distill = loss_logits + rep_loss + att_loss
#                     loss = alpha_distil * loss_distill + alpha_ce * loss




         
#             loss.backward()

#             tr_loss += loss.item()
#             if True:
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), max_grad_norm)

       
#                 optimizer.step()
#                 scheduler.step()  # Update learning rate schedule
#                 model.zero_grad()
#                 global_step += 1



#         results = evaluate(model, tokenizer)
#         print("Evaluate Accuracy", results)    
                  

#     return global_step, tr_loss / global_step



if __name__ == '__main__':
        
    output_mode = output_modes[task_name]
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    config = MaskedBertConfig.from_pretrained(model_name_or_path)
    model = BertForSequenceClassification(config=config)
    model.prune_heads(head_pruner_cfg)
    mask = torch.load(mask_path, map_location=device)
    weight_state_dict = torch.load(weight_path)
    # import pdb; pdb.set_trace()
    model.load_state_dict(weight_state_dict)

    model = model.to(device)
    cfg_list = [{'op_types':['Linear'], 'sparsity':0.1}]
    pruner = LevelPruner(model, cfg_list)
    pruner.compress()
    for name in mask:
        _, module = get_module_by_name(model, name)
        for key in mask[name]:
            setattr(module, key, mask[name][key])
    print(evaluate(model, tokenizer))
    train_dataset = load_and_cache_examples("qqp", tokenizer, evaluate=False)
    train(train_dataset, model, tokenizer)
    import pdb; pdb.set_trace()