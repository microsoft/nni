# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.pytorch.pruning import TransformerHeadPruner

import datasets
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)


logger = logging.getLogger("bert_pruning_example")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example: prune a Huggingface transformer and finetune on GLUE tasks.")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Pretrained model architecture.")
    parser.add_argument("--task_name", type=str, default=None,
                        help="The name of the GLUE task.",
                        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the model and mask.")
    parser.add_argument("--sparsity", type=float, required=True,
                        help="Sparsity: proportion of heads to prune (should be between 0 and 1)")
    parser.add_argument("--global_sort", action="store_true", default=False,
                        help="Rank the heads globally and prune the heads with lowest scores. If set to False, the "
                             "heads are only ranked within one layer")
    parser.add_argument("--ranking_criterion", type=str, default="l1_weight",
                        choices=["l1_weight", "l2_weight",
                                 "l1_activation", "l2_activation", "taylorfo"],
                        help="Criterion by which the attention heads are ranked.")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of pruning iterations (1 for one-shot pruning).")
    parser.add_argument("--epochs_per_iteration", type=int, default=1,
                        help="Epochs to finetune before the next pruning iteration "
                             "(only effective if num_iterations > 1).")
    parser.add_argument("--speedup", action="store_true", default=False,
                        help="Whether to speedup the pruned model")

    # parameters for model training; no need to change them for running examples
    parser.add_argument("--max_length", type=int, default=128,
                        help=("The maximum total input sequence length after tokenization. Sequences longer than this "
                              "will be truncated, sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--lr_scheduler_type", default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_raw_dataset(task_name):
    """
    Get a GLUE dataset using huggingface datasets.
    """
    raw_dataset = load_dataset("glue", task_name)
    is_regression = task_name == "stsb"
    num_labels = 1 if is_regression else len(
        raw_dataset["train"].features["label"].names)

    return raw_dataset, is_regression, num_labels


def preprocess(args, tokenizer, raw_dataset):
    """
    Tokenization and column renaming. 
    """
    assert args.task_name is not None

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def tokenize(data):
        texts = (
            (data[sentence1_key],) if sentence2_key is None else (
                data[sentence1_key], data[sentence2_key])
        )
        result = tokenizer(*texts, padding=False,
                           max_length=args.max_length, truncation=True)

        if "label" in data:
            result["labels"] = data["label"]
        return result

    processed_datasets = raw_dataset.map(
        tokenize, batched=True, remove_columns=raw_dataset["train"].column_names)
    return processed_datasets


def get_dataloader_and_optimizer(args, tokenizer, model, train_dataset, eval_dataset):
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                 batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    return optimizer, train_dataloader, eval_dataloader, data_collator


def train_model(args, model, is_regression, train_dataloader, eval_dataloader, optimizer, lr_scheduler, metric, device):
    """
    Train the model using train_dataloader and evaluate after every epoch using eval_dataloader.
    This function is called before and after pruning for "pretraining" on the GLUE task and further "finetuning". 
    """
    train_steps = args.num_train_epochs * len(train_dataloader)
    progress_bar = tqdm(range(train_steps), position=0, leave=True)

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            for field in batch.keys():
                batch[field] = batch[field].to(device)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            for field in batch.keys():
                batch[field] = batch[field].to(device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression \
                else outputs.logits.squeeze()
            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")


def trainer_helper(model, train_dataloader, optimizer, device):
    """
    This function is used for to create a "trainer" that is passed to the pruner. 
    Finetune the model for 1 epoch. This function is called by the pruner during pruning iterations (or called to
    calculate scores for pruning when ranking criterion is "taylorfo").
    """
    logger.info("Training for 1 epoch...")
    progress_bar = tqdm(range(len(train_dataloader)), position=0, leave=True)

    train_epoch = 1
    for epoch in range(train_epoch):
        for step, batch in enumerate(train_dataloader):
            for field in batch.keys():
                batch[field] = batch[field].to(device)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def forward_runner_helper(model, train_dataloader, device):
    """
    This function is used for to create a "forward_runner" that is passed to the pruner.
    The function just runs forward on the train set without updating the parameters.
    This allows the pruner to collect data for activation-based pruning methods.
    """
    logger.info("Running forward on the entire train set without updating parameters...")
    progress_bar = tqdm(range(len(train_dataloader)), position=0, leave=True)

    forward_epoch = 1
    for epoch in range(forward_epoch):
        for step, batch in enumerate(train_dataloader):
            for field in batch.keys():
                batch[field] = batch[field].to(device)
            _ = model(**batch)
            # note: no loss.backward or optimizer.step() is performed here
            progress_bar.update(1)


def final_eval_for_mnli(args, model, processed_datasets, metric, data_collator):
    """
    If the task is MNLI, perform a final evaluation on mismatched validation set
    """
    eval_dataset = processed_datasets["validation_mismatched"]
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.batch_size
    )

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )

    eval_metric = metric.compute()
    logger.info(f"mnli-mm: {eval_metric}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    #########################################################################
    # Prepare model, tokenizer, dataset, optimizer, and the scheduler
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # Load dataset and tokenizer, and then preprocess the dataset
    raw_dataset, is_regression, num_labels = get_raw_dataset(args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    processed_datasets = preprocess(args, tokenizer, raw_dataset)
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name ==
                                      "mnli" else "validation"]

    # Load pretrained model
    config = AutoConfig.from_pretrained(
        args.model_name, num_labels=num_labels, finetuning_task=args.task_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config)
    model.to(device)

    #########################################################################
    # Finetune on the target GLUE task before pruning
    optimizer, train_dataloader, eval_dataloader, data_collator = get_dataloader_and_optimizer(args, tokenizer,
                                                                                               model,
                                                                                               train_dataset,
                                                                                               eval_dataset)
    train_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=train_steps)
    metric = load_metric("glue", args.task_name)

    logger.info("================= Finetuning before pruning =================")
    train_model(args, model, is_regression, train_dataloader,
                eval_dataloader, optimizer, lr_scheduler, metric, device)

    if args.output_dir is not None:
        torch.save(model.state_dict(), args.output_dir + "/model_before_pruning.pt")

    if args.task_name == "mnli":
        final_eval_for_mnli(args, model, processed_datasets, metric, data_collator)

    #########################################################################
    # Pruning
    optimizer, train_dataloader, eval_dataloader, data_collator = get_dataloader_and_optimizer(args, tokenizer,
                                                                                               model,
                                                                                               train_dataset,
                                                                                               eval_dataset)
    dummy_input = next(iter(train_dataloader))["input_ids"].to(device)
    flops, params, results = count_flops_params(model, dummy_input)
    print(f"Initial model FLOPs {flops / 1e6:.2f} M, #Params: {params / 1e6:.2f}M")

    # Here criterion is embedded in the model. Upper levels can just pass None to trainer.
    def trainer(model, optimizer, criterion, epoch):
        return trainer_helper(model, train_dataloader, optimizer, device)

    def forward_runner(model):
        return forward_runner_helper(model, train_dataloader, device)

    # example: prune different layers with different sparsity
    attention_name_groups = list(zip(["bert.encoder.layer.{}.attention.self.query".format(i) for i in range(12)],
                                     ["bert.encoder.layer.{}.attention.self.key".format(i) for i in range(12)],
                                     ["bert.encoder.layer.{}.attention.self.value".format(i) for i in range(12)],
                                     ["bert.encoder.layer.{}.attention.output.dense".format(i) for i in range(12)]))

    kwargs = {"ranking_criterion": args.ranking_criterion,
              "global_sort": args.global_sort,
              "num_iterations": args.num_iterations,
              "epochs_per_iteration": args.epochs_per_iteration,
              "attention_name_groups": attention_name_groups,
              "head_hidden_dim": 64,
              "trainer": trainer,
              "optimizer": optimizer,
              "forward_runner": forward_runner}

    config_list = [{
        "sparsity": args.sparsity,
        "op_types": ["Linear"],
        "op_names": [x for layer in attention_name_groups[:6] for x in layer]
    }, {
        "sparsity": args.sparsity / 2,
        "op_types": ["Linear"],
        "op_names": [x for layer in attention_name_groups[6:] for x in layer]
    }]

    pruner = TransformerHeadPruner(model, config_list, **kwargs)
    pruner.compress()

    #########################################################################
    # uncomment the following part to export the pruned model masks
    # model_path = os.path.join(args.output_dir, "pruned_{}_{}.pth".format(args.model_name, args.task_name))
    # mask_path = os.path.join(args.output_dir, "mask_{}_{}.pth".format(args.model_name, args.task_name))
    # pruner.export_model(model_path=model_path, mask_path=mask_path)

    #########################################################################
    # Speedup
    # Currently, speeding up Transformers through NNI ModelSpeedup is not supported because of shape inference issues.
    # However, if you are using the transformers library, you can use the following workaround:
    # The following code gets the head pruning decisions from the pruner and calls the _prune_heads() function
    # implemented in models from the transformers library to speedup the model.
    if args.speedup:
        speedup_rules = {}
        for group_idx, group in enumerate(pruner.attention_name_groups):
            # get the layer index
            layer_idx = None
            for part in group[0].split("."):
                try:
                    layer_idx = int(part)
                    break
                except:
                    continue
            if layer_idx is not None:
                speedup_rules[layer_idx] = pruner.pruned_heads[group_idx]
        pruner._unwrap_model()
        model.bert._prune_heads(speedup_rules)
        print(model)

    #########################################################################
    # After pruning, finetune again on the target task
    # Get the metric function
    metric = load_metric("glue", args.task_name)

    # re-initialize the optimizer and the scheduler
    optimizer, _, _, data_collator = get_dataloader_and_optimizer(args, tokenizer, model, train_dataset,
                                                                  eval_dataset)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=train_steps)

    logger.info("================= Finetuning after Pruning =================")
    train_model(args, model, is_regression, train_dataloader,
                eval_dataloader, optimizer, lr_scheduler, metric, device)

    if args.output_dir is not None:
        torch.save(model.state_dict(), args.output_dir +
                   "/model_after_pruning.pt")

    if args.task_name == "mnli":
        final_eval_for_mnli(args, model, processed_datasets,
                            metric, data_collator)

    flops, params, results = count_flops_params(model, dummy_input)
    print(f"Final model FLOPs {flops / 1e6:.2f} M, #Params: {params / 1e6:.2f}M")


if __name__ == "__main__":
    main()
