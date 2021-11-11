import functools
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import load_metric, load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    set_seed
)

from nni.algorithms.compression.v2.pytorch.pruning import MovementPruner


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def criterion(input, target):
    return input.loss

def trainer(model, optimizer, criterion, train_dataloader):
    model.train()
    for batch in tqdm(train_dataloader):
        batch.to(device)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(outputs, None)
        loss.backward()
        optimizer.step()

def evaluator(model, metric, is_regression, eval_dataloader):
    model.eval()
    for batch in tqdm(eval_dataloader):
        batch.to(device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    return metric.compute()

if __name__ == '__main__':
    task_name = 'sst2'
    num_labels = 2
    is_regression = False
    algo = 'l1_head'
    train_batch_size = 32
    eval_batch_size = 32

    set_seed(1024)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # used to preprocess the raw data
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    raw_datasets = load_dataset('glue', task_name, cache_dir='./data')
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

    train_dataset = processed_datasets['train']
    validate_dataset = processed_datasets['validation_matched' if task_name == "mnli" else 'validation']

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
    validate_dataloader = DataLoader(validate_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

    metric = load_metric("glue", task_name)

    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels).to(device)

    print(evaluator(model, metric, is_regression, validate_dataloader))

    op_names = []
    op_names.extend(["bert.encoder.layer.{}.attention.self.query".format(i) for i in range(0, 12)])
    op_names.extend(["bert.encoder.layer.{}.attention.self.key".format(i) for i in range(0, 12)])
    op_names.extend(["bert.encoder.layer.{}.attention.self.value".format(i) for i in range(0, 12)])

    config_list = [{'op_types': ['Linear'], 'op_names': op_names, 'sparsity': 0.8}]
    p_trainer = functools.partial(trainer, train_dataloader=train_dataloader)
    optimizer = Adam(model.parameters(), lr=2e-5)
    pruner = MovementPruner(model, config_list, p_trainer, optimizer, criterion, 3, 500)

    _, masks = pruner.compress()
    pruner.show_pruned_weights()

    optimizer = Adam(model.parameters(), lr=2e-5)
    trainer(model, optimizer, criterion, train_dataloader)
    print(evaluator(model, metric, is_regression, validate_dataloader))
