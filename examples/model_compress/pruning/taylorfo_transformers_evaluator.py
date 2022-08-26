import numpy as np

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

import nni
from nni.compression.pytorch import TransformersEvaluator
from nni.compression.pytorch.pruning import TaylorFOWeightPruner


dataset = load_dataset('yelp_review_full')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)

training_args = TrainingArguments(output_dir='test_trainer')

metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir='./log',
    evaluation_strategy='epoch',
    per_device_train_batch_size=32,
    num_train_epochs=3,
    max_steps=-1
)

trainer = nni.trace(Trainer)(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

evaluator = TransformersEvaluator(trainer)
pruner = TaylorFOWeightPruner(model, [{'op_types': ['Linear'], 'sparsity': 0.5}], evaluator, 20)
_, masks = pruner.compress()
pruner.show_pruned_weights()
