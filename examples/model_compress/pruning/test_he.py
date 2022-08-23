from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    max_steps=-1
)

#################################################################################################
import nni
from nni.algorithms.compression.v2.pytorch import HuggingfaceEvaluator
from nni.algorithms.compression.v2.pytorch.pruning import TaylorFOWeightPruner

trainer = nni.trace(Trainer)(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

evaluator = HuggingfaceEvaluator(trainer)
pruner = TaylorFOWeightPruner(model, [{'op_types': ['Linear'], 'sparsity': 0.5}], evaluator, 20)
_, masks = pruner.compress()
pruner.show_pruned_weights()
