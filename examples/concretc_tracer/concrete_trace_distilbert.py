# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import modeling_outputs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nni.common.concrete_trace_utils import concrete_trace, ConcreteTracer

def check_equal(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    else:
        return a == b

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dummy_input = tokenizer("I like you. I love you", return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

traced_model = concrete_trace(
    model,
    dummy_input,
    autowrap_leaf_class={
        torch.finfo:                                ((), False),
        modeling_outputs.SequenceClassifierOutput:  ((), False),
        **ConcreteTracer.default_autowrap_leaf_class,
    },
)

with torch.no_grad():
    new_dummy_input = tokenizer("I dislike you. I hate you", return_tensors="pt")
    output_origin = model(**new_dummy_input)
    output_traced = traced_model(**new_dummy_input)

    assert check_equal(output_origin, output_traced), 'check_equal failed.'

print("traced code:", traced_model.code)
print("trace succeeded!")
