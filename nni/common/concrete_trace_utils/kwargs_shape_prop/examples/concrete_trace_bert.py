# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from nni.common.concrete_trace_utils import concrete_trace
from kwargs_shape_prop import KwargsShapeProp


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

model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dummy_input = tokenizer("I like you. I love you", return_tensors="pt")
cfg = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_config(cfg)
model.eval()

with torch.no_grad():
    output_origin = model(**dummy_input)

traced_model = concrete_trace(
    model,
    dummy_input,
    use_operator_patch=True,
    autowrap_leaf_class={
        torch.finfo:                                ((), False),
        type(output_origin):  ((), False),
    },
)

with torch.no_grad():
    output_traced = traced_model(**dummy_input)

assert check_equal(output_origin, output_traced), 'check_equal failed.'
print("trace succeeded!")

KwargsShapeProp(traced_model).propagate(dummy_input)
for node in traced_model.graph.nodes:
    if 'tensor_meta' in node.meta:
        try:
            print(node.name, '|', node.meta['type'], '|', node.meta['tensor_meta'].dtype, '|',node.meta['tensor_meta'].shape)
        except:
            print('exception:')
            print(node.name, '|', node.op, '|', node.target, '|', node.meta['type'], '|', node.meta['tensor_meta'])