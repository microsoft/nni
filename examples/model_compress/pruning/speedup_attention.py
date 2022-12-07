# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This is an example for pruning speedup the huggingface transformers.
Now nni officially support speedup bert, bart, t5, vit attention heads.
For other transforms attention or even any hyper-module, users could customize by implementation a Replacer.
"""

import torch

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils.external.atten_replacer import TransformersAttentionReplacer

config = BertConfig()
model = BertForSequenceClassification(config)

config_list = [{
    'op_types': ['Linear'],
    'op_partial_names': ['bert.encoder.layer.{}.attention.self'.format(i) for i in range(12)],
    'sparsity': 0.98
}]

pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()
pruner._unwrap_model()

replacer = TransformersAttentionReplacer(model)
ModelSpeedup(model, torch.randint(0, 30000, [4, 128]), masks, customized_replacers=[replacer]).speedup_model()

print(model(**{'input_ids': torch.randint(0, 30000, [4, 128])}))
print(model)
