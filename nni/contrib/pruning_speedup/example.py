import torch

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.contrib.pruning_speedup.replacor import TransformersAttentionReplacor

config = BertConfig()
model = BertForSequenceClassification(config)
print(model(**{'input_ids': torch.randint(0, 30000, [4, 128])}))

config_list = [{
    'op_types': ['Linear'],
    'op_partial_names': ['bert.encoder.layer.{}.attention.self'.format(i) for i in range(12)],
    'sparsity': 0.98
}]

pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()
pruner._unwrap_model()

replacor = TransformersAttentionReplacor(model)
ModelSpeedup(model, torch.randint(0, 30000, [4, 128]), masks, customized_replacors=[replacor]).speedup_model()

print(model(**{'input_ids': torch.randint(0, 30000, [4, 128])}))
print(model)
