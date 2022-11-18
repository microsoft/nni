from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, List

import torch

from nni.compression.pytorch.speedup.infer_mask import AutoMaskInference
from nni.algorithms.compression.v2.pytorch.utils.attr import get_nested_attr

try:
    from nni.algorithms.compression.v2.pytorch.utils.external.huggingface import parser_factory
except ImportError:
    pass


def _endwith(s: str, suffixes: List[str]):
    return all(s.endswith(suffix) for suffix in suffixes)


class CustomizedReplacer:
    def replace_modules(self, model, auto_inferences: Dict[str, AutoMaskInference]):
        pass


class TransformersAttentionReplacer(CustomizedReplacer):
    def __init__(self, model) -> None:
        self.parser = parser_factory(model)
        if self.parser is None:
            err_msg = f'Can not get the model parser of {type(model)}'
            raise RuntimeError(err_msg)

    def replace_modules(self, model, auto_inferences: Dict[str, AutoMaskInference]):
        # Note: This replace function base on prune_heads interface in Huggingface transformers.
        attention_name_dict = defaultdict(list)
        attention_patterns = [self.parser.TRANSFORMER_PREFIX + att_p for att_p in self.parser.ATTENTION]
        # find layers who has attention layer name prefix
        for unique_name, _ in auto_inferences.items():
            if self.parser.is_attention(unique_name):
                for attention_pattern in attention_patterns:
                    attention_layer_name = re.findall(attention_pattern, unique_name)[0]
                    attention_name_dict[attention_layer_name].append(unique_name)
        # prune heads
        for attention_layer_name, qkvo_names in attention_name_dict.items():
            attention_layer = get_nested_attr(model, attention_layer_name)
            qkvo_flatten_head_mask: torch.Tensor | None = None
            for name in qkvo_names:
                if _endwith(name, self.parser.QKVO):
                    if _endwith(name, self.parser.QKV):
                        _, out_masks, _ = auto_inferences[name].get_masks()
                        flatten_head_mask = torch.sum(out_masks, dim=[_ for _ in range(len(out_masks.shape) - 1)])
                        out_masks.fill_(1)
                    else:
                        in_masks, _, _ = auto_inferences[name].get_masks()
                        flatten_head_mask = torch.sum(in_masks[0], dim=[_ for _ in range(len(in_masks[0].shape) - 1)])
                        in_masks[0].fill_(1)
                    if qkvo_flatten_head_mask:
                        qkvo_flatten_head_mask += flatten_head_mask
                    else:
                        qkvo_flatten_head_mask = flatten_head_mask
            if qkvo_flatten_head_mask:
                head_mask = qkvo_flatten_head_mask.reshape([self.parser.get_num_heads(attention_layer_name, model), -1]).sum(-1).bool()
                head_idxs = torch.arange(len(head_mask))[head_mask].long().tolist()
                print(f'prune {attention_layer_name} head {head_idxs}')
                attention_layer.prune_heads(head_idxs)  # type: ignore
