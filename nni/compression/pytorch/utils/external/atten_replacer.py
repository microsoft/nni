# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import logging
import re
from typing import Dict, List

import torch

from nni.compression.pytorch.speedup.replacer import Replacer
from nni.compression.pytorch.speedup.infer_mask import AutoMaskInference
from nni.compression.pytorch.utils.attr import get_nested_attr
from nni.compression.pytorch.utils.external.huggingface import parser_factory, HuggingfaceModelParser

_logger = logging.getLogger(__name__)


def _endwith(s: str, suffixes: List[str]):
    return any(s.endswith(suffix) for suffix in suffixes)


def _prune_head_idxs(mask: torch.Tensor, num_heads: int) -> List[int]:
    head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
    return torch.arange(len(head_mask))[head_mask].long().tolist()


def _remained_idxs(mask: torch.Tensor, num_heads: int) -> List[int]:
    repeats = mask.shape[0] // num_heads
    remained = (mask.reshape([num_heads, -1]).sum(-1) != 0.).repeat_interleave(repeats)
    return torch.arange(len(mask))[remained].long().tolist()


def _fill_one_on_dims(mask: torch.Tensor, dims: int | List[int]) -> torch.Tensor:
    dims = dims if isinstance(dims, list) else [dims]
    dims = [d if d >= 0 else d + len(mask.shape) for d in dims]
    new_mask = torch.ones_like(mask)
    for i in range(len(mask.shape)):
        if i in dims:
            continue
        dim_mask = (mask.sum([_ for _ in range(len(mask.shape)) if _ != i]) == 0.)
        new_mask = new_mask.transpose(0, i)
        new_mask[torch.arange(len(dim_mask), device=new_mask.device)[dim_mask].long().tolist()] = 0.
        new_mask = new_mask.transpose(0, i)
    return new_mask


class TransformersAttentionReplacer(Replacer):
    """
    This replacer is used to prune huggingface transformers attention heads,
    it base on ``HuggingfaceModelParser`` to find the attention module,
    and prune heads with attention module built-in ``prune_heads`` interface.

    Parameters
    ----------
    model
        The transformer model, now nni officially support bert, bart, t5, vit.
    parser
        The model parser used to find the attention module.
        If the model passed in is not bert, bart, t5 or vit,
        please inherit ``nni.compression.pytorch.utils.external.huggingface.HuggingfaceModelParser``
        to customize a new model parser and pass in.
    """
    def __init__(self, model: torch.nn.Module, parser: HuggingfaceModelParser | None = None):
        self.parser = parser_factory(model) if parser is None else parser
        if self.parser is None:
            err_msg = f'Can not get the model parser of {type(model)}'
            raise RuntimeError(err_msg)

    def replace_modules(self, model: torch.nn.Module, auto_inferences: Dict[str, AutoMaskInference]):
        # Note: This replace function base on prune_heads interface in Huggingface transformers.
        attention_name_dict = defaultdict(list)
        attention_patterns = [self.parser.TRANSFORMER_PREFIX + att_p for att_p in self.parser.ATTENTION]
        # find layers which has attention layer name prefix
        for unique_name, _ in auto_inferences.items():
            if self.parser.is_attention(unique_name):
                for attention_pattern in attention_patterns:
                    attention_layer_name = re.findall(attention_pattern, unique_name)[0]
                    attention_name_dict[attention_layer_name].append(unique_name)
        # prune heads
        for attention_layer_name, qkvo_names in attention_name_dict.items():
            # qkvo_flatten_head_mask is the sum of qkv output mask and o input mask
            qkvo_flatten_head_mask: torch.Tensor | None = None
            for name in qkvo_names:
                if _endwith(name, self.parser.QKVO):
                    info_msg = f'Find QKVO layer `{name}`, try to prune head.'
                    _logger.info(info_msg)
                    if _endwith(name, self.parser.QKV):
                        _, out_masks, _ = auto_inferences[name].get_masks()
                        flatten_head_mask = (torch.sum(out_masks, dim=[_ for _ in range(len(out_masks.shape) - 1)]).detach() > 0.).float()
                    else:
                        in_masks, _, _ = auto_inferences[name].get_masks()
                        flatten_head_mask = (torch.sum(in_masks[0],
                            dim=[_ for _ in range(len(in_masks[0].shape) - 1)]).detach() > 0.).float()
                    if qkvo_flatten_head_mask is not None:
                        qkvo_flatten_head_mask *= flatten_head_mask
                    else:
                        qkvo_flatten_head_mask = flatten_head_mask
            if qkvo_flatten_head_mask is not None:
                original_num_heads = self.parser.get_num_heads(attention_layer_name, model)
                head_idxs = _prune_head_idxs(qkvo_flatten_head_mask, original_num_heads)
                info_msg = f'Prune {attention_layer_name} head {head_idxs}'
                _logger.info(info_msg)
                attention_layer = get_nested_attr(model, attention_layer_name)
                attention_layer.prune_heads(head_idxs)  # type: ignore
                # replace autoinfer masks with ones, assume QKVO are all Linear
                remained_idxs = _remained_idxs(qkvo_flatten_head_mask, original_num_heads)
                for name in qkvo_names:
                    if _endwith(name, self.parser.QKVO):
                        if _endwith(name, self.parser.QKV):
                            mask = auto_inferences[name].weight_mask['weight'][remained_idxs]
                            auto_inferences[name].weight_mask['weight'] = _fill_one_on_dims(mask, 0)
                            mask = auto_inferences[name].output_mask.transpose(0, -1)[remained_idxs].transpose(0, -1)
                            auto_inferences[name].output_mask = _fill_one_on_dims(mask, -1)
                        else:
                            mask = auto_inferences[name].weight_mask['weight'][:, remained_idxs]
                            auto_inferences[name].weight_mask['weight'] = _fill_one_on_dims(mask, 1)
                            mask = auto_inferences[name].in_masks[0].transpose(0, -1)[remained_idxs].transpose(0, -1)
                            auto_inferences[name].in_masks[0] = _fill_one_on_dims(mask, -1)
