# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import logging
import re
from typing import TYPE_CHECKING, List

import torch
from torch.utils._pytree import tree_map

from .huggingface import parser_factory, HuggingfaceModelParser
from ..attr import get_nested_attr
from ...speedup.replacer import Replacer

if TYPE_CHECKING:
    from ...speedup.model_speedup import ModelSpeedup

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
        please inherit ``nni.compression.utils.external.huggingface.HuggingfaceModelParser``
        to customize a new model parser and pass in.
    """
    def __init__(self, model: torch.nn.Module, parser: HuggingfaceModelParser | None = None):
        self.parser = parser_factory(model) if parser is None else parser
        if self.parser is None:
            err_msg = f'Can not get the model parser of {type(model)}'
            raise RuntimeError(err_msg)

    def replace_modules(self, speedup: 'ModelSpeedup'):
        # Note: This replace function base on prune_heads interface in Huggingface transformers.
        attention_name_dict = defaultdict(list)
        attention_patterns = [self.parser.TRANSFORMER_PREFIX + att_p for att_p in self.parser.ATTENTION]
        # find layers which has attention layer name prefix
        target2node = {}
        for node, node_info in speedup.node_infos.items():
            if node.op == 'call_module' and self.parser.is_attention(node.target):
                target2node[node.target] = node
                for attention_pattern in attention_patterns:
                    attention_layer_name = re.findall(attention_pattern, node.target)[0]
                    attention_name_dict[attention_layer_name].append(node.target)
        # prune heads
        for attention_layer_name, qkvo_names in attention_name_dict.items():
            # qkvo_flatten_head_mask is the sum of qkv output mask and o input mask
            qkvo_flatten_head_mask: torch.Tensor | None = None
            for name in qkvo_names:
                if _endwith(name, self.parser.QKVO):
                    info_msg = f'Find QKVO layer `{name}`, try to prune head.'
                    _logger.info(info_msg)
                    node = target2node[name]
                    node_info = speedup.node_infos[node]
                    if _endwith(name, self.parser.QKV):
                        out_masks = node_info.output_masks
                        flatten_head_mask = \
                            (torch.sum(out_masks, dim=[_ for _ in range(len(out_masks.shape) - 1)]).detach() > 0.).float()
                    else:
                        in_masks = tree_map(lambda n: speedup.node_infos[n].output_masks, node.args)
                        flatten_head_mask = \
                            (torch.sum(in_masks[0], dim=[_ for _ in range(len(in_masks[0].shape) - 1)]).detach() > 0.).float()
                    if qkvo_flatten_head_mask is not None:
                        qkvo_flatten_head_mask *= flatten_head_mask
                    else:
                        qkvo_flatten_head_mask = flatten_head_mask
            if qkvo_flatten_head_mask is not None:
                original_num_heads = self.parser.get_num_heads(attention_layer_name, speedup.bound_model)
                head_idxs = _prune_head_idxs(qkvo_flatten_head_mask, original_num_heads)
                info_msg = f'Prune {attention_layer_name} head {head_idxs}'
                _logger.info(info_msg)
                attention_layer = get_nested_attr(speedup.bound_model, attention_layer_name)
                attention_layer.prune_heads(head_idxs)  # type: ignore
                # replace autoinfer masks with ones, assume QKVO are all Linear
                remained_idxs = _remained_idxs(qkvo_flatten_head_mask, original_num_heads)
                for name in qkvo_names:
                    if _endwith(name, self.parser.QKVO):
                        node = target2node[name]
                        node_info = speedup.node_infos[node]
                        if _endwith(name, self.parser.QKV):
                            mask = node_info.param_masks['weight'][remained_idxs]
                            node_info.param_masks['weight'] = _fill_one_on_dims(mask, 0)
                            mask = node_info.output_masks.transpose(0, -1)[remained_idxs].transpose(0, -1)
                            node_info.output_masks = _fill_one_on_dims(mask, -1)
                        else:
                            mask = node_info.param_masks['weight'][:, remained_idxs]
                            node_info.param_masks['weight'] = _fill_one_on_dims(mask, 1)
                            masks = tree_map(lambda n: speedup.node_infos[n].output_masks, node.args)
                            mask = masks[0].transpose(0, -1)[remained_idxs].transpose(0, -1)
                            for n in node.args:
                                speedup.node_infos[n].output_masks = _fill_one_on_dims(mask, -1)
