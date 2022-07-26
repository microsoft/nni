# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import re
from typing import Set

from torch.nn import Module

try:
    from transformers import (
        PreTrainedModel,
        BartConfig,
        BertConfig,
        T5Config
    )
except ImportError:
    TRANSFORMERS_INSTALLED = False
else:
    TRANSFORMERS_INSTALLED = True

from nni.algorithms.compression.v2.pytorch.utils.attr import get_nested_attr

_logger = logging.getLogger(__name__)


def parser_factory(model: Module) -> HuggingfaceModelParser | None:
    if TRANSFORMERS_INSTALLED and isinstance(model, PreTrainedModel):
        cls2parser = {
            BartConfig: HuggingfaceBartParser,
            BertConfig: HuggingfaceBertParser,
            T5Config: HuggingfaceT5Parser
        }
        type2parser = {
            'bart': HuggingfaceBartParser,
            'bert': HuggingfaceBertParser,
            't5': HuggingfaceT5Parser
        }

        if hasattr(config, 'config_class'):
            parser = cls2parser.get(getattr(model.config, 'config_class'))
        elif hasattr(config, 'model_type'):
            parser = type2parser.get(getattr(model.config, 'model_type'))
        else:
            parser = None

        return parser
    else:
        return None


class HuggingfaceModelParser:
    TRANSFORMER_PREFIX: str
    QKV: Set[str]
    QKVO: Set[str]
    FFN1: Set[str]
    FFN2: Set[str]
    ATTENTION: Set[str]

    @classmethod
    def is_huggingface_model(cls, model: Module):
        return model.__module__.split('.')[0] == 'transformers'

    @classmethod
    def is_attention(cls, module_name: str, include_output: bool = True) -> bool:
        patterns = cls.QKVO if include_output else cls.QKV
        for pattern in patterns:
            if pattern in module_name:
                return True
        return False

    @classmethod
    def is_ffn(cls, module_name: str, ffn_num: int = 1) -> bool:
        if cls.is_attention(module_name):
            return False
        if ffn_num == 1:
            for pattern in cls.FFN1:
                if pattern in module_name:
                    return True
        if ffn_num == 2:
            for pattern in cls.FFN2:
                if pattern in module_name:
                    return True
        return False

    @classmethod
    def get_num_heads(cls, module_name: str, model: Module) -> int:
        if cls.is_attention(module_name, include_output=True):
            for pattern in cls.ATTENTION:
                match = re.search(pattern, module_name)
                if match:
                    attention_module_name = module_name[0: match.span()[1]]
                    module = get_nested_attr(model, attention_module_name)
                    if hasattr(module, 'num_attention_heads'):
                        num_heads = module.num_attention_heads
                    elif hasattr(module, 'num_heads'):
                        num_heads = module.num_heads
                    elif hasattr(module, 'n_heads'):
                        num_heads = module.n_heads
                    else:
                        warn_msg = f'Can not get the heads number of attention layer : {attention_module_name}.'
                        _logger.warning(warn_msg)
                        num_heads = 0
                    return num_heads
        else:
            warn_msg = f'The layer `{module_name}` might not an (Q|K|V) attention layer.'
            _logger.warning(warn_msg)
            return 0


class HuggingfaceBertParser(HuggingfaceModelParser):
    TRANSFORMER_PREFIX = r'bert\.encoder\.layer\.[0-9]+\.'
    QKV = ('attention.self.query', 'attention.self.key', 'attention.self.value')
    QKVO = QKV + ('attention.output.dense',)
    FFN1 = ('intermediate.dense',)
    FFN2 = ('output.dense',)
    ATTENTION = ('attention.self',)


class HuggingfaceBartParser(HuggingfaceModelParser):
    TRANSFORMER_PREFIX = r'(en|de)coder\.layer\.[0-9]+\.'
    QKV = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'encoder_attn.q_proj', 'encoder_attn.k_proj', 'encoder_attn.v_proj')
    QKVO = QKV + ('self_attn.out_proj', 'encoder_attn.out_proj')
    FFN1 = ('fc1',)
    FFN2 = ('fc2',)
    ATTENTION = ('self_attn', 'encoder_attn')


class HuggingfaceT5Parser(HuggingfaceModelParser):
    TRANSFORMER_PREFIX = r'(en|de)coder\.block\.[0-9]+\.layer\.[0-9]+.'
    QKV = ('SelfAttention.q', 'SelfAttention.k', 'SelfAttention.v', 'EncDecAttention.q', 'EncDecAttention.k', 'EncDecAttention.v')
    QKVO = QKV + ('SelfAttention.o', 'EncDecAttention.o')
    FFN1 = ('DenseReluDense.wi',)
    FFN2 = ('DenseReluDense.wo')
    ATTENTION = ('SelfAttention', 'EncDecAttention')


if __name__ == '__main__':
    from transformers import T5Model
    _logger.setLevel(40)
    parser = HuggingfaceT5Parser
    config = T5Config()
    model = T5Model(config)
    print(f'{parser.is_huggingface_model(model)}')
    for module_name, module in model.named_modules():
        print(f'{parser.is_attention(module_name)} {parser.is_ffn(module_name)} {parser.get_num_heads(module_name, model)} : {module_name}')
