from operator import mod
import torch

support_patterns = ["mobilenet_coarse_int8", "bert_coarse_fp32", "bert_coarse_int8", "hubert_coarse_fp32", "hubert_coarse_int8"]

def generate_code(config: dict, pattern: str) -> dict:
    assert(pattern in support_patterns, f"only support support_patterns: {support_patterns}")
    if pattern == "mobilenet_coarse_int8":
        result = mobilenet_coarse_int8_codegen(config)
    elif pattern == "bert_coarse_fp32":
        result = bert_coarse_fp32_codegen(config)
    elif pattern == "bert_coarse_int8":
        result = bert_coarse_int8_codegen(config)
    elif pattern == "hubert_coarse_fp32":
        result = hubert_coarse_fp32_codegen(config)
    elif pattern == "hubert_coarse_int8":
        result = hubert_coarse_int8_codegen(config)
    return result

def mobilenet_coarse_int8_codegen(config: dict) -> dict:
    ...

def bert_coarse_fp32_codegen(config: dict) -> dict:
    for name, val_dict in config:
        
    ...

def bert_coarse_int8_codegen(config: dict) -> dict:
    ...

def hubert_coarse_fp32_codegen(config: dict) -> dict:
    ...

def hubert_coarse_int8_codegen(config: dict) -> dict:
    ...