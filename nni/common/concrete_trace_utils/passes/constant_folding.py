import torch
import torch.fx
from torch.fx import Node
from torch.fx.node import map_arg
from typing import Callable, Union, List, Any


def _format_target(base: str, target: str) -> str:
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r

class ConstantFolding:
    def __init__(self, gm: torch.fx.GraphModule, model: torch.nn.Module):
        self.gm = gm
        self.model = model

    def fold_constant(self):
        for node in self.gm.graph.nodes:
            if node.op != 'get_attr':
                continue
            constant = eval(_format_target('self.model', node.target))
            if not isinstance(constant, torch.nn.Parameter):
                node.replace_all_uses_with(constant)
        
        self.gm.recompile()
