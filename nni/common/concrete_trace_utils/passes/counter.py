"""
This file contains the counter pass for profiling the flops and parameters of a GraphModule.

I failed to reuse the code from nni.compression.pytorch.utils.counter.py
"""
import builtins
import torch
import torch.fx
from torch.fx import Interpreter
from torch.fx.node import Target, Argument, Node
from typing import Any, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field

from .flop_utils import flop_count
from ..kwargs_interpreter import KwargsInterpreter


def _format_flops(flops: float) -> str:
    """Returns a formatted flops string"""
    if flops > 1e12:
        return f'{flops / 1e12:.2f} TFLOPs'
    elif flops > 1e9:
        return f'{flops / 1e9:.2f} GFLOPs'
    elif flops > 1e6:
        return f'{flops / 1e6:.2f} MFLOPs'
    elif flops > 1e3:
        return f'{flops / 1e3:.2f} kFLOPs'
    return f'{flops} FLOPs'


def _format_memory(nbytes) -> str:
    """Returns a formatted memory size string"""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if (abs(nbytes) >= GB):
        return '{:.2f} Gb'.format(nbytes * 1.0 / GB)
    elif (abs(nbytes) >= MB):
        return '{:.2f} Mb'.format(nbytes * 1.0 / MB)
    elif (abs(nbytes) >= KB):
        return '{:.2f} Kb'.format(nbytes * 1.0 / KB)
    else:
        return str(nbytes) + ' b'

def compute_size_in_bytes(elem: Union[torch.Tensor, Dict, List, Tuple, int]) -> int:
    """Compute the size of a tensor or a collection of tensors in bytes.

    Args:
        elem (torch.Tensor | Dict | List | Tuple | int): Arbitrary nested ``torch.Tensor`` data structure.

    Returns:
        int: The size of the tensor or the collection of tensors in bytes.
    """
    nbytes = 0
    if isinstance(elem, torch.Tensor):
        if elem.is_quantized:
            nbytes += elem.numel() * torch._empty_affine_quantized([], dtype=elem.dtype).element_size()
        else:
            nbytes += elem.numel() * torch.tensor([], dtype=elem.dtype).element_size()
    elif isinstance(elem, dict):
        value_list = [v for _, v in elem.items()]
        nbytes += compute_size_in_bytes(value_list)
    elif isinstance(elem, tuple) or isinstance(elem, list) or isinstance(elem, set):
        for e in elem:
            nbytes += compute_size_in_bytes(e)
    return nbytes


@dataclass
class NInfo:
    r"""
    The base class to store all profiling and static graph analysis information
    needed for a ``Node`` in FX.Graph
    """

    # binded back to ``Node``
    node: Node

    # directory
    mod_dir: str = ''   # TODO: trace this in concrete_trace

    # parameter within this ``Node``
    parameters: Dict[str, torch.nn.Parameter] = field(default_factory=lambda: {})

    # compute cost
    flops: Optional[int] = 0

    def __new__(cls, node: Node, **kwargs):
        orig_init = cls.__init__

        # if initialized, return the existing one
        # should disable the __init__ function
        if node.meta.get('info', None) is not None:

            def _dummy(self, *args, **kwargs):
                if getattr(self, '_is_init', False):
                    self._is_init = True
                    orig_init(self, *args, **kwargs)
                cls.__init__ = orig_init

            cls.__init__ = _dummy
            return node.meta['info']
        return super().__new__(cls)

    def __post_init__(self):
        self.node.meta['info'] = self
    
    @property
    def param_size(self) -> int:
        return compute_size_in_bytes(self.parameters)


class GraphCounter(KwargsInterpreter):
    _to_profile = ['call_module', 'call_function']
    _maybe_profile = ['get_attr']

    flop_count_skip_functions = [builtins.next, ]

    def __init__(self, module):
        super().__init__(module)

    def run_node(self, node: Node) -> Any:
        """Dispatch to the appropriate method for running a node.

        This method inherits ``run_node`` in `Interpreter` but adds the following features:

        - ``call_module`` and ``call_function`` are the only two types of nodes that are profiled.
        - ``call_method``, ``placeholder``, and ``output`` are not profiled because they are not 
        computationally intensive.
        - ``get_attr`` is a maybe_profiled node. It is profiled because its ``rst`` can be a `nn.Parameter`.

        Parameters
        ----------
        node: Node
            The node to run.

        Returns
        -------
        rst: Any
            The result of running the node.
        """
        rst = super().run_node(node)
        if node.op in self._to_profile:
            n_info = NInfo(node, flops=rst[1], parameters=rst[2])
            rst = rst[0]
        elif node.op in self._maybe_profile:
            n_info = NInfo(node, parameters=rst[2])
            rst = rst[0]
        else:
            n_info = NInfo(node)
        return rst
    
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        rst = super().call_function(target, args, kwargs)
        if target in self.flop_count_skip_functions:
            flop = 0
        else:
            try:
                flop = flop_count(target, *args, **kwargs)
            except RuntimeError:
                print(f'INFO: RuntimeError when flop count {target}, set the count result as 0')
                flop = 0
        return rst, flop, {}
        
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return (
            submod(*args, **kwargs), 
            flop_count(submod, *args, **kwargs),
            {
                k: v for k, v in submod.named_parameters()
            }
        )
    
    def get_attr(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        rst = self.fetch_attr(target)
        if isinstance(rst, torch.nn.Parameter):
            return (
                rst,
                0,
                {target: rst}
            )
        return rst, 0, {}
    
    def summarize(self) -> Dict[str, NInfo]:
        """
        Summarizes the profiled statistics of the `GraphModule` in
        tabular format. Note that this API requires the ``tabulate`` module
        to be installed.

        Returns:
            str: The summary of the profiled statistics
        """
        # https://github.com/pytorch/pytorch/blob/master/torch/fx/graph.py
        try:
            from tabulate import tabulate
        except ImportError:
            print("`summary` relies on the library `tabulate`, "
                  "which could not be found on this machine. Run `pip "
                  "install tabulate` to install the library.")

        # Build up a list of summary information for each node
        node_summaries: List[List[Any]] = []

        for node in self.module.graph.nodes:
            node: Node
            n_info = NInfo(node)
            node_summaries.append([
                node.op,
                str(node),
                _format_memory(n_info.param_size),
                _format_flops(n_info.flops),
            ])

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers: List[str] = [
            'Op type',
            'Op',
            'Param size',
            'FLOPs',
        ]

        return tabulate(node_summaries, headers=headers, stralign='right')
    

def counter_pass(module: torch.fx.GraphModule, concrete_args, verbose=False) -> torch.fx.GraphModule:
    """A pass that counts the number of FLOPs and parameters in a model.

    Parameters
    ----------
    module: torch.fx.GraphModule
        The module to be profiled.
    
    verbose: bool
        Whether to print the summary of the profiled statistics. Default: False.

    Returns
    -------
    module: torch.fx.GraphModule
        The same module as the input.
    """
    interp = GraphCounter(module)
    interp.run(concrete_args)
    if verbose:
        print(interp.summarize())
    return module


if __name__ == '__main__':
    from torch.fx import symbolic_trace
    import torchvision.models as tm

    zoo = [
        tm.alexnet,
        tm.convnext_base,   # stochastic depth
        tm.densenet121,
        tm.efficientnet_v2_s,   # stochastic depth
        tm.googlenet,   # output bad case
        # tm.inception_v3,  # bad case
        tm.mobilenet_v2,
        tm.mobilenet_v3_small,
        tm.mnasnet0_5,
        tm.resnet18,
        tm.regnet_x_16gf,
        tm.resnext50_32x4d,
        tm.shufflenet_v2_x0_5,
        tm.squeezenet1_0,
        # tm.swin_s,  # fx bad case
        tm.vgg11,
        tm.vit_b_16,
        tm.wide_resnet50_2,
    ]

    for m in zoo:
        model = m()
        model = symbolic_trace(model)
        model = counter_pass(model, torch.randn(8, 3, 224, 224), verbose=True)
