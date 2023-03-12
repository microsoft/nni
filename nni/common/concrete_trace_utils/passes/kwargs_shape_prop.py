import torch
import traceback
from torch.fx.node import Node, map_aggregate
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from typing import Union, Tuple, Any, Dict
from ..kwargs_interpreter import KwargsInterpreter

class KwargsShapeProp(KwargsInterpreter):
    def run_node(self, n: Node):
        try:
            result = super().run_node(n)
        except Exception:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with "
                f"meta={n.meta}"
            )
        
        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj
            
        # if the obj is a tensor, then wrap it into a TensorMetaData
        # else recursively descend and wrap
        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta
        n.meta['type'] = type(result)
        return result
    
    def propagate(self, concrete_args: Union[Dict[str, Any], Tuple]):
        return super().run(concrete_args)
