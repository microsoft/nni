import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx import Interpreter, Node
from torch.fx.node import map_arg

from typing import Optional, Union, Tuple, Dict, List, Any, Iterator, Callable

Target = Union[Callable[..., Any], str]

Argument = Optional[Union[
    Tuple[Any, ...],
    List[Any],
    Dict[str, Any],
    Node
]]

class KwargsInterpreter(Interpreter):
    def run(self, 
            concrete_args: Union[Dict[str, Any], Tuple],
            initial_env: Optional[Dict[Node, Any]] = None,
            enable_io_preocessing: bool = True) -> Any:
        
        self.env = initial_env if initial_env else {}


        if isinstance(concrete_args, tuple):
            # if concrete_args is a tuple, then they are positional args
            # then they are consumed left-to-right by `placeholder` nodes.
            # Use an iterator to keep track of position and extract those values
            # TODO: to support positional arguments
            if enable_io_preocessing:
                args = self.module.graph.process_inputs(*concrete_args)
            self.args_iter: Iterator[Any] = iter(args)
        else:
            # concrete_args is a kwargs dict
            self.concrete_kwargs = concrete_args
            self.used_concrete_kwargs = []
            import inspect
            fw = inspect.unwrap(self.module.forward)
            args_default_values = fw.__defaults__
            if args_default_values is not None:
                fw_code = fw.__code__
                n_args = fw_code.co_argcount + fw_code.co_kwonlyargcount
                names_iter = iter(fw_code.co_varnames)
                start_idx = 0
                if fw_code.co_varnames[0] == 'self':
                    _ = next(names_iter)  # skip self
                    start_idx = 1
                args_names = [next(names_iter) for idx in range(start_idx, n_args)]
                diff_len = len(args_names) - len(args_default_values)
                self.default_args = {args_names[idx + diff_len]: args_default_values[idx] for idx in range(len(args_default_values))}
            else:
                self.default_args = {}

        for node in self.module.graph.nodes:
            if node in self.env:
                continue
            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                msg = f'While executing {node.format_node()}'
                msg = '{}\n\n{}'.format(e.args[0], msg) if e.args else str(msg)
                msg += f"\nOriginal traceback:\n{node.stack_trace}"
                e.args = (msg,) + e.args[1:]
                if isinstance(e, KeyError):
                    raise RuntimeError(*e.args)
                raise                

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]
            
            if node.op == 'output':
                output_val = self.env[node]
                return self.module.graph.process_outputs(output_val) if enable_io_preocessing else output_val
            
    def run_node(self, n: Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        with fx_traceback.append_stack_trace(n.stack_trace):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            return getattr(self, n.op)(n.target, args, kwargs)
        
    def placeholder(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a `placeholder` node. 

        Args:
            target(Target): The call target for this node, 
                exactly the argument name of the forward function
            args(Tuple): Tuple of positional args for this invocation
            kwargs(Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        assert isinstance(target, str)
        if target.startswith('**'):
            # For a douvle-starred parameter, e.g., `**kwargs`, 
            # retrieve all the remaining values from the concrete kwargs dict
            remaining_keys = [key for key in self.concrete_kwargs if key not in self.used_concrete_kwargs]
            return {key: self.concrete_kwargs[key] for key in remaining_keys}
        elif target.startswith('*'):
            raise RuntimeError('positional args supported')
        else:
            try:
                ret_arg = self.concrete_kwargs[target]
            except KeyError as ke:
                # TODO: deal with the arguments with default values
                return self.default_args[target]
                # raise RuntimeError(f'Expected keyword argument for parameter {target}, but one was not passed in!')
            else:
                self.used_concrete_kwargs.append(target)
                return ret_arg

    def call_method(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a `call_method` node and return the result

        Args:
            target(Target): The call target for this node.
            args(Tuple): Tuple of positional args for this invocation.
            kwargs(Dict): Dict of keyword arguments for this invocation.

        Returns:
            Any: The value returned by the method invocation
        """
        # args[0] is the object the method call belongs to
        obj, *args_tail = args

        # Execute the method call and return the result
        assert isinstance(target, str)
        if not hasattr(obj, target):
            # to handle the case where there is a `**kwargs` in the parameters of forward function
            print(f'WARNING: skipped nonexistent attr of {obj}.{target}')
            return {}
        else:
            return getattr(obj, target)(*args_tail, **kwargs)
    
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument:
        """
        Recursively descend through ``args`` and look up the concrete value
        for each `Node` in the current execution environment
        """
        def load_arg(n_arg: Node) -> Any:
            """
            Args:
                n_arg(Node): an arg whose type is Node

            Returns:
                the concrete value of the node arg
            """
            if n_arg not in self.env:
                raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() '
                                   f'to diagnose such issues')
            return self.env[n_arg]
        return map_arg(args, load_arg)