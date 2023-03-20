# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['PytorchGraphModelSpace']

import logging
from typing import Any, TYPE_CHECKING

import torch

from nni.nas.evaluator import Evaluator
from nni.nas.space import GraphModelSpace, Mutator
from nni.nas.nn.pytorch.repeat import repeat_jit_forward_patch
from .codegen import model_to_pytorch_script
from .converter import GraphConverter, GraphConverterWithShape
from .mutator import process_inline_mutation

if TYPE_CHECKING:
    from nni.nas.nn.pytorch import ModelSpace

_logger = logging.getLogger(__name__)


class PytorchGraphModelSpace(GraphModelSpace):
    """:class:`~nni.nas.space.GraphModelSpace` specialized for PyTorch.
    It converts a PyTorch model into a graph, and provides a method to convert the graph back.

    Warning
    -------
    As of now, :class:`PytorchGraphModelSpace` is known to be problematic, and NOT recommended to use unless necessary.

    Firstly, the graph converter will put users' models through :meth:`torch.jit.script`,
    which will cause some ad-hoc models to fail.
    The traced model will be converted into our graph representation,
    which has quite limited supported for control flows, loops, etc.

    Other than unsupported types of models,
    the graph converter will also induce some unexpected behaviors due to the implementation changes.
    For example candidate names of :class:`~nni.nas.nn.pytorch.LayerChoice` will be prefixed,
    so that ``freeze()`` might not work as expected.
    """

    framework_type: str = 'pytorch'

    @classmethod
    @repeat_jit_forward_patch()
    def from_model(cls, model_space: ModelSpace, evaluator: Evaluator | None = None,
                   dummy_input: tuple[int, ...] | tuple[torch.Tensor, ...] | list[int] | None = None) -> GraphModelSpace:
        """Create a GraphModelSpace instance based on a model and evaluator.
        Model-to-IR conversion happens here.
        """
        if isinstance(dummy_input, list):
            dummy_input = tuple(dummy_input)

        try:
            script_module = torch.jit.script(model_space)
        except:
            _logger.error('Your base model cannot be parsed by torch.jit.script, please fix the following error:')
            raise
        if dummy_input is not None:
            if isinstance(dummy_input, tuple) and all(isinstance(i, int) for i in dummy_input):
                dummy_input = torch.randn(*dummy_input)  # type: ignore
            converter = GraphConverterWithShape()
            base_model_ir = cls.convert_to_graph(script_module, model_space, converter, dummy_input=dummy_input)
        else:
            base_model_ir = cls.convert_to_graph(script_module, model_space)

        mutator_generated = len(base_model_ir.mutators) > 0

        if hasattr(model_space, 'mutables'):
            for mutable in model_space.mutables:
                if isinstance(mutable, Mutator) and mutator_generated:
                    base_model_ir.mutators.append(mutable)
                elif not isinstance(mutable, Mutator):
                    _logger.warning(f'Mutable is not a mutator. Will be ignored: {mutable}')

        base_model_ir.evaluator = evaluator

        mutators = process_inline_mutation(base_model_ir)
        if len(base_model_ir.mutators) > 0 and mutators:
            _logger.warning('Some mutators have been generated automatically. '
                            'We do not recommend a mixed usage of generated mutator and manually defined mutator, '
                            'because sometimes it induces unexpected side effects.')
        base_model_ir.mutators.extend(mutators)

        return base_model_ir

    @classmethod
    def convert_to_graph(cls, script_module, module, converter=None, **kwargs):
        """
        Convert module to our graph ir, i.e., build a :class:`GraphModelSpace` type.

        Parameters
        ----------
        script_module : torch.jit.RecursiveScriptModule
            the script module obtained with torch.jit.script
        module : nn.Module
            the targeted module instance
        converter : `TorchConverter`
            default `GraphConverter` is used
        kwargs:
            will be passed to `converter.convert_module()`

        Returns
        -------
        GraphModelSpace
            the constructed IR model
        """
        model = cls(_internal=True)
        module_name = '_model'
        if converter is None:
            converter = GraphConverter()
        converter.convert_module(script_module, module, module_name, model, **kwargs)
        return model

    def to_code(self) -> str:
        """Convert the model to Python code."""
        return model_to_pytorch_script(self)

    def executable_model(self) -> Any:
        """Convert the model to Python code, and execute the code to get the model."""
        model_code = self.to_code()
        _logger.debug('Generated model code:')
        _logger.debug(model_code)
        exec_vars = {}
        try:
            exec(model_code, exec_vars)
        except:
            _logger.critical('Generated model code cannot be executed, please report this issue to NNI. The code is:\n%s', model_code)
            raise
        model_cls = exec_vars['_model']
        return model_cls()
