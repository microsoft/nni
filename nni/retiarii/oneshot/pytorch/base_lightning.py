# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from warnings import warn
from typing import Any, Dict
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn


def _replace_module_with_type(root_module, replace_dict, modules):
    """
    Replace xxxChoice in user's model with NAS modules.

    Parameters
    ----------
    root_module : nn.Module
        User-defined module with xxxChoice in it. In fact, since this method is
        called in the ''__init__'' of BaseOneShotLightningModule, this will be a
        pl.LightningModule.
    replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]]
        Functions to replace xxxChoice modules. Keys should be xxxChoice type and values should be a
        function that return an nn.module.
    modules : list[ nn.Module ]
        The replace result. This is also the return value of this function.

    Returns
    ----------
    modules : list[ nn.Module ]
        The replace result.
    """
    if modules is None:
        modules = []

    def apply(m):
        for name, child in m.named_children():
            child_type = type(child)
            if child_type in replace_dict.keys():
                setattr(m, name, replace_dict[child_type](child))
                modules.append((child.key, getattr(m, name)))
            else:
                apply(child)

    apply(root_module)
    return modules


class BaseOneShotLightningModule(pl.LightningModule):
    """
    The base class for all one-shot NAS models. Essential functions such as
    preprocessing user's model, redirect lightning hooks for user's model, configure
    optimizers and export NAS result are implemented in this class.

    Attributes
    ----------
    nas_modules : list[nn.Module]
        modules a paticular NAS method replaces with

    Parameters
    ----------
    base_model : pl.LightningModule
        The module in evaluators in nni.retiarii.evaluator.lightning. User defined model
        is wrapped by base_model, and base_model will be wrapped by this model.
    custom_replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]], default = None
        The custom xxxChoice replace method. Keys should be xxxChoice type and values should
        return an nn.module. This custom replace dict will override the default replace
        dict of each NAS method.
    """
    def __init__(self, base_model, custom_replace_dict=None):
        super().__init__()
        assert isinstance(base_model, pl.LightningModule)
        self.model = base_model

        # replace xxxChoice with respect to NAS alg
        # replaced modules are stored in self.nas_modules
        self.nas_modules = []
        choice_replace_dict = self.default_replace_dict
        if custom_replace_dict is not None:
            for k, v in custom_replace_dict.items():
                assert isinstance(v, nn.Module)
                choice_replace_dict[k] = v
        _replace_module_with_type(
            self.model, choice_replace_dict, self.nas_modules)

    def __getattr__(self, name):
        """
        Redirect lightning hooks.
        Only hooks in the list below will be redirect to user-deifend ones.
        The list is not customizable now. But if you need to redirect other user-defined
        hooks in your NAS method, you can override it and call user-defined methods manually.

        Warnings
        ----------
        Validation-related hooks are bypassed as default, since architecture selection
        is exactly what NAS aims to do.
        """
        if name in [
            'on_train_end',
            'on_fit_start',
            'on_fit_end',
            'on_train_batch_start',
            'on_train_batch_end',
            'on_epoch_start',
            'on_epoch_end',
            'on_train_epoch_start',
            'on_train_epoch_end',
            'on_before_backward',
            'on_after_backward',
            'configure_gradient_clipping'
        ]:
            return getattr(self.__dict__['_modules']['model'], name)
        return super().__getattr__(name)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        warn('Validation is skipped by the NAS method you chose.')

    def configure_optimizers(self):
        """
        Combine architecture optimizers and user's model optimizers.
        Overwrite configure_architecture_optimizers if architecture optimizers
        are needed in your NAS algorithm.
        """
        arc_optimizers = self.configure_architecture_optimizers()
        if isinstance(arc_optimizers, optim.Optimizer):
            arc_optimizers = [arc_optimizers]

        if arc_optimizers is None:
            return self.model.configure_optimizers()

        if isinstance(arc_optimizers, optim.Optimizer):
            arc_optimizers = [arc_optimizers]
        self.arc_optim_count = len(arc_optimizers)

        w_optimizers = self.model.configure_optimizers()
        if isinstance(w_optimizers, optim.Optimizer):
            w_optimizers = [w_optimizers]
        else:
            w_optimizers = list(w_optimizers)

        return arc_optimizers + w_optimizers

    def configure_architecture_optimizers(self):
        '''
        Hook kept for subclasses. Each specific NAS method returns the optimizer of
        architecture parameters.

        Returns
        ----------
        arc_optimizers : List[Optimizer], Optimizer
            optimizers used by a specific NAS algorithm for its architecture parameters
        '''
        arc_optimizers = []
        return arc_optimizers

    @property
    def default_replace_dict(self):
        """
        Default xxxChoice replace dict. This is called in __init__ to get the default replace
        functions for your NAS algorithm. Note that your default replace functions will be
        override by user-defined custom_replace_dict.

        Returns
        ----------
        replace_dict : dict{ type : func }
            Keys should be xxxChoice type, and values should be a function that returns a nn.Module
            instance. This is of the same type of 'custom_replace_dict' in __init__, but this will
            be override if users defined their own replace functions.
        """
        replace_dict = {}
        return replace_dict

    def export(self) -> Dict[str, Any]:
        """
        Export the NAS result, idealy the best choice of each nas_modules.
        You may implement an export method for your customized nas_module.

        Returns
        --------
        result : dict { str : int }
            Keys are names of nas_modules, and values are the choice of each nas_module.
        """
        result = {}
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
