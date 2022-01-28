# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from warnings import warn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler


def _replace_module_with_type(root_module, replace_dict, modules):
    """
    Replace xxxChoice in user's model with NAS modules.

    Parameters
    ----------
    root_module : nn.Module
        User-defined module with xxxChoice in it. In fact, since this method is
        called in the ``__init__`` of BaseOneShotLightningModule, this will be a
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
    automatic_optimization = False

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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # You can use self.optimizers() in training_step to get a list of all optimizers.
        # Model optimizers comes after architecture optimizers, and the number of architecture
        # optimizers is self.arc_optim_count.
        #
        # Example :
        # optims = self.optimizers()
        # arc_optims = optims[:self.arc_optim_count] # architecture optimizers
        # w_optimizers = optims[self.arc_optim_count:] # model optimizers

        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        warn('Validation is skipped by the NAS method you chose.')

    def configure_optimizers(self):
        """
        Combine architecture optimizers and user's model optimizers.
        Overwrite configure_architecture_optimizers if architecture optimizers
        are needed in your NAS algorithm.
        By default ``self.model`` is currently a _SupervisedLearningModule in
        nni.retiarii.evaluator.pytorch.lightning, and it only returns 1 optimizer.
        But for extendibility, codes for other return value types are also implemented.
        """
        # pylint: disable=assignment-from-none
        arc_optimizers = self.configure_architecture_optimizers()
        if arc_optimizers is None:
            return self.model.configure_optimizers()

        if isinstance(arc_optimizers, optim.Optimizer):
            arc_optimizers = [arc_optimizers]
        self.arc_optim_count = len(arc_optimizers)

        # The third return value ``frequency`` and ``monitor`` are ignored since lightning
        # requires len(optimizers) == len(frequency), and gradient backword
        # is handled manually.
        w_optimizers, lr_schedulers, self.frequencies, monitor = \
            self.trainer._configure_optimizers(self.model.configure_optimizers())

        lr_schedulers = self.trainer._configure_schedulers(lr_schedulers, monitor, not self.automatic_optimization)
        if any(sch["scheduler"].optimizer not in w_optimizers for sch in lr_schedulers):
            raise Exception(
            "Some schedulers are attached with an optimizer that wasn't returned from `configure_optimizers`."
        )

        self.cur_optimizer_step = 0
        self.cur_optimizer_index = 0

        return arc_optimizers + w_optimizers, lr_schedulers

    def on_train_start(self):
        # let users have access to the trainer and log
        self.model.trainer = self.trainer
        self.model.log = self.log
        return self.model.on_train_start()

    def on_train_end(self):
        return self.model.on_train_end()

    def on_fit_start(self):
        return self.model.on_train_start()

    def on_fit_end(self):
        return self.model.on_train_end()

    def on_train_batch_start(self, batch, batch_idx, unused = 0):
        return self.model.on_train_batch_start(batch, batch_idx, unused)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused = 0):
        return self.model.on_train_batch_end(outputs, batch, batch_idx, unused)

    def on_epoch_start(self):
        return self.model.on_epoch_start()

    def on_epoch_end(self):
        return self.model.on_epoch_end()

    def on_train_epoch_start(self):
        return self.model.on_train_epoch_start()

    def on_train_epoch_end(self):
        return self.model.on_train_epoch_end()

    def on_before_backward(self, loss):
        return self.model.on_before_backward(loss)

    def on_after_backward(self):
        return self.model.on_after_backward()

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val = None, gradient_clip_algorithm = None):
        return self.model.configure_gradient_clipping(optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm)

    def configure_architecture_optimizers(self):
        """
        Hook kept for subclasses. Each specific NAS method returns the optimizer of
        architecture parameters.

        Returns
        ----------
        arc_optimizers : List[Optimizer], Optimizer
            Optimizers used by a specific NAS algorithm for its architecture parameters.
            Return None if no architecture optimizers are needed.
        """
        return None

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

    def call_lr_schedulers(self, batch_index):
        def apply(lr_scheduler):
            # single scheduler is called every epoch
            if  isinstance(lr_scheduler, _LRScheduler) and \
                self.trainer.is_last_batch:
                lr_schedulers.step()
            # lr_scheduler_config is called as configured
            elif isinstance(lr_scheduler, dict):
                interval = lr_scheduler['interval']
                frequency = lr_scheduler['frequency']
                if  interval == 'step' and \
                    batch_index % frequency == 0 \
                    or \
                    interval == 'epoch' and \
                    self.trainer.is_last_batch and \
                    (self.trainer.current_epoch + 1) % frequency == 0:
                        lr_scheduler.step()

        lr_schedulers = self.lr_schedulers()

        if isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                apply(lr_scheduler)
        else:
            apply(lr_schedulers)

    def call_user_optimizers(self, optimizers, method):
        def apply_method(optimizer, method):
            if method == 'step':
                optimizer.step()
            elif method == 'zero_grad':
                optimizer.zero_grad()

        if len(self.frequencies) > 0:
            self.cur_optimizer_step += 1
            if self.frequencies[self.cur_optimizer_index] == self.cur_optimizer_step:
                self.cur_optimizer_step = 0
                self.cur_optimizer_index = self.cur_optimizer_index + 1 \
                    if self.cur_optimizer_index + 1 < len(optimizers) \
                    else 0
            apply_method(optimizers[self.cur_optimizer_index], method)
        else:
            for optimizer in optimizers:
                apply_method(optimizer, method)



    def export(self):
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
