# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler


def _replace_module_with_type(root_module, prior_replace, default_replace, modules):
    """
    Replace xxxChoice in user's model with NAS modules.valuechoice

    Parameters
    ----------
    root_module : nn.Module
        User-defined module with xxxChoice in it. In fact, since this method is
        called in the ``__init__`` of ``BaseOneShotLightningModule``, this will be a
        pl.LightningModule.
    prior_match_and_replace : List[Callable[[nn.Module], (nn.Module, nn.Module)]]
        Takes an nn.Module as input and returns another nn.Module if replacement is needed or
        to override default replace method, otherwise returns None to leave it as default. Prior
        means if this method returns a not None result for a module, the following
        ''match_and_replace'' will be ignored.
    match_and_replace : List[Callable[[nn.Module], (nn.Module, nn.Module)]]
        Takes an nn.Module as input and returns another nn.Module if replacement is needed.
        Returns None otherwise.
    modules : List[nn.Module]
        The replace result. This is also the return value of this function.

    Returns
    ----------
    modules : List[nn.Module]
        The replace result.
    """
    if modules is None:
        modules = []

    def apply(m):
        for name, child in m.named_children():
            # 先使用用户的 match and replace，如果返回了一个 Module，就认为覆盖 default 实现
            # 如果返回 None，就认为使用 default 实现
            # 应该不会有用户返回 None 但想表达不替换的意思的情况把？
            replace_result = None
            if prior_replace is not None:
                for f in prior_replace:
                    replace_result = f(child)
                    if replace_result is not None:
                        break
            
            if replace_result is None:
                for f in default_replace:
                    replace_result = f(child)
                    if replace_result is not None:
                        break
            
            # 如果用户的和 default 的至少有一个没返回 None，就是要替换
            if replace_result is not None:
                # 为了适配 valuechoice，分成 tosample 和 to replace 两个
                # tosample 的是原来那个 valuechoice
                # toreplace 的是一个具体的 module，比如 Linear, conv 可能都要分别实现
                to_samples, to_replace = replace_result
                setattr(m, name, to_replace)
                if isinstance(to_samples, list):
                    for to_sample in to_samples:
                        item = (to_sample.label, to_sample)
                        if item not in modules:
                            modules.append(item)
                else:
                    item = (to_samples.label, to_samples)
                    if item not in modules:
                        modules.append(item)
            else:
                apply(child)

    apply(root_module)
    return modules


class BaseOneShotLightningModule(pl.LightningModule):
    """
    The base class for all one-shot NAS models. Essential function such as preprocessing user's model,
    redirecting lightning hooks for user's model, configuring optimizers and exporting NAS result are
    implemented in this class.

    Attributes
    ----------
    nas_modules : List[nn.Module]
        modules a paticular NAS method replaces with

    Parameters
    ----------
    base_model : pl.LightningModule
        The evaluator in ``nni.retiarii.evaluator.lightning``. User defined model is wrapped by base_model,
        and base_model will be wrapped by this model.
    custom_match_and_replace : List[Callable[[nn.Module], (nn.Module, nn.Moduel)]]
        The custom xxxChoice match and replace method. Each method should take an nn.Module and yields
        two modules (to_sample, to_replace). The ''to_sample'' will be placed in ''self.nas_module'' to be
        sampled, and the ''to_replace'' will replace the input module and forward instead of it.
    """
    automatic_optimization = False

    def __init__(self, base_model, custom_match_and_replace=[lambda m: None]):
        super().__init__()
        assert isinstance(base_model, pl.LightningModule)
        self.model = base_model

        # replace xxxChoice with respect to NAS alg
        # replaced modules are stored in self.nas_modules
        self.nas_modules = []
        _replace_module_with_type(
            self.model, custom_match_and_replace, self.match_and_replace(), self.nas_modules)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # You can use self.architecture_optimizers or self.user_optimizers to get optimizers in
        # your own training step.
        return self.model.training_step(batch, batch_idx)

    def configure_optimizers(self):
        """
        Combine architecture optimizers and user's model optimizers.
        Overwrite configure_architecture_optimizers if architecture optimizers
        are needed in your NAS algorithm.
        By now ``self.model`` is currently a :class:`nni.retiarii.evaluator.pytorch.lightning.
        _SupervisedLearningModule`, and it only returns 1 optimizer.
        But for extendibility, codes for other return value types are also implemented.
        """
        # pylint: disable=assignment-from-none
        arc_optimizers = self.configure_architecture_optimizers()
        if arc_optimizers is None:
            return self.model.configure_optimizers()

        if isinstance(arc_optimizers, optim.Optimizer):
            arc_optimizers = [arc_optimizers]
        self.arc_optim_count = len(arc_optimizers)

        # The third return value ``frequency`` and ``monitor`` are ignored because lightning requires
        # ``len(optimizers) == len(frequency)``, and gradient backword is handled manually.
        # For data structure of variables below, please see pytorch lightning docs of ``configure_optimizers``.
        w_optimizers, lr_schedulers, self.frequencies, monitor = \
            self.trainer._configure_optimizers(self.model.configure_optimizers())
        lr_schedulers = self.trainer._configure_schedulers(lr_schedulers, monitor, not self.automatic_optimization)
        if any(sch["scheduler"].optimizer not in w_optimizers for sch in lr_schedulers):
            raise Exception(
            "Some schedulers are attached with an optimizer that wasn't returned from `configure_optimizers`."
        )

        # variables used to handle optimizer frequency
        self.cur_optimizer_step = 0
        self.cur_optimizer_index = 0

        return arc_optimizers + w_optimizers, lr_schedulers

    def on_train_start(self):
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
        Hook kept for subclasses. Each specific NAS method returns the optimizer of architecture
        parameters. Note that lr schedulers are not supported now for architecture_optimizers.

        Returns
        ----------
        arc_optimizers : List[Optimizer], Optimizer
            Optimizers used by a specific NAS algorithm for its architecture parameters.
            Return None if no architecture optimizers are needed.
        """
        return None

    @staticmethod
    def match_and_replace():
        """
        Default xxxChoice replace method. This is called in __init__ to get the default replace
        functions for your NAS algorithm. Note that your default replace functions will be
        override by user-defined custom_replace_dict.

        Returns
        ----------
        match_and_replace_funcs : List[Callable[[nn.Module], (nn.Module, nn.Module)]]
            The replace function list. Each function takes an nn.Module and yields two modules like
            ''to_sample, to_replace''. The ''to_sample'' will be placed in ''self.nas_module'' to be
            sampled, and the ''to_replace'' will replace the input module and forward instead of it.
        """
        return []

    def call_lr_schedulers(self, batch_index):
        """
        Function that imitates lightning trainer's behaviour of calling user's lr schedulers. Since
        auto_optimization is turned off by this class, you can use this function to make schedulers
        behave as they were automatically handled by the lightning trainer.

        Parameters
        ----------
        batch_idx : int
            batch index
        """
        def apply(lr_scheduler):
            # single scheduler is called every epoch
            if  isinstance(lr_scheduler, _LRScheduler) and \
                self.trainer.is_last_batch:
                lr_schedulers.step()
            # lr_scheduler_config is called as configured
            elif isinstance(lr_scheduler, dict):
                interval = lr_scheduler['interval']
                frequency = lr_scheduler['frequency']
                if  (
                        interval == 'step' and
                        batch_index % frequency == 0
                    ) or \
                    (
                        interval == 'epoch' and
                        self.trainer.is_last_batch and
                        (self.trainer.current_epoch + 1) % frequency == 0
                    ):
                        lr_scheduler.step()

        lr_schedulers = self.lr_schedulers()

        if isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                apply(lr_scheduler)
        else:
            apply(lr_schedulers)

    def call_user_optimizers(self, method):
        """
        Function that imitates lightning trainer's behaviour of calling user's optimizers. Since
        auto_optimization is turned off by this class, you can use this function to make user
        optimizers behave as they were automatically handled by the lightning trainer.

        Parameters
        ----------
        method : str
            Method to call. Only ``step`` and ``zero_grad`` are supported now.
        """
        def apply_method(optimizer, method):
            if method == 'step':
                optimizer.step()
            elif method == 'zero_grad':
                optimizer.zero_grad()

        optimizers = self.user_optimizers
        if optimizers is None:
            return

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

    @property
    def architecture_optimizers(self):
        """
        Get architecture optimizers from all optimizers. Use this to get your architecture optimizers
        in training stage.

        Returns
        ----------
        opts : List[Optimizer], Optimizer, None
            Architecture optimizers you defined in ``configure_architecture_optimizers``. This
            will be None if there is no architecture optimizers.
        """
        opts = self.optimizers()
        if isinstance(opts,list):
            # pylint: disable=unsubscriptable-object
            arc_opts = opts[:self.arc_optim_count]
            if len(arc_opts) == 1:
                arc_opts = arc_opts[0]
            return arc_opts
        # If there is only 1 optimizer and it is the architecture optimizer
        if self.arc_optim_count == 1:
            return opts
        return None

    @property
    def user_optimizers(self):
        """
        Get user optimizers from all optimizers. Use this to get user optimizers in training stage.

        Returns
        ----------
        opts : List[Optimizer], Optimizer, None
            Optimizers defined by user's model. This will be None if there is no user optimizers.
        """
        opts = self.optimizers()
        if isinstance(opts,list):
            # pylint: disable=unsubscriptable-object
            return opts[self.arc_optim_count:]
        # If there is only 1 optimizer and no architecture optimizer
        if self.arc_optim_count == 0:
            return opts
        return None

    def export(self):
        """
        Export the NAS result, idealy the best choice of each nas_modules.
        You may implement an export method for your customized nas_module.

        Returns
        --------
        result : Dict[str, int]
            Keys are names of nas_modules, and values are the choice index of it.
        """
        result = {}
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
