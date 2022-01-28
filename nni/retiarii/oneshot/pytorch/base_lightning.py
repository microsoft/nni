# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from warnings import warn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


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
        _replace_module_with_type(self.model, choice_replace_dict, self.nas_modules)

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

    @staticmethod
    def _configure_optimizers(optim_conf):
        optimizers, lr_schedulers, optimizer_frequencies = [], [], []
        monitor = None

        # single output, single optimizer
        if isinstance(optim_conf, Optimizer):
            optimizers = [optim_conf]
        # two lists, optimizer + lr schedulers
        elif (
            isinstance(optim_conf, (list, tuple))
            and len(optim_conf) == 2
            and isinstance(optim_conf[0], list)
            and all(isinstance(opt, Optimizer) for opt in optim_conf[0])
        ):
            opt, sch = optim_conf
            optimizers = opt
            lr_schedulers = sch if isinstance(sch, list) else [sch]
        # single dictionary
        elif isinstance(optim_conf, dict):
            optimizers = [optim_conf["optimizer"]]
            monitor = optim_conf.get("monitor", None)
            lr_schedulers = [optim_conf["lr_scheduler"]] if "lr_scheduler" in optim_conf else []
        # multiple dictionaries
        elif isinstance(optim_conf, (list, tuple)) and all(isinstance(d, dict) for d in optim_conf):
            for opt_dict in optim_conf:
                optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
                scheduler_dict = (
                    lambda scheduler, opt_idx: dict(scheduler, opt_idx=opt_idx)
                    if isinstance(scheduler, dict)
                    else {"scheduler": scheduler, "opt_idx": opt_idx}
                )

                lr_schedulers = [
                    scheduler_dict(opt_dict["lr_scheduler"], opt_idx)
                    for opt_idx, opt_dict in enumerate(optim_conf)
                    if "lr_scheduler" in opt_dict
                ]
                optimizer_frequencies = [
                    opt_dict["frequency"] for opt_dict in optim_conf if opt_dict.get("frequency", None) is not None
                ]
                # assert that if frequencies are present, they are given for all optimizers
                if optimizer_frequencies and len(optimizer_frequencies) != len(optimizers):
                    raise ValueError("A frequency must be given to each optimizer.")
        # single list or tuple, multiple optimizer
        elif isinstance(optim_conf, (list, tuple)) and all(isinstance(opt, Optimizer) for opt in optim_conf):
            optimizers = list(optim_conf)
        # unknown configuration
        else:
            raise Exception(
                "Unknown configuration for model optimizers."
                " Output from `model.configure_optimizers()` should either be:\n"
                " * `torch.optim.Optimizer`\n"
                " * [`torch.optim.Optimizer`]\n"
                " * ([`torch.optim.Optimizer`], [`torch.optim.lr_scheduler`])\n"
                ' * {"optimizer": `torch.optim.Optimizer`, (optional) "lr_scheduler": `torch.optim.lr_scheduler`}\n'
                ' * A list of the previously described dict format, with an optional "frequency" key (int)'
            )
        return optimizers, lr_schedulers, optimizer_frequencies, monitor

    @staticmethod
    def _configure_schedulers(schedulers, monitor, is_manual_optimization):
        """Convert each scheduler into dict structure with relevant information."""
        lr_schedulers = []
        default_config = {
            "scheduler": None,
            "name": None,  # no custom name
            "interval": "epoch",  # after epoch is over
            "frequency": 1,  # every epoch/batch
            "reduce_on_plateau": False,  # most often not ReduceLROnPlateau scheduler
            "monitor": None,  # value to monitor for ReduceLROnPlateau
            "strict": True,  # enforce that the monitor exists for ReduceLROnPlateau
            "opt_idx": None,  # necessary to store opt_idx when optimizer frequencies are specified
        }
        for scheduler in schedulers:
            if is_manual_optimization:
                if isinstance(scheduler, dict):
                    invalid_keys = {"interval", "frequency", "reduce_on_plateau", "monitor", "strict"}
                    keys_to_warn = [k for k in scheduler.keys() if k in invalid_keys]

                    if keys_to_warn:
                        warn(
                            f"The lr scheduler dict contains the key(s) {keys_to_warn}, but the keys will be ignored."
                            " You need to call `lr_scheduler.step()` manually in manual optimization.",
                            RuntimeWarning,
                        )

                    scheduler = {key: scheduler[key] for key in scheduler if key not in invalid_keys}
                    lr_schedulers.append({**default_config, **scheduler})
                else:
                    lr_schedulers.append({**default_config, "scheduler": scheduler})
            else:
                if isinstance(scheduler, dict):
                    # check provided keys
                    extra_keys = [k for k in scheduler.keys() if k not in default_config.keys()]
                    if extra_keys:
                        warn(f"Found unsupported keys in the lr scheduler dict: {extra_keys}", RuntimeWarning)
                    if "scheduler" not in scheduler:
                        raise Exception(
                            'The lr scheduler dict must have the key "scheduler" with its item being an lr scheduler'
                        )
                    if "interval" in scheduler and scheduler["interval"] not in ("step", "epoch"):
                        raise Exception(
                            'The "interval" key in lr scheduler dict must be "step" or "epoch"'
                            f' but is "{scheduler["interval"]}"'
                        )
                    scheduler["reduce_on_plateau"] = isinstance(
                        scheduler["scheduler"], optim.lr_scheduler.ReduceLROnPlateau
                    )
                    if scheduler["reduce_on_plateau"] and scheduler.get("monitor", None) is None:
                        raise Exception(
                            "The lr scheduler dict must include a monitor when a `ReduceLROnPlateau` scheduler is used."
                            ' For example: {"optimizer": optimizer, "lr_scheduler":'
                            ' {"scheduler": scheduler, "monitor": "your_loss"}}'
                        )
                    is_one_cycle = isinstance(scheduler["scheduler"], optim.lr_scheduler.OneCycleLR)
                    if is_one_cycle and scheduler.get("interval", "epoch") == "epoch":
                        warn(
                            "A `OneCycleLR` scheduler is using 'interval': 'epoch'."
                            " Are you sure you didn't mean 'interval': 'step'?",
                            RuntimeWarning,
                        )
                    lr_schedulers.append({**default_config, **scheduler})
                elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if monitor is None:
                        raise Exception(
                            "`configure_optimizers` must include a monitor when a `ReduceLROnPlateau`"
                            " scheduler is used. For example:"
                            ' {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}'
                        )
                    lr_schedulers.append(
                        {**default_config, "scheduler": scheduler, "reduce_on_plateau": True, "monitor": monitor}
                    )
                elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                    lr_schedulers.append({**default_config, "scheduler": scheduler})
                else:
                    raise ValueError(f'The provided lr scheduler "{scheduler}" is invalid')
        return lr_schedulers

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
            self._configure_optimizers(self.model.configure_optimizers())
        lr_schedulers = self._configure_schedulers(lr_schedulers, monitor, not self.automatic_optimization)
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
