# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Optional, Union

import torch
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

import nni


class BypassPlugin(TrainingTypePlugin):
    """ Plugin that handles communication on a single device. """

    def __init__(self, device: str):
        super().__init__()
        self.device: str = device
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    def connect(self, model: torch.nn.Module) -> torch.nn.Module:
        self._model = model
        self.model_to_device()
        return self.model

    @property
    def on_tpu(self) -> bool:
        return False

    @property
    def on_gpu(self) -> bool:
        return "cuda" in self.device and torch.cuda.is_available()

    def reduce(self, tensor: Union[Any, torch.Tensor], *args: Any, **kwargs: Any) -> Union[Any, torch.Tensor]:
        """
        Reduces a tensor from several distributed processes to one aggregated tensor.
        As this plugin only operates with a single device, the reduction is simply the identity.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation
        """
        return tensor

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes """
        return tensor

    def teardown(self):
        """
        This method is called to teardown the training process.
        It is the right place to release memory and free other resources.
        """
        pass

    @property
    def root_device(self) -> torch.device:
        return torch.device(self.device)

    def model_to_device(self) -> None:
        # bypass device placement from pytorch lightning
        pass

    def setup(self) -> None:
        pass

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args, **kwargs) -> None:
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj


def get_accelerator_connector(
        num_processes: int = 1,
        devices: Optional[Union[List[int], str, int]] = None,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        ipus: Optional[int] = None,
        distributed_backend: Optional[str] = None,
        accelerator: Optional[Union[str, Accelerator]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        num_nodes: int = 1,
        sync_batchnorm: bool = False,
        benchmark: bool = False,
        replace_sampler_ddp: bool = True,
        deterministic: bool = False,
        precision: int = 32,
        amp_backend: str = 'native',
        amp_level: Optional[str] = None,
        plugins: Optional[Union[List[Union[TrainingTypePlugin, ClusterEnvironment, str]],
                                TrainingTypePlugin, ClusterEnvironment, str]] = None,
        **other_trainier_kwargs) -> AcceleratorConnector:
    gpu_ids = Trainer()._parse_devices(gpus, auto_select_gpus, tpu_cores)
    return AcceleratorConnector(
        num_processes,
        devices,
        tpu_cores,
        ipus,
        distributed_backend,
        accelerator,
        gpus,
        gpu_ids,
        num_nodes,
        sync_batchnorm,
        benchmark,
        replace_sampler_ddp,
        deterministic,
        precision,
        amp_backend,
        amp_level,
        plugins,
    )


@nni.trace
class BypassAccelerator(Accelerator):
    def __init__(self, precision_plugin=None, device="cpu", **trainer_kwargs):
        if precision_plugin is None:
            precision_plugin = get_accelerator_connector(**trainer_kwargs).select_precision_plugin()

        # pylint: disable=abstract-class-instantiated
        super().__init__(precision_plugin=precision_plugin, training_type_plugin=BypassPlugin(device))
