from typing import Any, Union, Optional, List
import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

from pytorch_lightning.plugins import Plugin
from pytorch_lightning.plugins.environments import ClusterEnvironment

from ....serializer import serialize_cls


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

    @property
    def root_device(self) -> torch.device:
        return torch.device(self.device)

    def model_to_device(self) -> None:
        # bypass device placement from pytorch lightning
        pass

    def setup(self, model: torch.nn.Module) -> torch.nn.Module:
        self.model_to_device()
        return self.model

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args, **kwargs) -> None:
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj


def get_accelerator_connector(
        num_processes: int = 1,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        distributed_backend: Optional[str] = None,
        auto_select_gpus: bool = False,
        gpus: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        sync_batchnorm: bool = False,
        benchmark: bool = False,
        replace_sampler_ddp: bool = True,
        deterministic: bool = False,
        precision: int = 32,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        plugins: Optional[Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]] = None):
    return AcceleratorConnector(
        num_processes, tpu_cores, distributed_backend, auto_select_gpus, gpus, num_nodes, sync_batchnorm, benchmark,
        replace_sampler_ddp, deterministic, precision, amp_backend, amp_level, plugins
    )


@serialize_cls
class BypassAccelerator(Accelerator):
    def __init__(self, precision_plugin=None, device="cpu"):
        if precision_plugin is None:
            precision_plugin = get_accelerator_connector().precision_plugin
        # pylint: disable=abstract-class-instantiated
        super().__init__(precision_plugin=precision_plugin, training_type_plugin=BypassPlugin(device))
