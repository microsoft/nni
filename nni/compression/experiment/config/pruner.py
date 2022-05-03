from dataclasses import dataclass
from typing_extensions import Literal
from nni.experiment.config.base import ConfigBase


@dataclass
class PrunerConfig(ConfigBase):
    pass


@dataclass
class L1NormPrunerConfig(PrunerConfig):
    mode: Literal['normal', 'dependency_aware'] = 'dependency_aware'


@dataclass
class TaylorFOWeightPrunerConfig(PrunerConfig):
    pruner_type: Literal['TaylorFOWeightPruner'] = 'TaylorFOWeightPruner'
    mode: Literal['normal', 'dependency_aware', 'global'] = 'dependency_aware'
    training_batches: int = 30
