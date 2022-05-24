# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, asdict
from typing_extensions import Literal
from nni.experiment.config.base import ConfigBase


@dataclass
class PrunerConfig(ConfigBase):
    """
    Use to config the initialization parameters of a quantizer used in the compression experiment.
    """

    pruner_type: Literal['Pruner']

    def json(self):
        canon = self.canonical_copy()
        return asdict(canon)


@dataclass
class L1NormPrunerConfig(PrunerConfig):
    pruner_type: Literal['L1NormPruner'] = 'L1NormPruner'
    mode: Literal['normal', 'dependency_aware'] = 'dependency_aware'


@dataclass
class TaylorFOWeightPrunerConfig(PrunerConfig):
    pruner_type: Literal['TaylorFOWeightPruner'] = 'TaylorFOWeightPruner'
    mode: Literal['normal', 'dependency_aware', 'global'] = 'dependency_aware'
    training_batches: int = 30
