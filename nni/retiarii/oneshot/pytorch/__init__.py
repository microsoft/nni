# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .darts import DartsTrainer
from .enas import EnasTrainer
from .proxyless import ProxylessTrainer
from .random import SinglePathTrainer, RandomTrainer
from .utils import replace_input_choice, replace_layer_choice
from .differentiable import DartsModel, ProxylessModel
from .sampling import EnasModel,RandomSampleModel
from .base_lightning import MergeTrainValDataset