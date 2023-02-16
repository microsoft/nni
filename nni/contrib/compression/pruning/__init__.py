# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .basic_pruner import LevelPruner, L1NormPruner, L2NormPruner
from .movement_pruner import MovementPruner
from .scheduled_pruner import LinearPruner, AGPPruner
from .slim_pruner import SlimPruner
from .taylor_pruner import TaylorPruner
