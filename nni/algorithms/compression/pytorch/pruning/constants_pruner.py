# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .one_shot_pruner import LevelPruner, L1FilterPruner, L2FilterPruner, FPGMPruner

PRUNER_DICT = {
    'level': LevelPruner,
    'l1': L1FilterPruner,
    'l2': L2FilterPruner,
    'fpgm': FPGMPruner
}
