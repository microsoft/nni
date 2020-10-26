# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .one_shot import LevelPruner, L1FilterPruner, L2FilterPruner

PRUNER_DICT = {
    'level': LevelPruner,
    'l1': L1FilterPruner,
    'l2': L2FilterPruner
}
