# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from ..pruning import LevelPrunerMasker, SlimPrunerMasker, L1FilterPrunerMasker, L2FilterPrunerMasker, FPGMPrunerMasker

masker_dict = {
    'level': LevelPrunerMasker,
    'slim': SlimPrunerMasker,
    'l1': L1FilterPrunerMasker,
    'l2': L2FilterPrunerMasker,
    'fpgm': FPGMPrunerMasker
}
