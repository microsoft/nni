# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from ..pruning import LevelPrunerMasker, SlimPrunerMasker, L1FilterPrunerMasker, \
    L2FilterPrunerMasker, FPGMPrunerMasker, TaylorFOWeightFilterPrunerMasker, \
    ActivationAPoZRankFilterPrunerMasker, ActivationMeanRankFilterPrunerMasker, \
    L1ConstrainedFilterPrunerMasker, L2ConstrainedFilterPrunerMasker, \
    ConstrainedActivationMeanRankFilterPrunerMasker

MASKER_DICT = {
    'level': LevelPrunerMasker,
    'slim': SlimPrunerMasker,
    'l1': L1FilterPrunerMasker,
    'l1_constrained': L1ConstrainedFilterPrunerMasker,
    'l2': L2FilterPrunerMasker,
    'l2_constrained': L2ConstrainedFilterPrunerMasker,
    'fpgm': FPGMPrunerMasker,
    'taylorfo': TaylorFOWeightFilterPrunerMasker,
    'apoz': ActivationAPoZRankFilterPrunerMasker,
    'mean_activation': ActivationMeanRankFilterPrunerMasker,
    'mean_activation_constrained': ConstrainedActivationMeanRankFilterPrunerMasker,
    'attention': ConstrainedAttentionPrunerMasker
}
