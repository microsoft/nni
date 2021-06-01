# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..pruning import LevelPruner, SlimPruner, L1FilterPruner, L2FilterPruner, FPGMPruner, TaylorFOWeightFilterPruner, \
    ActivationAPoZRankFilterPruner, ActivationMeanRankFilterPruner
from ..quantization.quantizers import NaiveQuantizer, QAT_Quantizer, DoReFaQuantizer, BNNQuantizer


PRUNER_DICT = {
    'level': LevelPruner,
    'slim': SlimPruner,
    'l1': L1FilterPruner,
    'l2': L2FilterPruner,
    'fpgm': FPGMPruner,
    'taylorfo': TaylorFOWeightFilterPruner,
    'apoz': ActivationAPoZRankFilterPruner,
    'mean_activation': ActivationMeanRankFilterPruner
}

QUANTIZER_DICT = {
    'naive': NaiveQuantizer,
    'qat': QAT_Quantizer,
    'dorefa': DoReFaQuantizer,
    'bnn': BNNQuantizer
}
