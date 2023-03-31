# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from ...assets.registry import model_zoo
from ...assets.testing import speedup_pipeline

mmdet_models = model_zoo.get('mmdet')

@pytest.mark.skip(reason='cannot prune mmdet models')
@pytest.mark.parametrize('mod_fn', mmdet_models)
def test_l1_pruning_mmcv_models(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    mod = mod_fn()
    speedup_pipeline(mod)
