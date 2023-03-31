# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from ..assets.registry import model_zoo
from ..assets.testing import trace_pipeline

mmcv_models = model_zoo.get('mmdet')

@pytest.mark.parametrize("mod_fn", mmcv_models)
def test_nni_trace_mmcv(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    mod = mod_fn()
    trace_pipeline(mod)
 