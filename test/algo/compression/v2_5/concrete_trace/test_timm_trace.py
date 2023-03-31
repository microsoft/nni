# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from ..assets.registry import model_zoo
from ..assets.testing import trace_pipeline

timm_models = model_zoo.get('timm')

@pytest.mark.parametrize("mod_fn", timm_models)
def test_nni_trace_timm(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    mod = mod_fn()
    trace_pipeline(mod)
