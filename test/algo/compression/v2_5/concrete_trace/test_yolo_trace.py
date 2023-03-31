# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from ..assets.registry import model_zoo
from ..assets.testing import trace_pipeline

yolo_models = model_zoo.get('yolo')

@pytest.mark.skip(reason='no download')
@pytest.mark.parametrize("mod_fn", yolo_models)
def test_nni_trace(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    mod = mod_fn()
    trace_pipeline(mod)
