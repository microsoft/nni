# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from ..assets.registry import model_zoo
from ..assets.testing import trace_pipeline

tv_basic_models = model_zoo.get('torchvision')
tv_detection_models = model_zoo.get('torchvision.detection')
tv_segmentation_models = model_zoo.get('torchvision.segmentation')
tv_video_models = model_zoo.get('torchvision.video')

all_models = tv_basic_models + tv_detection_models + tv_segmentation_models + tv_video_models

@pytest.mark.parametrize("mod_fn", all_models)
def test_nni_trace_torchvision(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    mod = mod_fn()
    trace_pipeline(mod)
