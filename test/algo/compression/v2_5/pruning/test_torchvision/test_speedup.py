# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from ...assets.registry import model_zoo
from ...assets.testing import speedup_pipeline

tv_basic_models = model_zoo.get('torchvision')
tv_detection_models = model_zoo.get('torchvision.detection')
tv_segmentation_models = model_zoo.get('torchvision.segmentation')
tv_video_models = model_zoo.get('torchvision.video')


@pytest.mark.parametrize('mod_fn', tv_basic_models)
def test_l1_pruning_tv_models(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    speedup_pipeline(mod_fn)

@pytest.mark.parametrize('mod_fn', tv_detection_models)
def test_l1_pruning_tv_detection_models(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    speedup_pipeline(mod_fn)

@pytest.mark.parametrize('mod_fn', tv_segmentation_models)
def test_l1_pruning_tv_segmentation_models(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    speedup_pipeline(mod_fn)

@pytest.mark.parametrize('mod_fn', tv_video_models)
def test_l1_pruning_tv_video_models(mod_fn):
    if mod_fn.skip_reason:
        pytest.skip(mod_fn.skip_reason)

    speedup_pipeline(mod_fn)
    