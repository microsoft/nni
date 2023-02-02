# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.common.framework import shortcut_framework

from .evaluator import *
from .functional import FunctionalEvaluator

shortcut_framework(__name__)

del shortcut_framework
