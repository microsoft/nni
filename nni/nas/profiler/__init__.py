# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .profiler import Profiler, ExpressionProfiler

from nni.common.framework import shortcut_framework

shortcut_framework(__name__)

del shortcut_framework

