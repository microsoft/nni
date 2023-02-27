# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.common.framework import shortcut_framework

from .profiler import Profiler, ExpressionProfiler

shortcut_framework(__name__)

del shortcut_framework
