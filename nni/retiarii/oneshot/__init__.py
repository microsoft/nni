# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.common.framework import shortcut_framework

from .interface import BaseOneShotTrainer

shortcut_framework(__name__)

del shortcut_framework
