"""Trigger import of some modules to write some caches,
so that static analysis (e.g., pyright) can know the type."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import nni
import nni.nas.nn.pytorch
