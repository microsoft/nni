# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entrypoint for trials.
"""
import os

from .auto_compress_engine import AutoCompressEngine

if __name__ == '__main__':
    AutoCompressEngine.trial_execute_compress()
