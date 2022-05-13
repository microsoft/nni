# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

_multi_phase = False

def enable_multi_phase():
    global _multi_phase
    _multi_phase = True

def multi_phase_enabled():
    return _multi_phase
