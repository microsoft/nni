# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import nni

if nni.get_default_framework() == 'pytorch':
    from .pytorch import model_to_pytorch_script
else:
    raise NotImplementedError('Unsupported framework: ' + nni.get_default_framework())

del nni
