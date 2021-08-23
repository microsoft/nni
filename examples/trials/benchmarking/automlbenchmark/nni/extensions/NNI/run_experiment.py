# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .architectures.run_random_forest import *
from .architectures.run_mlp import *


def run_experiment(dataset, config, tuner, log):
    if 'arch_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "arch_type" field.')
    
    if config.framework_params['arch_type'] == 'random_forest':
        return run_random_forest(dataset, config, tuner, log)

    elif config.framework_params['arch_type'] == 'mlp':
        return run_mlp(dataset, config, tuner, log)
    
    else:
        raise RuntimeError('The requested arch type in framework.yaml is unavailable.') 
