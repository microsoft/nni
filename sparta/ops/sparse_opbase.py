# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class SparseOPBase(torch.nn.Module):
    """
    The base class of the sparse OP.
    """
    def __init__(self):
        super(SparseOPBase, self).__init__()

    def specialize(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError
    
    def export_kernel(self, path):
        # export the current specialized kernel configuration
        raise NotImplementedError
    
    def load_kernel(self, path):
        # load the kernel configuration from path and rebind the kernel
        raise NotImplementedError
    
    def performance_optimization(self):
        # try to optimize the kernel performance
        raise NotImplementedError
