import inspect
import logging
import torch.nn as nn


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_records = None

def enable_record_args():
    global _records
    _records = {}
    _logger.info('args recording enabled')

def disable_record_args():
    global _records
    _records = None
    _logger.info('args recording disabled')

def get_records():
    global _records
    return _records


class Placeholder(nn.Module):
    def __init__(self, label, related_info):
        global _records
        if _records is not None:
            _records[id(self)] = related_info
        self.label = label
        self.related_info = related_info
        super(Placeholder, self).__init__()

    def forward(self, x):
        return x


class Module(nn.Module):
    def __init__(self, *args, **kwargs):
        # TODO: users have to pass init's arguments to super init's arguments
        global _records
        if _records is not None:
            # TODO: change tuple to dict
            _records[id(self)] = (args, kwargs)
            #print('my module: ', id(self), args, kwargs)
        super(Module, self).__init__()

class Sequential(nn.Sequential):
    def __init__(self, *args):
        global _records
        if _records is not None:
            _records[id(self)] = {} # no args need to be recorded
        super(Sequential, self).__init__(*args)

def wrap_module(original_class):
    orig_init = original_class.__init__
    argname_list = list(inspect.signature(original_class).parameters.keys())
    # Make copy of original __init__, so we can call it without recursion

    def __init__(self, *args, **kws):
        global _records
        if _records is not None:
            full_args = {}
            full_args.update(kws)
            for i, arg in enumerate(args):
                full_args[argname_list[i]] = args[i]
            _records[id(self)] = full_args
        orig_init(self, *args, **kws) # Call the original __init__

    original_class.__init__ = __init__ # Set the class' __init__ to the new one
    return original_class

Conv2d = wrap_module(nn.Conv2d)
BatchNorm2d = wrap_module(nn.BatchNorm2d)
ReLU = wrap_module(nn.ReLU)
Dropout = wrap_module(nn.Dropout)
Linear = wrap_module(nn.Linear)
