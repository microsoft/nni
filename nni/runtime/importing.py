# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import importlib
import sys


def _lazy_import(caller, imports):
    if sys.version_info < (3, 7):
        _lazy_import_36(caller, imports)
    else:
        _lazy_import_37(caller, imports)


def _shortcut(caller, module_list):
    if sys.version_info < (3, 7):
        _shortcut_36(caller, module_list)
    else:
        _shortcut_37(caller, module_list)


def _lazy_import_37(caller, imports):
    lookup_table = {}
    for defining_module_name, symbols in imports:
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols:
            lookup_table[symbol] = defining_module_name
    sys.modules[caller].__getattr__ = lambda symbol: _lazy_import_getattr(caller, lookup_table, symbol)


def _shortcut_37(caller, module_list):
    if isinstance(module_list, str):
        module_list = [module_list]
    sys.modules[caller].__getattr__ = lambda symbol: _shortcut_getattr(caller, module_list, symbol)


def _lazy_import_getattr(caller, lookup_table, symbol):
    if symbol not in lookup_table:
        raise AttributeError(f"Module '{caller}' has no attribute '{symbol}'")
    defining_module = _import_relative(caller, lookup_table[symbol])
    return getattr(defining_module, symbol)


def _shortcut_getattr(caller, lookup_list, symbol):
    for defining_caller in lookup_list:
        defining_module = _import_relative(caller, lookup_table[symbol])
        if hasattr(defining_module, symbol):
            return getattr(defining_module, symbol)
    raise AttributeError(f"Module '{caller}' has no attribute '{symbol}'")


def _lazy_import_36(caller, imports):
    caller_module = sys.modules[caller]
    for defining_module_name, symbols in imports:
        with contextlib.suppress(Exception):
            defining_module = _import_relative(caller, defining_module_name)
            if isinstance(symbols, str):
                symbols = [symbols]
            for symbol in symbols:
                setattr(caller_module, symbol, getattr(defining_module, symbol))


def _shortcut_36(caller, module_list):
    caller_module = sys.modules[caller]
    if isinstance(module_list, str):
        module_list = [module_list]
    for defining_module_name in module_list:
        with contextlib.suppress(Exception):
            defining_module = _import_relative(caller, defining_module_name)
            if hasattr(defining_module, '__all__'):
                symbols = defining_module.__all__
            else:
                symbols = [symbol for symbol in dir(defining_module) if not symbol.startswith('_')]
            for symbol in symbols:
                setattr(caller_module, symbol, getattr(defining_module, symbol))


def _import_relative(relative_to, module):
    assert not module.startswith('.')
    if not module.startswith('nni'):
        module = relative_to + '.' + module
    return importlib.import_module(module)
