#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from . import pt_converter


class ModuleHook(object):
    """ Hook for nn.Module that will expire after running `life_count` times
    """

    def __init__(self, hook_func, life_count=-1):
        """ will never expire if life_count <= 0
        """
        self.hook_func = hook_func
        self.max_count = life_count
        self.handle = None
        self.count = 0

    def __del__(self):
        self.remove_hook()

    def register_forward_hook(self, m):
        assert not isinstance(
            m, torch.jit.ScriptModule
        ), f"Not supported type `torch.jit.ScriptModule` for {m}."
        assert not self.is_registered()
        self.handle = m.register_forward_hook(self)
        self.count = 0
        return self

    def remove_hook(self):
        if self.handle is None:
            return
        self.handle.remove()
        self.handle = None
        self.count = 0

    def is_registered(self):
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.remove_hook()
        return True

    def __call__(self, *args, **kwargs):
        assert self.handle is not None
        assert self.max_count <= 0 or self.count < self.max_count
        self.hook_func(*args, **kwargs)
        self.count += 1
        if self.max_count > 0 and self.count >= self.max_count:
            self.remove_hook()


class ModuleData(object):
    def __init__(self, data=None):
        self.data = data or {}

    def get(self, module, key_name=None):
        ret = self.data.get(id(module), None)
        if ret is not None and key_name is not None:
            ret = ret.get(key_name, None)
        return ret

    def set(self, module, info):
        self.data[id(module)] = info

    def __getitem__(self, key):
        """ Get a subset of ModuleData with values from `key` only
            Set to None if `key` does not exist
        """
        ret_data = {x: y.get(key, None) for x, y in self.data.items()}
        return ModuleData(ret_data)

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


class NestedModuleHook(object):
    """ Apply hooks to module and/or its children, store hook output in self.data
        The stored hook data will be returned if with is used
    """

    def __init__(self, hook_func, leaf_only=True, life_count=-1):
        self.hook_func = hook_func
        self.leaf_only = leaf_only
        self.life_count = life_count
        self._data = ModuleData()
        self._hooks = []
        self._callback = None
        self._callback_hook = None

    def __del__(self):
        self.remove_hook()

    def set_callback(self, callback):
        """ callback will be called after all hooks are called
              callback(module, module_data)
        """
        assert not self.is_registered(), "set_callback before registering"
        self._callback = callback
        return self

    def register_forward_hook(self, module):
        assert not self.is_registered()

        def _hook(m, *args, **kwargs):
            info = self.hook_func(m, *args, **kwargs)
            self._data.set(m, info)

        def _add_hook(m):
            # skip `torch.jit.ScriptModule` as it does not forward hook
            if isinstance(m, torch.jit.ScriptModule):
                return
            if self.leaf_only and len(list(m.children())) > 0:
                return
            self._hooks.append(
                ModuleHook(_hook, self.life_count).register_forward_hook(m)
            )

        module.apply(_add_hook)

        # register a hook that will be called after all other hooks are called
        if self._callback is not None:

            def _hook_adapter(m, input, output):
                return self._callback(m, self._data)

            self._hooks.append(
                ModuleHook(
                    _hook_adapter, self.life_count
                ).register_forward_hook(module)
            )

        return self

    def remove_hook(self):
        for x in self._hooks:
            x.remove_hook()
        self._hooks = []

    def is_registered(self):
        return len(self._hooks) > 0

    def __enter__(self):
        return self.data

    def __exit__(self, type, value, traceback):
        self.remove_hook()
        return True

    @property
    def data(self):
        return self._data


def _extract_shapes(data):
    if isinstance(data, torch.Tensor):
        return list(data.size())
    if isinstance(data, (list, tuple)):
        return [_extract_shapes(x) for x in data]
    if isinstance(data, dict):
        return {x: _extract_shapes(y) for x, y in data.items()}
    return None


def collect_op_shape(m, input, output):
    ret = {
        "input_shapes": _extract_shapes(input),
        "output_shapes": _extract_shapes(output),
    }
    return ret


def convert_to_lut_ops(model, input_shapes):
    input = torch.zeros(input_shapes)
    return convert_to_lut_ops_from_inputs(model, input)


def convert_to_lut_ops_from_inputs(model, inputs):
    with torch.no_grad():
        model.eval()
        with NestedModuleHook(
            collect_op_shape, leaf_only=True
        ).register_forward_hook(model) as model_data:
            model(inputs)
            ret = pt_converter.convert_all_modules(
                model, model_data["input_shapes"]
            )

    return ret
