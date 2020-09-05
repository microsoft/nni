#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest.mock as mock
from contextlib import contextmanager

import torch

from .. import lut_helper
from . import pt_converter, utils


class FlopsEstimation(object):
    """ Compute the flops of a pytorch model.
        callback(module, module_data) will be called after the shapes/flops
          are computed
    """

    def __init__(self, model):
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self._hook = utils.NestedModuleHook(
            utils.collect_op_shape, leaf_only=False
        )
        self._patchers = []
        self._is_patched = False
        self._init_repr_shape()

    def __del__(self):
        self.set_enable(False)
        self.set_enable_repr_shape(False)

    def set_callback(self, callback):
        """ callback(self, module, module_data)
        """

        def _adapter(model, model_data):
            callback(self, model, model_data)

        self._hook.set_callback(_adapter)
        return self

    @property
    def is_enabled(self):
        return self._hook.is_registered()

    @property
    def cost(self):
        return self._hook.data

    def set_enable(self, is_enable):
        if self.is_enabled == is_enable:
            return
        if is_enable:
            self._hook.register_forward_hook(self.model)
        else:
            self._hook.remove_hook()
        self.set_enable_repr_shape(is_enable)

    @contextmanager
    def enable(self):
        self.set_enable(True)
        yield
        self.set_enable(False)

    def add_flops_info(self, unit=1e6):
        """ add flops info to hook_info after shape is available
        """
        add_flops_info(self.model, self._hook.data, unit)

    def get_lut_ops(self):
        assert self.is_enabled, "Call enable/set_enable and run model first"
        ret = pt_converter.convert_all_modules(
            self.model, self._hook.data["input_shapes"]
        )
        return ret

    def get_flops(self, unit=1e6):
        """ Returns: (num_params, num_flops)
        """
        ops = self.get_lut_ops()
        return lut_helper.compute_flops(ops, unit)

    def _init_repr_shape(self):
        """ Add patchers to modify nn.Module.extra_repr to add additional info
              when the model is printted and the patchers are enabled.
        """
        assert len(self._patchers) == 0

        REPR_ITEMS = ["input_shapes", "output_shapes", "nparams", "nflops"]

        def decor_extra_repr(orig_extra_repr):
            def new_extra_repr(module):
                info = self._hook.data(module)
                info_str = []
                if info is not None:
                    # input and output shapes
                    info_str = [
                        f"{k}={v}" for k, v in info.items() if k in REPR_ITEMS
                    ]

                ret = orig_extra_repr(module)
                info_str = ", ".join(info_str)
                if len(ret) > 0 and len(info_str) > 0:
                    ret = ret + ",\n"
                ret += info_str
                return ret

            return new_extra_repr

        def _get_unique_parents(types):
            """ Mocking a class method (patch.object) may fail in some cases when
                  handling derived classes. To avoid this issue, we need to:
                  1. For all subclasses that did not overwrite the class method,
                     we should only mock the base class method.
                  2. Mock the sublcass method if it is overwritten.
                This could be done by grouping all the classes by their methods
                  that will be mocked, and remove classes that are subclasses of
                  others.
                Inputs: [(Class, method), ...]
                Outputs: Filtered input
            """
            assert all(isinstance(x, tuple) for x in types)
            types_unique = {x[1]: [] for x in types}
            for cur_type, cur_method in types:
                types_unique[cur_method].append(cur_type)

            ret = []
            for method, cur_types in types_unique.items():
                filtered = get_unique_parent_types(cur_types)
                ret.extend([(ct, method) for ct in filtered])

            return ret

        # nn.Module subclass types that needs to be patched
        # {(module type, module extra_repr function)}
        all_types = set()
        self.model.apply(lambda m: all_types.add((type(m), type(m).extra_repr)))
        all_types = _get_unique_parents(all_types)

        self._patchers = [
            mock.patch.object(
                x, "extra_repr", side_effect=decor_extra_repr(mt), autospec=True
            )
            for x, mt in all_types
        ]

    def set_enable_repr_shape(self, is_enable):
        """ Enable or disable adding shapes/flops when str(model) is called
        """
        if self._is_patched == is_enable:
            return
        if is_enable:
            for x in self._patchers:
                x.start()
        else:
            for x in self._patchers:
                x.stop()
        self._is_patched = is_enable


def get_unique_parent_types(type_list):
    """ Given a list of types, remove types that are subclasses of others
    """
    ret = []
    for idx, x in enumerate(type_list):
        if issubclass(x, tuple(type_list[idx + 1 :])):
            continue
        if len(ret) > 0 and issubclass(x, tuple(ret)):
            continue
        ret.append(x)

    return ret


def add_flops_info(model, model_info, unit=1e6):
    def _set_flops(m):
        info = model_info(m)
        if info is None:
            return
        if "nparams" in info and "nflops" in info:
            return
        nparams, nflops = lut_helper.compute_flops(
            pt_converter.convert_all_modules(m, model_info["input_shapes"]),
            unit,
        )
        info["nparams"] = nparams
        info["nflops"] = nflops

    model.apply(_set_flops)


def print_model_flops(model, input):
    fest = FlopsEstimation(model)
    with fest.enable():
        output = model(input)
        fest.add_flops_info()
        nparams, nflops = fest.get_flops()
        print(model)
        print(f"nparams: {nparams}, nflops {nflops}")
    return output
