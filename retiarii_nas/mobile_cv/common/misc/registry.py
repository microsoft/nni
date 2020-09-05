#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
      users' custom modules.

    To create a registry (inside detectron2):
        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:
        @BACKBONE_REGISTRY.register("MyBackbone")
        class MyBackbone():
            ...
    Or:
        BACKBONE_REGISTRY.register(name="MyBackbone", obj=MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, name=None, obj=None):
        """
        Register the given object under the the name or `obj.__name__` if name is None.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                nonlocal name
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def register_dict(self, mapping):
        """
        Register a dict of objects
        """
        assert isinstance(mapping, dict)
        [self.register(name, obj) for name, obj in mapping.items()]

    def get(self, name, is_raise=True):
        """
            Raise an exception if the key is not found if `is_raise` is True,
              return None otherwise
        """
        ret = self._obj_map.get(name)
        if ret is None and is_raise:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret

    def get_names(self):
        return self._obj_map.keys()

    def items(self):
        return self._obj_map.items()

    def __len__(self):
        return len(self._obj_map)

    def keys(self):
        return self._obj_map.keys()

    def __contains__(self, key):
        return key in self._obj_map

    def __getitem__(self, key):
        return self._obj_map[key]
