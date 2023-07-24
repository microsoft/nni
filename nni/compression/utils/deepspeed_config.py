# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import base64
import io
import json
import os
from copy import deepcopy


## This config is copied from accelerate == 0.20.3
class HfDeepSpeedConfig:
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A `weakref` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
    it's important that this object remains alive while the program is still running.

    [`Trainer`] uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
    with values of [`TrainingArguments`] by replacing special placeholder values: `"auto"`. Without this special logic
    the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """

    def __init__(self, config_file_or_dict):
        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            config = deepcopy(config_file_or_dict)
        elif os.path.exists(config_file_or_dict):
            with io.open(config_file_or_dict, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            try:
                config_decoded = base64.urlsafe_b64decode(config_file_or_dict).decode("utf-8")
                config = json.loads(config_decoded)
            except (UnicodeDecodeError, AttributeError, ValueError):
                raise ValueError(
                    "Expected a string path to an existing deepspeed config, or a dictionary, or a base64 encoded string. " +
                    f"Received: {config_file_or_dict}"
                )

        self.config = config

        self.set_stage_and_offload()

    def set_stage_and_offload(self):
        # zero stage - this is done as early as possible, before model is created, to allow
        # ``is_deepspeed_zero3_enabled`` query and getting to the early deepspeed config object
        # during ``zero.Init()`` which needs to know the dtype, and some other hparams.
        self._stage = self.get_value("zero_optimization.stage", -1)

        # offload
        self._offload = False
        if self.is_zero2() or self.is_zero3():
            offload_devices_valid = set(["cpu", "nvme"])
            offload_devices = set(
                [
                    self.get_value("zero_optimization.offload_optimizer.device"),
                    self.get_value("zero_optimization.offload_param.device"),
                ]
            )
            if len(offload_devices & offload_devices_valid) > 0:
                self._offload = True

    def find_config_node(self, ds_key_long):
        config = self.config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key

        return config, ds_key

    def get_value(self, ds_key_long, default=None):
        """
        Returns the set value or `default` if no value is set
        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return default
        return config.get(ds_key, default)

    def del_config_sub_tree(self, ds_key_long, must_exist=False):
        """
        Deletes a sub-section of the config file if it's found.

        Unless `must_exist` is `True` the section doesn't have to exist.
        """
        config = self.config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        for node in nodes:
            parent_config = config
            config = config.get(node)
            if config is None:
                if must_exist:
                    raise ValueError(f"Can't find {ds_key_long} entry in the config: {self.config}")
                else:
                    return

        # if found remove it
        if parent_config is not None:
            parent_config.pop(node)

    def is_true(self, ds_key_long):
        """
        Returns `True`/``False` only if the value is set, always `False` otherwise. So use this method to ask the very
        specific question of whether the value is set to `True` (and it's not set to `False`` or isn't set).

        """
        value = self.get_value(ds_key_long)
        return False if value is None else bool(value)

    def is_false(self, ds_key_long):
        """
        Returns `True`/``False` only if the value is set, always `False` otherwise. So use this method to ask the very
        specific question of whether the value is set to `False` (and it's not set to `True`` or isn't set).
        """
        value = self.get_value(ds_key_long)
        return False if value is None else not bool(value)

    def is_zero2(self):
        return self._stage == 2

    def is_zero3(self):
        return self._stage == 3

    def is_offload(self):
        return self._offload