# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from logging import Logger
from typing import Dict, List
from schema import Schema, And, SchemaError

from torch.nn import Module


class CompressorSchema:
    def __init__(self, data_schema: List[Dict], model: Module, logger: Logger):
        assert isinstance(data_schema, list)
        self.data_schema = data_schema
        self.compressor_schema = Schema(self._modify_schema(data_schema, model, logger))

    def _modify_schema(self, data_schema: List[Dict], model: Module, logger: Logger) -> List[Dict]:
        if not data_schema:
            return data_schema

        for i, sub_schema in enumerate(data_schema):
            for k, old_schema in sub_schema.items():
                if k == 'op_types' or (isinstance(k, Schema) and k._schema == 'op_types'):
                    new_schema = And(old_schema, lambda n: validate_op_types(model, n, logger))
                    sub_schema[k] = new_schema
                if k == 'op_names' or (isinstance(k, Schema) and k._schema == 'op_names'):
                    new_schema = And(old_schema, lambda n: validate_op_names(model, n, logger))
                    sub_schema[k] = new_schema

            data_schema[i] = And(sub_schema, lambda d: validate_op_types_op_names(d))

        return data_schema

    def validate(self, data):
        self.compressor_schema.validate(data)


def validate_op_names(model, op_names, logger):
    found_names = set(map(lambda x: x[0], model.named_modules()))

    not_found_op_names = list(set(op_names) - found_names)
    if not_found_op_names:
        logger.warning('op_names %s not found in model', not_found_op_names)

    return True


def validate_op_types(model, op_types, logger):
    found_types = set(['default']) | set(map(lambda x: type(x[1]).__name__, model.named_modules()))

    not_found_op_types = list(set(op_types) - found_types)
    if not_found_op_types:
        logger.warning('op_types %s not found in model', not_found_op_types)

    return True


def validate_op_types_op_names(data):
    if not ('op_types' in data or 'op_names' in data or 'op_partial_names' in data):
        raise SchemaError('At least one of the followings must be specified: op_types, op_names or op_partial_names.')
    return True
