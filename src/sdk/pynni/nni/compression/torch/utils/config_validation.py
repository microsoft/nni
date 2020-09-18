# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from schema import Schema, And, SchemaError

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
    if not ('op_types' in data or 'op_names' in data):
        raise SchemaError('Either op_types or op_names must be specified.')
    return True

class CompressorSchema:
    def __init__(self, data_schema, model, logger):
        assert isinstance(data_schema, list) and len(data_schema) <= 1
        self.data_schema = data_schema
        self.compressor_schema = Schema(self._modify_schema(data_schema, model, logger))

    def _modify_schema(self, data_schema, model, logger):
        if not data_schema:
            return data_schema

        for k in data_schema[0]:
            old_schema = data_schema[0][k]
            if k == 'op_types' or (isinstance(k, Schema) and k._schema == 'op_types'):
                new_schema = And(old_schema, lambda n: validate_op_types(model, n, logger))
                data_schema[0][k] = new_schema
            if k == 'op_names' or (isinstance(k, Schema) and k._schema == 'op_names'):
                new_schema = And(old_schema, lambda n: validate_op_names(model, n, logger))
                data_schema[0][k] = new_schema

        data_schema[0] = And(data_schema[0], lambda d: validate_op_types_op_names(d))

        return data_schema

    def validate(self, data):
        self.compressor_schema.validate(data)
