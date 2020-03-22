# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def validate_op_names(model, op_names, logger):
    found_names = set()
    for name, _ in model.named_modules():
        found_names.add(name)

    not_found_op_names = list(set(op_names) - found_names)
    if not_found_op_names:
        logger.warning('op_names %s not found in model', not_found_op_names)

    return True

def validate_op_types(model, op_types, logger):
    found_types = set(['default'])
    for _, module in model.named_modules():
        found_types.add(type(module).__name__)

    not_found_op_types = set(op_types) - found_types
    if not_found_op_types:
        logger.warning('op_types %s not found in model', not_found_op_types)

    return True
