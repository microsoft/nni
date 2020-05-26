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
    found_types = set(['default']) | set(
        map(lambda x: type(x[1]).__name__, model.named_modules()))

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


valid_layer_names_resnet18 = [
    "layer1.0.conv1",
    "layer1.1.conv1",
    "layer2.0.conv1",
    "layer2.1.conv1",
    "layer3.0.conv1",
    "layer3.1.conv1",
    "layer4.0.conv1",
    "layer4.1.conv1"
]

valid_layer_names_mobilenet_v2 = [
    "features.0.0",
    "features.1.conv.0.0",
    "features.1.conv.1",
    "features.2.conv.0.0",
    "features.2.conv.1.0",
    "features.3.conv.0.0",
    "features.3.conv.1.0",
    "features.4.conv.0.0",
    "features.4.conv.1.0",
    "features.5.conv.0.0",
    "features.5.conv.1.0",
    "features.6.conv.0.0",
    "features.6.conv.1.0",
    "features.7.conv.0.0",
    "features.7.conv.1.0",
    "features.8.conv.0.0",
    "features.8.conv.1.0",
    "features.9.conv.0.0",
    "features.9.conv.1.0",
    "features.10.conv.0.0",
    "features.10.conv.1.0",
    "features.11.conv.0.0",
    "features.11.conv.1.0",
    "features.12.conv.0.0",
    "features.12.conv.1.0",
    "features.13.conv.0.0",
    "features.13.conv.1.0",
    "features.14.conv.0.0",
    "features.14.conv.1.0",
    "features.15.conv.0.0",
    "features.15.conv.1.0",
    "features.16.conv.0.0",
    "features.16.conv.1.0",
    "features.17.conv.0.0",
    "features.17.conv.1.0",
    "features.17.conv.2"]


valid_layer_names_retinaface = [
    "body.conv1",
    "body.layer1.0.conv1",
    "body.layer1.0.conv2",
    "body.layer1.1.conv1",
    "body.layer1.1.conv2",
    "body.layer1.2.conv1",
    "body.layer1.2.conv2",
    "body.layer2.0.conv1",
    "body.layer2.0.conv2",
    "body.layer2.1.conv1",
    "body.layer2.1.conv2",
    "body.layer2.2.conv1",
    "body.layer2.2.conv2",
    "body.layer2.3.conv1",
    "body.layer2.3.conv2",
    "body.layer3.0.conv1",
    "body.layer3.0.conv2",
    "body.layer3.1.conv1",
    "body.layer3.1.conv2",
    "body.layer3.2.conv1",
    "body.layer3.2.conv2",
    "body.layer3.3.conv1",
    "body.layer3.3.conv2",
    "body.layer3.4.conv1",
    "body.layer3.4.conv2",
    "body.layer3.5.conv1",
    "body.layer3.5.conv2",
    "body.layer4.0.conv1",
    "body.layer4.0.conv2",
    "body.layer4.1.conv1",
    "body.layer4.1.conv2",
    "body.layer4.2.conv1",
    "body.layer4.2.conv2",
    "fpn.merge1.0",
    "ssh1.conv3X3.0",
    "ssh1.conv5X5_1.0",
    "ssh1.conv5X5_2.0",
    "ssh1.conv7X7_2.0",
    "ssh1.conv7x7_3.0",
    "ssh2.conv3X3.0",
    "ssh2.conv5X5_1.0",
    "ssh2.conv5X5_2.0",
    "ssh2.conv7X7_2.0",
    "ssh2.conv7x7_3.0",
    "ssh3.conv3X3.0",
    "ssh3.conv5X5_1.0",
    "ssh3.conv5X5_2.0",
    "ssh3.conv7X7_2.0",
    "ssh3.conv7x7_3.0",
    "BboxHead.0.conv1x1",
    "BboxHead.1.conv1x1",
    "BboxHead.2.conv1x1",
    "ClassHead.0.conv1x1",
    "ClassHead.1.conv1x1",
    "ClassHead.2.conv1x1",
    "LandmarkHead.0.conv1x1",
    "LandmarkHead.1.conv1x1",
    "LandmarkHead.2.conv1x1"]


def get_layers_no_dependency(model):
    if model == 'ResNet':
        return valid_layer_names_resnet18
    elif model == 'MobileNetV2':
        return valid_layer_names_mobilenet_v2
    elif model == 'RetinaFace':
        return valid_layer_names_retinaface
