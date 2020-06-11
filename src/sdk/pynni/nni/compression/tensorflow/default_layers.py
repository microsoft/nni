# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tensorflow import keras

supported_layers = {
    keras.layers.Conv1D: ('Conv1D', 0),
    keras.layers.Conv2D: ('Conv2D', 0),
    keras.layers.Conv2DTranspose: ('Conv2DTranspose', 0),
    keras.layers.Conv3D: ('Conv3D', 0),
    keras.layers.Conv3DTranspose: ('Conv3DTranspose', 0),
    keras.layers.ConvLSTM2D: ('ConvLSTM2D', 0),
    keras.layers.Dense: ('Dense', 0),
    keras.layers.Embedding: ('Embedding', 0),
    keras.layers.GRU: ('GRU', 0),
    keras.layers.LSTM: ('LSTM', 0),
}

default_layers = [x[0] for x in supported_layers.values()]

def get_op_type(layer_type):
    if layer_type in supported_layers:
        return supported_layers[layer_type][0]
    else:
        return None

def get_weight_index(op_type):
    for k in supported_layers:
        if supported_layers[k][0] == op_type:
            return supported_layers[k][1]
    return None
