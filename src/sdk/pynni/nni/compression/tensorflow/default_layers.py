from tensorflow import keras

supported_layers = {
    keras.layers.Dense: ('Dense', 0),
    keras.layers.Conv1D: ('Conv1D', 0),
    keras.layers.Conv2D: ('Conv2D', 0),
}

default_layers = ['Dense', 'Conv2D']

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
