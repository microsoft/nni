import sys
import numpy

def tanh():
    pass

def Relu():
    pass

images=load_data()

"""@nni.architecture
{
    layer_1: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [images],
        input_num: 1,
        input_aggregate: None,
        outputs: layer_1_out,
    },

    layer_2: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [layer_1_out],
        input_num: 1,
        input_aggregate: None,
        outputs: layer_2_out,
    },

    layer_3: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [layer_1_out, layer_2_out],
        input_num: 1,
        input_aggregate: None,
        outputs: layer_3_out,
    }
}"""
final_output = layer_3_out
all = [layer_1_out, layer_2_out, layer_3_out]

if __name__ == 'main':
    func()