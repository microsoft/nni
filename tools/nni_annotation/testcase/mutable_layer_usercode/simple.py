import time

def add_one(inputs):
    return inputs + 1

def add_two(inputs):
    return inputs + 2

def add_three(inputs):
    return inputs + 3

def add_four(inputs):
    return inputs + 4


def main():

    images = 5

    """@nni.mutable_layers(
    {
        layer_choice: [add_one(), add_two(), add_three(), add_four()],
        optional_inputs: [images],
        optional_input_size: 1,
        layer_output: layer_1_out
    },
    {
        layer_choice: [add_one(), add_two(), add_three(), add_four()],
        optional_inputs: [layer_1_out],
        optional_input_size: 1,
        layer_output: layer_2_out
    },
    {
        layer_choice: [add_one(), add_two(), add_three(), add_four()],
        optional_inputs: [layer_1_out, layer_2_out],
        optional_input_size: 1,
        layer_output: layer_3_out
    }
    )"""

    """@nni.report_intermediate_result(layer_1_out)"""
    time.sleep(2)
    """@nni.report_intermediate_result(layer_2_out)"""
    time.sleep(2)
    """@nni.report_intermediate_result(layer_3_out)"""
    time.sleep(2)

    layer_3_out = layer_3_out + 10

    """@nni.report_final_result(layer_3_out)"""

if __name__ == '__main__':
    main()
