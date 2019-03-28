import time

def add_one(layer_name, out, inputs):
    return inputs + 1

def add_two(layer_name, out, inputs):
    return inputs + 2

def add_three(layer_name, out, inputs):
    return inputs + 3

def add_four(layer_name, out, inputs):
    return inputs + 4

def add_five(layer_name, out, inputs):
    return inputs + 5

def post_process(layer_name, out, inputs):
    return out




def main():

    images = 5

    """@nni.architecture
    {
        platform: others
        layer_1: {
            layer_choice: [add_one, add_two, add_three, add_four],
            input_candidates: [images],
            input_num: 1,
            input_aggregate: None,
            outputs: layer_1_out,
            post_process_outputs: post_process
        },

        layer_2: {
            layer_choice: [add_one, add_two, add_three, add_four],
            input_candidates: [layer_1_out],
            input_num: 1,
            input_aggregate: None,
            outputs: layer_2_out,
            post_process_outputs: post_process
        },

        layer_3: {
            layer_choice: [add_one, add_two, add_three, add_four],
            input_candidates: [layer_1_out, layer_2_out],
            input_num: 1,
            input_aggregate: None,
            outputs: layer_3_out,
            post_process_outputs: post_process
        }
    }"""

    """@nni.report_intermediate_result(layer_1_out)"""
    time.sleep(2)
    """@nni.report_intermediate_result(layer_2_out)"""
    time.sleep(2)
    """@nni.report_intermediate_result(layer_3_out)"""
    time.sleep(2)

    layer_3_out = layer_3_out + 10

    """@nni.report_final_result(layer_3_out)"""

if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        print(exception)
        raise
