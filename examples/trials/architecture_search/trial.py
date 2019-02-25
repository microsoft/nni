import time

def add_one(x):
    return x+1

def add_two(x):
    return x+2

def add_three(x):
    return x+3

def add_four(x):
    return x+4

def add_five(x):
    return x+5

def main():

    images = 5

    """@nni.architecture
    {
        layer_1: {
            layer_choice: [add_one, add_two, add_three, add_four],
            input_candidates: [images],
            input_num: 1,
            input_aggregate: None,
            outputs: layer_1_out,
        },

        layer_2: {
            layer_choice: [add_one, add_two, add_three, add_four],
            input_candidates: [layer_1_out],
            input_num: 1,
            input_aggregate: None,
            outputs: layer_2_out,
        },

        layer_3: {
            layer_choice: [add_one, add_two, add_three, add_four],
            input_candidates: [layer_1_out, layer_2_out],
            input_num: 1,
            input_aggregate: None,
            outputs: layer_3_out,
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
