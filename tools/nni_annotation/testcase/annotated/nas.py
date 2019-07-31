import nni
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
    layer_1_out = nni.mutable_layer('mutable_block_39', 'mutable_layer_0',
        {'add_one()': add_one, 'add_two()': add_two, 'add_three()':
        add_three, 'add_four()': add_four}, {'add_one()': {}, 'add_two()':
        {}, 'add_three()': {}, 'add_four()': {}}, [], {'images': images}, 1,
        'classic_mode')
    layer_2_out = nni.mutable_layer('mutable_block_39', 'mutable_layer_1',
        {'add_one()': add_one, 'add_two()': add_two, 'add_three()':
        add_three, 'add_four()': add_four}, {'add_one()': {}, 'add_two()':
        {}, 'add_three()': {}, 'add_four()': {}}, [], {'layer_1_out':
        layer_1_out}, 1, 'classic_mode')
    layer_3_out = nni.mutable_layer('mutable_block_39', 'mutable_layer_2',
        {'add_one()': add_one, 'add_two()': add_two, 'add_three()':
        add_three, 'add_four()': add_four}, {'add_one()': {}, 'add_two()':
        {}, 'add_three()': {}, 'add_four()': {}}, [], {'layer_1_out':
        layer_1_out, 'layer_2_out': layer_2_out}, 1, 'classic_mode')
    nni.report_intermediate_result(layer_1_out)
    time.sleep(2)
    nni.report_intermediate_result(layer_2_out)
    time.sleep(2)
    nni.report_intermediate_result(layer_3_out)
    time.sleep(2)
    layer_3_out = layer_3_out + 10
    nni.report_final_result(layer_3_out)


if __name__ == '__main__':
    main()
