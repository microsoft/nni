op_flops_dict = pickle.load(open('./data/op_flops_dict.pkl', 'rb'))
backbone_info = [  # inp, oup, img_h, img_w, stride
    (3, 16, 224, 224, 2),  # conv1
    (16, 64, 112, 112, 2),
    (64, 64, 56, 56, 1),
    (64, 64, 56, 56, 1),
    (64, 64, 56, 56, 1),
    (64, 160, 56, 56, 2),  # stride = 2
    (160, 160, 28, 28, 1),
    (160, 160, 28, 28, 1),
    (160, 160, 28, 28, 1),
    (160, 320, 28, 28, 2),  # stride = 2
    (320, 320, 14, 14, 1),
    (320, 320, 14, 14, 1),
    (320, 320, 14, 14, 1),
    (320, 320, 14, 14, 1),
    (320, 320, 14, 14, 1),
    (320, 320, 14, 14, 1),
    (320, 320, 14, 14, 1),
    (320, 640, 14, 14, 2),  # stride = 2
    (640, 640, 7, 7, 1),
    (640, 640, 7, 7, 1),
    (640, 640, 7, 7, 1),
    (640, 1000, 7, 7, 1),  # rest_operation
]
blocks_keys = [
    'shufflenet_3x3',
    'shufflenet_5x5',
    'shufflenet_7x7',
    'xception_3x3',
]


def get_cand_flops(cand):
    conv1_flops = op_flops_dict['conv1'][(3, 16, 224, 224, 2)]
    rest_flops = op_flops_dict['rest_operation'][(640, 1000, 7, 7, 1)]
    total_flops = conv1_flops + rest_flops
    for i in range(len(cand)):
        op_ids = cand[i]
        inp, oup, img_h, img_w, stride = backbone_info[i + 1]
        key = blocks_keys[op_ids] + '_stride_' + str(stride)
        mid = int(oup // 2)
        mid = int(mid)
        total_flops += op_flops_dict[key][
            (inp, oup, mid, img_h, img_w, stride)]
    return total_flops


def main():
    for i in range(4):
        print(i, get_cand_flops((i,) * 20))


if __name__ == '__main__':
    main()
