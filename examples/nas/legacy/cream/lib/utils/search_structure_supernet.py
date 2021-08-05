# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

def search_for_layer(flops_op_dict, arch_def, flops_minimum, flops_maximum):
    sta_num = [1, 1, 1, 1, 1]
    order = [2, 3, 4, 1, 0, 2, 3, 4, 1, 0]
    limits = [3, 3, 3, 2, 2, 4, 4, 4, 4, 4]
    size_factor = 224 // 32
    base_min_flops = sum([flops_op_dict[i][0][0] for i in range(5)])
    base_max_flops = sum([flops_op_dict[i][5][0] for i in range(5)])

    if base_min_flops > flops_maximum:
        while base_min_flops > flops_maximum and size_factor >= 2:
            size_factor = size_factor - 1
            flops_minimum = flops_minimum * (7. / size_factor)
            flops_maximum = flops_maximum * (7. / size_factor)
        if size_factor < 2:
            return None, None, None
    elif base_max_flops < flops_minimum:
        cur_ptr = 0
        while base_max_flops < flops_minimum and cur_ptr <= 9:
            if sta_num[order[cur_ptr]] >= limits[cur_ptr]:
                cur_ptr += 1
                continue
            base_max_flops = base_max_flops + \
                flops_op_dict[order[cur_ptr]][5][1]
            sta_num[order[cur_ptr]] += 1
        if cur_ptr > 7 and base_max_flops < flops_minimum:
            return None, None, None

    cur_ptr = 0
    while cur_ptr <= 9:
        if sta_num[order[cur_ptr]] >= limits[cur_ptr]:
            cur_ptr += 1
            continue
        base_max_flops = base_max_flops + flops_op_dict[order[cur_ptr]][5][1]
        if base_max_flops <= flops_maximum:
            sta_num[order[cur_ptr]] += 1
        else:
            break

    arch_def = [item[:i] for i, item in zip([1] + sta_num + [1], arch_def)]
    # print(arch_def)

    return sta_num, arch_def, size_factor * 32
