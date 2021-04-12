# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

# This dictionary is generated from calculating each operation of each layer to quickly search for layers.
# flops_op_dict[which_stage][which_operation] =
# (flops_of_operation_with_stride1, flops_of_operation_with_stride2)

flops_op_dict = {}
for i in range(5):
    flops_op_dict[i] = {}
flops_op_dict[0][0] = (21.828704, 18.820752)
flops_op_dict[0][1] = (32.669328, 28.16048)
flops_op_dict[0][2] = (25.039968, 23.637648)
flops_op_dict[0][3] = (37.486224, 35.385824)
flops_op_dict[0][4] = (29.856864, 30.862992)
flops_op_dict[0][5] = (44.711568, 46.22384)
flops_op_dict[1][0] = (11.808656, 11.86712)
flops_op_dict[1][1] = (17.68624, 17.780848)
flops_op_dict[1][2] = (13.01288, 13.87416)
flops_op_dict[1][3] = (19.492576, 20.791408)
flops_op_dict[1][4] = (14.819216, 16.88472)
flops_op_dict[1][5] = (22.20208, 25.307248)
flops_op_dict[2][0] = (8.198, 10.99632)
flops_op_dict[2][1] = (12.292848, 16.5172)
flops_op_dict[2][2] = (8.69976, 11.99984)
flops_op_dict[2][3] = (13.045488, 18.02248)
flops_op_dict[2][4] = (9.4524, 13.50512)
flops_op_dict[2][5] = (14.174448, 20.2804)
flops_op_dict[3][0] = (12.006112, 15.61632)
flops_op_dict[3][1] = (18.028752, 23.46096)
flops_op_dict[3][2] = (13.009632, 16.820544)
flops_op_dict[3][3] = (19.534032, 25.267296)
flops_op_dict[3][4] = (14.514912, 18.62688)
flops_op_dict[3][5] = (21.791952, 27.9768)
flops_op_dict[4][0] = (11.307456, 15.292416)
flops_op_dict[4][1] = (17.007072, 23.1504)
flops_op_dict[4][2] = (11.608512, 15.894528)
flops_op_dict[4][3] = (17.458656, 24.053568)
flops_op_dict[4][4] = (12.060096, 16.797696)
flops_op_dict[4][5] = (18.136032, 25.40832)