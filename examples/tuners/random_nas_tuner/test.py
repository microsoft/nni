from random_nas_tuner import RandomNASTuner

tuner = RandomNASTuner()
tuner.update_search_space({'mnist/mutable_block_126': {'_type': 'mutable_layer',
                             '_value': {'mutable_layer_0': {'layer_choice': ['op.conv2d(size=1, '
                                                                             'in_ch=1, '
                                                                             'out_ch=self.channel_1_num)',
                                                                             'op.conv2d(size=3, '
                                                                             'in_ch=1, '
                                                                             'out_ch=self.channel_1_num)',
                                                                             'op.twice_conv2d(size=3, '
                                                                             'in_ch=1, '
                                                                             'out_ch=self.channel_1_num)',
                                                                             'op.twice_conv2d(size=7, '
                                                                             'in_ch=1, '
                                                                             'out_ch=self.channel_1_num)',
                                                                             'op.dilated_conv(in_ch=1, '
                                                                             'out_ch=self.channel_1_num)',
                                                                             'op.separable_conv(size=3, '
                                                                             'in_ch=1, '
                                                                             'out_ch=self.channel_1_num)',
                                                                             'op.separable_conv(size=5, '
                                                                             'in_ch=1, '
                                                                             'out_ch=self.channel_1_num)',
                                                                             'op.separable_conv(size=7, '
                                                                             'in_ch=1, '
                                                                             'out_ch=self.channel_1_num)'],
                                                            'optional_input_size': 0,
                                                            'optional_inputs': []},
                                        'mutable_layer_1': {'layer_choice': ['op.post_process(ch_size=self.channel_1_num)'],
                                                            'optional_input_size': 0,
                                                            'optional_inputs': []},
                                        'mutable_layer_2': {'layer_choice': ['op.max_pool(size=3)',
                                                                             'op.max_pool(size=5)',
                                                                             'op.max_pool(size=7)',
                                                                             'op.avg_pool(size=3)',
                                                                             'op.avg_pool(size=5)',
                                                                             'op.avg_pool(size=7)'],
                                                            'optional_input_size': 0,
                                                            'optional_inputs': []},
                                        'mutable_layer_3': {'layer_choice': ['op.conv2d(size=1, '
                                                                             'in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)',
                                                                             'op.conv2d(size=3, '
                                                                             'in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)',
                                                                             'op.twice_conv2d(size=3, '
                                                                             'in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)',
                                                                             'op.twice_conv2d(size=7, '
                                                                             'in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)',
                                                                             'op.dilated_conv(in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)',
                                                                             'op.separable_conv(size=3, '
                                                                             'in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)',
                                                                             'op.separable_conv(size=5, '
                                                                             'in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)',
                                                                             'op.separable_conv(size=7, '
                                                                             'in_ch=self.channel_1_num, '
                                                                             'out_ch=self.channel_2_num)'],
                                                            'optional_input_size': [0,
                                                                                    1],
                                                            'optional_inputs': ['post1_out']},
                                        'mutable_layer_4': {'layer_choice': ['op.post_process(ch_size=self.channel_2_num)'],
                                                            'optional_input_size': 0,
                                                            'optional_inputs': []},
                                        'mutable_layer_5': {'layer_choice': ['op.max_pool(size=3)',
                                                                             'op.max_pool(size=5)',
                                                                             'op.max_pool(size=7)',
                                                                             'op.avg_pool(size=3)',
                                                                             'op.avg_pool(size=5)',
                                                                             'op.avg_pool(size=7)'],
                                                            'optional_input_size': [0,
                                                                                    1],
                                                            'optional_inputs': ['post1_out',
                                                                                'pool1_out']}}}})

for i in range(10):
    print(tuner.generate_parameters(0))
