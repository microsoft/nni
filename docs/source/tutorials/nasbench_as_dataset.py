"""
Use NAS Benchmarks as Datasets
==============================

In this tutorial, we show how to use NAS Benchmarks as datasets.
For research purposes we sometimes desire to query the benchmarks for architecture accuracies,
rather than train them one by one from scratch.
NNI has provided query tools so that users can easily get the retrieve the data in NAS benchmarks.
"""

# %%
# Prerequisites
# -------------
# This tutorial assumes that you have already prepared your NAS benchmarks under cache directory
# (by default, ``~/.cache/nni/nasbenchmark``).
# If you haven't, please follow the data preparation guide in :doc:`/nas/benchmarks`.
#
# As a result, the directory should look like:

import os
os.listdir(os.path.expanduser('~/.cache/nni/nasbenchmark'))

# %%
import pprint

from nni.nas.benchmark.nasbench101 import query_nb101_trial_stats
from nni.nas.benchmark.nasbench201 import query_nb201_trial_stats
from nni.nas.benchmark.nds import query_nds_trial_stats

# %%
# NAS-Bench-101
# -------------
#
# Use the following architecture as an example:
#
# .. image:: ../../img/nas-bench-101-example.png

arch = {
    'op1': 'conv3x3-bn-relu',
    'op2': 'maxpool3x3',
    'op3': 'conv3x3-bn-relu',
    'op4': 'conv3x3-bn-relu',
    'op5': 'conv1x1-bn-relu',
    'input1': [0],
    'input2': [1],
    'input3': [2],
    'input4': [0],
    'input5': [0, 3, 4],
    'input6': [2, 5]
}
for t in query_nb101_trial_stats(arch, 108, include_intermediates=True):
    pprint.pprint(t)

# %%
# An architecture of NAS-Bench-101 could be trained more than once.
# Each element of the returned generator is a dict which contains one of the training results of this trial config
# (architecture + hyper-parameters) including train/valid/test accuracy,
# training time, number of epochs, etc. The results of NAS-Bench-201 and NDS follow similar formats.
#
# NAS-Bench-201
# -------------
#
# Use the following architecture as an example:
#
# .. image:: ../../img/nas-bench-201-example.png

arch = {
    '0_1': 'avg_pool_3x3',
    '0_2': 'conv_1x1',
    '1_2': 'skip_connect',
    '0_3': 'conv_1x1',
    '1_3': 'skip_connect',
    '2_3': 'skip_connect'
}
for t in query_nb201_trial_stats(arch, 200, 'cifar100'):
    pprint.pprint(t)

# %%
# Intermediate results are also available.

for t in query_nb201_trial_stats(arch, None, 'imagenet16-120', include_intermediates=True):
    print(t['config'])
    print('Intermediates:', len(t['intermediates']))

# %%
# NDS
# ---
#
# Use the following architecture as an example:
#
# .. image:: ../../img/nas-bench-nds-example.png
#
# Here, ``bot_muls``, ``ds``, ``num_gs``, ``ss`` and ``ws`` stand for "bottleneck multipliers",
# "depths", "number of groups", "strides" and "widths" respectively.

# %%
model_spec = {
    'bot_muls': [0.0, 0.25, 0.25, 0.25],
    'ds': [1, 16, 1, 4],
    'num_gs': [1, 2, 1, 2],
    'ss': [1, 1, 2, 2],
    'ws': [16, 64, 128, 16]
}

# %%
# Use none as a wildcard.
for t in query_nds_trial_stats('residual_bottleneck', None, None, model_spec, None, 'cifar10'):
    pprint.pprint(t)

# %%
model_spec = {
    'bot_muls': [0.0, 0.25, 0.25, 0.25],
    'ds': [1, 16, 1, 4],
    'num_gs': [1, 2, 1, 2],
    'ss': [1, 1, 2, 2],
    'ws': [16, 64, 128, 16]
}
for t in query_nds_trial_stats('residual_bottleneck', None, None, model_spec, None, 'cifar10', include_intermediates=True):
    pprint.pprint(t['intermediates'][:10])

# %%
model_spec = {'ds': [1, 12, 12, 12], 'ss': [1, 1, 2, 2], 'ws': [16, 24, 24, 40]}
for t in query_nds_trial_stats('residual_basic', 'resnet', 'random', model_spec, {}, 'cifar10'):
    pprint.pprint(t)

# %%
# Get the first one.
pprint.pprint(next(query_nds_trial_stats('vanilla', None, None, None, None, None)))

# %%
# Count number.
model_spec = {'num_nodes_normal': 5, 'num_nodes_reduce': 5, 'depth': 12, 'width': 32, 'aux': False, 'drop_prob': 0.0}
cell_spec = {
    'normal_0_op_x': 'avg_pool_3x3',
    'normal_0_input_x': 0,
    'normal_0_op_y': 'conv_7x1_1x7',
    'normal_0_input_y': 1,
    'normal_1_op_x': 'sep_conv_3x3',
    'normal_1_input_x': 2,
    'normal_1_op_y': 'sep_conv_5x5',
    'normal_1_input_y': 0,
    'normal_2_op_x': 'dil_sep_conv_3x3',
    'normal_2_input_x': 2,
    'normal_2_op_y': 'dil_sep_conv_3x3',
    'normal_2_input_y': 2,
    'normal_3_op_x': 'skip_connect',
    'normal_3_input_x': 4,
    'normal_3_op_y': 'dil_sep_conv_3x3',
    'normal_3_input_y': 4,
    'normal_4_op_x': 'conv_7x1_1x7',
    'normal_4_input_x': 2,
    'normal_4_op_y': 'sep_conv_3x3',
    'normal_4_input_y': 4,
    'normal_concat': [3, 5, 6],
    'reduce_0_op_x': 'avg_pool_3x3',
    'reduce_0_input_x': 0,
    'reduce_0_op_y': 'dil_sep_conv_3x3',
    'reduce_0_input_y': 1,
    'reduce_1_op_x': 'sep_conv_3x3',
    'reduce_1_input_x': 0,
    'reduce_1_op_y': 'sep_conv_3x3',
    'reduce_1_input_y': 0,
    'reduce_2_op_x': 'skip_connect',
    'reduce_2_input_x': 2,
    'reduce_2_op_y': 'sep_conv_7x7',
    'reduce_2_input_y': 0,
    'reduce_3_op_x': 'conv_7x1_1x7',
    'reduce_3_input_x': 4,
    'reduce_3_op_y': 'skip_connect',
    'reduce_3_input_y': 4,
    'reduce_4_op_x': 'conv_7x1_1x7',
    'reduce_4_input_x': 0,
    'reduce_4_op_y': 'conv_7x1_1x7',
    'reduce_4_input_y': 5,
    'reduce_concat': [3, 6]
}

for t in query_nds_trial_stats('nas_cell', None, None, model_spec, cell_spec, 'cifar10'):
    assert t['config']['model_spec'] == model_spec
    assert t['config']['cell_spec'] == cell_spec
    pprint.pprint(t)

# %%
# Count number.
print('NDS (amoeba) count:', len(list(query_nds_trial_stats(None, 'amoeba', None, None, None, None, None))))
