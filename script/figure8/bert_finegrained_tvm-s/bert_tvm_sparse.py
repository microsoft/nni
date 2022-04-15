# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# tvm, relay
import tvm
from tvm import relay
from tvm import autotvm
import scipy.sparse as sp
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import itertools
import numpy as np
import os.path
import time
from tvm.contrib import graph_runtime
import logging
import onnx
import argparse
from tvm.relay import data_dep_optimization as ddo

def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype="float32"):
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.uniform(-0.1, 0.1, (BS_R, BS_C))
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.data.size >= nnz
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s.todense()

def random_sparse_bert_params(func, params, BS_R, BS_C, density):
    def deepcopy(param_dic):
        ret = {}
        for k, v in param_dic.items():
            ret[k] = tvm.nd.array(v.asnumpy())
        return ret

    new_params = deepcopy(params)
    dense_weight_names = relay.analysis.sparse_dense._search_dense_op_weight(func)
    for item in dense_weight_names:
        name = str(item)
        shape = new_params[name].shape
        print(name)
        print(f"shape:{shape}")
        print(np.sum(new_params[name].numpy()))
        if np.sum(new_params[name].numpy()) == 0:
            print("find one empty weights")
            weights = new_params[name].numpy()
            shape_len = len(new_params[name].shape)
            if shape_len == 1:
                weights[0] = 1
            if shape_len == 2:
                weights[0][0] = 1
            if shape_len == 3:
                weights[0][0][0] = 1
            if shape_len == 4:
                weights[0][0][0][0] = 1
            new_params[name] = tvm.nd.array(weights)


        #if shape[0] % BS_R == 0 and shape[1] % BS_C == 0:
        #    new_w = random_bsr_matrix(shape[0], shape[1], BS_R, BS_C, density)
        #    new_params[name] = tvm.nd.array(new_w)
    return new_params

parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str, default='/data/znx/nnfusion/models/pytorch2onnx/bert_coarse_sota_no_propagation_folded.onnx', help='The file name of the frozen graph.')
#parser.add_argument('--model_path', type=str, default='/data/znx/SpargenCks/bert_coarse_sota_baseline/bert_coarse_sota_kernel.onnx', help='The file name of the frozen graph.')
parser.add_argument('--model_path', type=str, default='../../checkpoints/bert/artifact_bert_finegrained_no_propagation_onnx_with_tesa/model_tesa.onnx', help='The file name of the frozen graph.')
parser.add_argument('--warmup', type=int, default=5, help='The number of warmup iterations.')
parser.add_argument('--num_iter', type=int, default=100, help='The number of execution iterations.')
# parser.add_argument('--autotvm_log', type=str, default='', help='autotvm kernel tuning log')
parser.add_argument('--tuning_step', type=int, default=5000, help='tuning steps')
args = parser.parse_args()

# Target settings
# Use these commented settings to build for cuda.
target = 'cuda'
target_host = 'llvm'
layout = "NCHW"
ctx = tvm.gpu(0)
# target = 'llvm'
# target_host = 'llvm'
# layout = None
# ctx = tvm.cpu(0)


######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

model_path = args.model_path

onnx_model = onnx.load(args.model_path)
print(onnx_model.graph.input)
#print(onnx_model.graph.output)

# exit(0)

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
# shape_dict = {'a_input': (FLAGS.batch_size, 256), 'b_input': (FLAGS.batch_size, 3008)}
# shape_dict = {'input.1': (1280, 5), 'attention_mask': (1280, 5), 'input.2': (1280, 5)}
shape_dict = {'input.1': (32, 128), 'attention_mask': (32, 128), 'input.3': (32, 128)}
# dummy_input = (torch.rand(32, 128).numpy(), torch.rand(32, 128).numpy(), torch.rand(32, 128).numpy())
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

print("ONNX imported to relay frontend.")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.


mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)

sparsity = 0.95

params = random_sparse_bert_params(mod, params, BS_R=32, BS_C=32, density=1 - sparsity)
mod, params = ddo.bsr_dense.convert(mod, params, (1, 1), sparsity_threshold=1)
print("Block Sparse Model with {blocksize}x32 blocks:".format(blocksize=32))

log_file = "autotvm_tuned_fine_bert_5000.log"
# log_file = f'autotvm_tuned_fine_bert_{args.tuning_step}.log'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': args.tuning_step,
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=None),
        runner=autotvm.LocalRunner(number=10, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    # os.remove(tmp_log_file)


tasks = autotvm.task.extract_from_program(mod, target=target,
                                        params=params,
                                        ops=(relay.op.get("nn.dense"),relay.op.get("nn.sparse_dense")))




print("Tuning...")
# tune_tasks(tasks, **tuning_option)


print("Compile...")
with autotvm.apply_history_best(log_file):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params)


"""
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                    target=target,
                                    target_host=target_host,
                                    params=params)
"""

m = graph_runtime.create(graph, lib, ctx)
input1_shape = (32, 128)
x1 = np.random.uniform(size=input1_shape)
x2 = np.random.uniform(size=input1_shape)
x3 = np.random.uniform(size=input1_shape)

data_tvm1 = tvm.nd.array(x1.astype('int64'))
data_tvm2 = tvm.nd.array(x2.astype('int64'))
data_tvm3 = tvm.nd.array(x3.astype('int64'))

m.set_input("input.1", data_tvm1)
m.set_input("attention_mask", data_tvm2)
m.set_input("input.3", data_tvm3)

print("Begin to evaluate")

m.set_input(**{k:tvm.nd.array(v, ctx) for k, v in params.items()})
m.run()
e = m.module.time_evaluator("run", ctx, number=50, repeat=1)
t = e((data_tvm1, data_tvm2, data_tvm3)).results
t = np.array(t) * 1000
print('{} (batch={}): {} ms'.format("mlp", 1024, t.mean()))
{'input.1': (32, 128), 'attention_mask': (32, 128), 'input.3': (32, 128)}
######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

# from tvm.contrib import graph_runtime
# dtype = 'float32'
# m = graph_runtime.create(graph, lib, ctx)
# # set inputs
# x1 = np.random.rand(1024, 1, 32, 32)
# m.set_input('a_input', tvm.nd.array(x1.astype(dtype)))
# m.set_input(**params)

# # execute
# # warm up
# for i in range(5):
#     m.run()

# iter_times = []

# for i in range(FLAGS.num_iter):
#     start_time = time.time()
#     m.run()
#     iter_time = (time.time() - start_time) * 1000
#     iter_times.append(iter_time)
#     print("Iteration time %f ms" % (iter_time))

# print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
#     min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))
