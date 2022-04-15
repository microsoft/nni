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
from tvm import relay, auto_scheduler
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import numpy as np
import os.path
import time
from tvm.contrib import graph_runtime
import logging
import onnx
import argparse
from tvm.relay import data_dep_optimization as ddo
from tvm.contrib import graph_executor

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
# /data/znx/SpargenCks/hubert_sota_finegrained_no_tesa.onnx
parser.add_argument('--model_path', type=str, default='../../checkpoints/hubert/artifact_hubert_finegrained_no_propagation_onnx_with_tesa/model_no_tesa.onnx', help='The file name of the frozen graph.')
#parser.add_argument('--model_path', type=str, default='/data/znx/SpargenCks/hubert_sota_finegrained_no_tesa.onnx', help='The file name of the frozen graph.')
parser.add_argument('--warmup', type=int, default=5, help='The number of warmup iterations.')
parser.add_argument('--num_iter', type=int, default=100, help='The number of execution iterations.')
# parser.add_argument('--autotvm_log', type=str, default='', help='autotvm kernel tuning log')
parser.add_argument('--tuning_step', type=int, default=25000, help='tuning steps')
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
shape_dict = {'0': (32, 16000)}
# dummy_input = (torch.rand(32, 128).numpy(), torch.rand(32, 128).numpy(), torch.rand(32, 128).numpy())
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

print("ONNX imported to relay frontend.")

log_file = f'autotvm_hubert_5000.log'

tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=25000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

print("Tuning...")
#run_tuning()

print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
m = graph_executor.GraphModule(lib["default"](dev))

input1_shape = (32, 16000)
x1 = np.random.uniform(size=input1_shape)

data_tvm1 = tvm.nd.array(x1.astype('int32'))

m.set_input("0", data_tvm1)

print(m.benchmark(dev, repeat=3, min_repeat_ms=500))