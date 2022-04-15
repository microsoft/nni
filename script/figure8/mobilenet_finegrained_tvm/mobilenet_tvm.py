import onnx
import numpy as np
import itertools
import scipy.sparse as sp
import torch
import torch.nn as nn

import tvm
from tvm import relay, auto_scheduler
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.relay import data_dep_optimization as ddo
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor

import os

import argparse

class MobileNet(nn.Module):
    def __init__(self, n_class=1000):
        super(MobileNet, self).__init__()
        self.nclass = n_class

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, self.nclass)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def load_model_from_onnx(args, shape_list):
    onnx_model = onnx.load(args.model_path)
    print(onnx_model.graph.input)
    # convert onnx model to tvm model
    print("Begin to generate tvm model by parsing onnx model....")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_list)
    print("Parse successfully!")
    return mod, dict(params.items())

def load_model_from_pytorch(shape_list):
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    #model.eval()
    model = MobileNet()
    model = model.eval()
    input_shape = list(shape_list["input_1"])
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    input_name = "input.1"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params

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

"""
def random_sparse_bert_params(func, params, density, BS_R, BS_C):
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
        if shape[0] % BS_R == 0 and shape[1] % BS_C == 0:
            new_w = random_bsr_matrix(shape[0], shape[1], BS_R, BS_C, density)
            new_params[name] = tvm.nd.array(new_w)
    return new_params
"""

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

def run_relay_graph(mod, params, shape_dict, target, ctx):
    with relay.build_config(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    input_shape = shape_dict["input.1"]
    dummy_data = np.random.uniform(size=input_shape, low=0, high=input_shape[1]).astype("int32")

    m = graph_runtime.GraphModule(lib["default"](ctx))
    m.set_input(0, dummy_data)
    m.run()
    tvm_output = m.get_output(0)

    ftimer = m.module.time_evaluator("run", ctx, repeat=5, number=5)
    prof_res = np.array(ftimer().results) * 1000
    print(
        "%-20s %-19s (%s)"
        % ("Runtime:", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )
    return tvm_output

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

def quantize(mod, params, data_aware):
    input_shape = [batch_size, 3, 224, 224]
    dummy_input = torch.rand(input_shape)
    calibration_dataset = {'data':dummy_input}
    if data_aware:
        with relay.quantize.qconfig(nbit_input=8, nbit_weight=8, dtype_input="int8", dtype_weight="int8", dtype_activation = "int8", calibrate_mode="kl_divergence", skip_dense_layer=False, weight_scale="max"):
            mod = relay.quantize.quantize(mod, params, calibration_dataset)
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)


    return mod


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../../checkpoints/mobilenet/artifact_mobilenet_finegrained_no_propagation/model_no_tesa.onnx', help='The file name of the frozen graph.')
#parser.add_argument('--model_path', type=str, default='/data/znx/SpargenCks/mobilenet_sota_finegrained_no_tesa.onnx', help='The file name of the frozen graph.')
parser.add_argument('--warmup', type=int, default=5, help='The number of warmup iterations.')
parser.add_argument('--num_iter', type=int, default=100, help='The number of execution iterations.')
parser.add_argument('--tuning_step', type=int, default=1000, help='tuning steps')
args = parser.parse_args()

target = 'cuda'
target_host = 'llvm'
layout = 'NCHW'
ctx = tvm.gpu(0)

# onnx parameter definition
input_name = "input.1"
batch_size = 32
input_shape = (batch_size, 3, 224, 224)
shape_list = {input_name: input_shape}

# block sparse block size
bs_r = 1
sparsity = 0.95

mod, params = load_model_from_onnx(args, shape_list)
# mod, params = load_model_from_pytorch(shape_list)

# mod = quantize(mod, params, data_aware=False)
#tasks = autotvm.task.extract_from_program(mod["main"], target=target,
#                                        params=params,
#                                        ops=(relay.op.get("nn.dense"),relay.op.get("nn.conv2d")))

# run sparse
#mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
#params = random_sparse_bert_params(mod, params, BS_R=1, BS_C=1, density=1 - sparsity)
#mod, params = ddo.bsr_dense.convert(mod, params, (32, 32), sparsity_threshold=1)
#print("Block Sparse Model with {blocksize}x1 blocks:".format(blocksize=bs_r))

log_file = "mobilenet_autotvm_5000.log"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 1000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}

"""
tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                        params=params,
                                        ops=(relay.op.get("nn.dense"),relay.op.get("nn.conv2d")))
"""

tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

print("Tuning...")
# run_tuning()

print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

input_shape = [batch_size, 3, 224, 224]
x = np.random.uniform(size=input_shape)
data_tvm = tvm.nd.array(x.astype('float32'))

module.set_input("input.1", data_tvm)

print(module.benchmark(dev, repeat=100, min_repeat_ms=500))
