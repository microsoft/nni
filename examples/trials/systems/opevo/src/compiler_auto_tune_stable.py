#!/usr/bin/env python3

## TODO: optimize c-mcpu metric; early-stop handler; fp16/int8; Kill pyRPC;

import numpy as np
import tvm
import logging
import math
import re
import sys, time, subprocess, os, random, hashlib
from tvm import autotvm
import topi
import json
from topi.util import get_const_tuple
import importlib
from tvm.autotvm.task.dispatcher import ApplyConfig
from tvm.autotvm.task import ConfigEntity
from threading import Timer

backend = os.environ['BACKEND'] if 'BACKEND' in os.environ else 'c-cuda'

def system_lock(key_ids):
  import socket, time
  occupied_sock = None
  while not occupied_sock:
    for key_id in key_ids:
      try:
        sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 9050 + key_id))
        sock.listen(1)
        occupied_sock = (sock, key_id)
        break
      except:
        try:
          sock.shutdown(socket.SHUT_RDWR)
          sock.close()
        except:
          sock.close()
    if occupied_sock:
      break
    # print('still waiting ..')
    time.sleep(0.2)

  # print('Using key_id = %d' % occupied_sock[1])
  sock = occupied_sock[0]

  def unlock_fd():
    try:
      sock.shutdown(socket.SHUT_RDWR)
      sock.close()
    except:
      sock.close()
  return unlock_fd, occupied_sock[1]

def show_search_space(config_space, printable):
  search_space = {}
  for _, name in enumerate(config_space.space_map):
    curr = config_space.space_map[name]
    if (curr.__class__ == tvm.autotvm.task.space.SplitSpace):
      search_space[name] = {"_type": "factor", "_value": [curr.product, curr.num_output]}
    elif (curr.__class__ == tvm.autotvm.task.space.OtherOptionSpace):
      search_space[name] = {"_type": "choice", "_value": [x.val for x in curr.entities]}
    else:
      raise Exception("Cannot recognize search space type: %s" % (config_space.space_map[name].__class__))
  json_space = json.dumps(search_space)
  print("\n>> Search Space = %s" % json_space)
  if printable:
    print("\n>> Writing Search Space to './search_space.json'..")
    with open("search_space.json", "w") as fp:
      fp.write(json_space)
    print("\n>> Done")
    sys.exit(0)

def get_tuning_parallism():
  if 'DEV_NUM' in os.environ:
    dev_num = int(os.environ['DEV_NUM'])
  else:
    if backend in ['c-rocm', '#rocm']:
      devices = subprocess.getoutput('/opt/rocm/bin/rocm_agent_enumerator | grep -v gfx000').split()
      if not devices:
        raise Exception("Not valid rocm device found.")
      dev_num = len(devices)
    elif backend in ['c-cuda', '#cuda']:
      devices = subprocess.getoutput('ls /dev/nvidia[0-9]* 2>/dev/null').split()
      if not devices:
        raise Exception("Not valid rocm device found.")
      dev_num = len(devices)
    else:
      raise Exception("Unrecognized backend: %s" % backend)
  print('  >> Tuning parallism = %d' % dev_num)
  return dev_num

def local_get_dir_file(rel_file, dir_sid=None):
  if not dir_sid:
    dir_sid = os.environ['DIR_SID'] if 'DIR_SID' in os.environ else '_'
  dir_space = '/tmp/tvm_autotvm_engine'
  os.system('mkdir -p "%s/%s"' % (dir_space, dir_sid))
  return "%s/%s/%s" % (dir_space, dir_sid, rel_file)

def run_process_with_timeout(args, timeout=None, envs=None):
  try:
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=envs)
    retcode = proc.wait(timeout=timeout)
    return retcode == 0
  except subprocess.TimeoutExpired:
    print('Timed out - killing', proc.pid)
    proc.kill()
    return False

def parse_launch_bounds(code):
  func_arr = code.split('extern "C" __global__ ')
  for i in range(1, len(func_arr)):
    axis_map = dict()
    lines = func_arr[i].split('\n')
    for it in lines:
      if it.startswith('  // [thread_extent] '):
        words = it.split(' ')
        nthread = int(words[-1])
        axis = words[-3]
        if axis in axis_map:
          if axis_map[axis] != nthread:
            assert(False)
        else:
          axis_map[axis] = nthread
    block_bound = axis_map.get('threadIdx.x', 1) * axis_map.get('threadIdx.y', 1) * axis_map.get('threadIdx.z', 1)
    func_arr[i] = 'extern "C" __global__ __launch_bounds__(%d) %s' % (block_bound, func_arr[i])

  code = ''.join(func_arr)
  return code

def translate_code(code):
  if backend == 'c-rocm':
    code = parse_launch_bounds(code)
    code = '#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n\n'+ code.replace('(__shared__ float4*)', '(float4*)').replace('#include <cuda_fp16.h>', '').replace('typedef unsigned long long uint64_t;', '')
  elif backend in ['#cuda', 'c-cuda']:
    code = parse_launch_bounds(code)
    code = '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n\n' + code
  else:
    raise Exception("Unrecognized backend: %s" % backend)
  return code

@tvm.register_func
def tvm_callback_backend_proc(code):
  native_code = translate_code(code)
  # Compile code
  module_data = None
  if backend == 'c-rocm':
    gcn_arch = subprocess.getoutput('/opt/rocm/bin/rocm_agent_enumerator | sort | uniq | grep -v gfx000 | tail -n 1').strip()
    if not gcn_arch:
      raise RuntimeError("Compilation error: no valid gcn_arch gpu detected!")
    temp_code = local_get_dir_file("my_kernel.cc")
    temp_cobj = local_get_dir_file("my_kernel.hsaco")
    args = ['/opt/rocm/bin/lpl', temp_code, '-t=' + gcn_arch, '-f="-Wno-ignored-attributes -D__HIP_PLATFORM_HCC__=1"', '-o', temp_cobj]
  elif backend in ['#cuda', 'c-cuda']:
    temp_code = local_get_dir_file("my_kernel.cu")
    temp_cobj = local_get_dir_file("my_kernel.ptx")
    args = ['/usr/local/cuda/bin/nvcc', temp_code, '--ptx', '-O3', '-o', temp_cobj]
  else:
    raise Exception("Unrecognized backend: %s" % backend)
  with open(temp_code, 'w') as fp:
    fp.write(native_code)
  print('[Build @%x]' % os.getpid(), ' '.join(args))
  if not run_process_with_timeout(args, 10):
    raise Exception("Compilation failed or time limit exceeded")
  if module_data is None:
    module_data = bytearray(open(temp_cobj, "rb").read())
  return module_data

def run_config_entity(params_given, dir_sid, expected_timecost='inf', tune_slot_id=0):
  dir_sid = str(dir_sid)
  result_file = local_get_dir_file('result.txt', dir_sid)
  try:
    os.remove(result_file)
  except:
    pass
  config_str = json.dumps(params_given)
  envs = os.environ.copy()
  envs['CONFIG'] = config_str
  envs['DIR_SID'] = dir_sid
  envs['CUDA_VISIBLE_DEVICES'] = str(tune_slot_id)
  print("  >> Try param_entity on sid = %s: config = %s, slot_id = %d" % (dir_sid, config_str, tune_slot_id))
  try:
    assert(True == run_process_with_timeout(["python%d" % sys.version_info.major] + sys.argv, envs=envs))
    result = float(open(result_file, 'r').read().strip())
  except:
    result = float('inf')
  print("  >> Try param_entity on sid = %s: result = `%.6f`" % (dir_sid, result))
  return result

def compute_gflops(flop, t):
  return flop / (t * 1e3) / 1e6

def search_op_config(code_only=False):
  tvm_target = 'cuda'
  logging.getLogger('autotvm').setLevel(logging.DEBUG)
  logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

  default_tune_op = importlib.import_module('templates.' + (os.environ['OP']))
  print('  >> Backend = %s, Python PID = %s, Task = %s;' % (backend, os.getpid(), default_tune_op.__name__))

  task = autotvm.task.create(default_tune_op.get_template_op, args=(), target=tvm_target)
  op_attributes = default_tune_op.op_attributes
  op_summary = '_'.join([k + str(op_attributes[k]) for k in op_attributes])

  def json_to_config(json_dict):
    config = ConfigEntity.from_json_dict({"i": -1, "t": "", "c": None, "e": json_dict})
    return config

  def config_to_json(config):
    jobj = config.to_json_dict()['e']
    json_dict = dict()
    for i in range(len(jobj)):
      assert(jobj[i][1] in ['sp', 'ot'])
      json_dict[jobj[i][0]] = jobj[i][2]
    return json_dict

  num_trials = int(os.environ['STEP']) if 'STEP' in os.environ else 0

  if 'CONFIG' in os.environ:
    params_given = json.loads(os.environ['CONFIG'])
    print("====>> [Current Config Option]", os.environ['CONFIG'])

    trial_config = []
    for key in params_given:
      trial_config.append([key, "sp" if type(params_given[key]) is list else "ot", params_given[key]])
    best_config = json_to_config(trial_config)

  elif 'NNI_TRIAL_JOB_ID' in os.environ:
    show_search_space(task.config_space, os.environ['NNI_TRIAL_JOB_ID'] == '@')
    import nni
    params_given = nni.get_next_parameter()
    if params_given is None:
      raise
    local_dir_id = os.environ['NNI_TRIAL_JOB_ID']
    t = run_config_entity(params_given, local_dir_id)
    gflops = compute_gflops(task.flop, t)
    print('[TVM-engine] Final entity result is: %g' % gflops)
    try:
      nni.report_final_result(gflops)
    except:
      print('[TVM-engine] (not reporting final result to NNI.)')
    exit(0)

  elif num_trials > 0:
    n_parallel = 16 if 'BATCH' not in os.environ else int(os.environ['BATCH'])
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=n_parallel),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )
    # if DO_TUNING:
    tuner = autotvm.tuner.XGBTuner(task, num_threads=8)

    from concurrent.futures import ThreadPoolExecutor
    thread_pool = ThreadPoolExecutor(max_workers=n_parallel)

    dev_num = get_tuning_parallism()

    def parse_configs(task, configs):
      results = []
      futures = []
      expected_timecost = 'inf'
      for i in range(len(configs)):
        futures.append(thread_pool.submit(run_config_entity, config_to_json(configs[i]), i, expected_timecost, i % dev_num))
      for i in range(len(configs)):
        t = futures[i].result()
        if t < tuner.task.best_config[0]:
          tuner.task.best_config = (t, configs[i])
        results.append(autotvm.measure.MeasureResult(costs=(t,), error_no=0, all_cost=i, timestamp=time.time()))
      return results

    tuner.task.best_config = (float('inf'), None)
    tuner.parse_configs = parse_configs

    tuner.tune(n_trial=num_trials, measure_option=measure_option, callbacks=[])
    assert(not math.isinf(tuner.task.best_config[0]))
    best_config = tuner.task.best_config[1]
    print('\n[Best Config]', json.dumps(config_to_json(best_config)))
  else:
    best_config = task.config_space

  with ApplyConfig(best_config):
    with tvm.target.create(tvm_target):
      s, arg_bufs = default_tune_op.get_template_op()
      lower_source = str(tvm.lower(s, arg_bufs, simple_mode=True))

      # Verify Source Code
      assert(len(('\n' + lower_source).split('\nproduce ')) == 2)
      lower_file = local_get_dir_file('my_kernel.lower')
      with open(lower_file, 'w') as fp:
        fp.write(lower_source)

      max_threads_per_block = tvm.ndarray.gpu(0).max_threads_per_block
      max_shared_memory_per_block = tvm.ndarray.gpu(0).max_shared_memory_per_block

      thread_extents = subprocess.getoutput("cat '%s' | grep '^ *// attr.*iter_var.*thread_extent'" % (lower_file)).split('\n')
      reserved_axes = dict({'threadIdx.x': None, 'threadIdx.y': None, 'threadIdx.z': None, 'blockIdx.x': None, 'blockIdx.y': None, 'blockIdx.z': None})
      for line in thread_extents:
        thread_name = line.split('[iter_var(')[-1].split(',')[0]
        if thread_name in reserved_axes:
          thread_val = int(line.split('thread_extent = ')[-1])
          if reserved_axes[thread_name] is not None:
            if reserved_axes[thread_name] != thread_val:
              assert(False)
          else:
            reserved_axes[thread_name] = thread_val
        else:
          raise Exception("Invalid thread_axis name: %s" % thread_name)

      num_threads = 1
      for thread_name in ['threadIdx.x', 'threadIdx.y', 'threadIdx.z']:
        if reserved_axes[thread_name] is not None:
          num_threads *= reserved_axes[thread_name]
      if num_threads > max_threads_per_block:
        raise Exception("Invalid kernel code: using num_threads %d > max_threads_per_block %d" % (num_threads, max_threads_per_block))

      allocate_shared = subprocess.getoutput("cat '%s' | grep 'allocate .*shared\[.*\]'" % (lower_file)).split('\n')
      shared_memory_in_bytes = 0
      for line in allocate_shared:
        if not line:
          continue
        parts = line.split('[')
        assert(len(parts) == 2)
        parts = parts[1].split(' * ')
        assert(len(parts) == 2)
        assert(parts[1][-1] == ']')
        allocate_type = parts[0]
        allocate_val = int(parts[1][:-1])
        if allocate_type in ['float32']:
          shared_memory_in_bytes += allocate_val * 4
        else:
          raise Exception("Unrecognized shared memory data type: %s" % allocate_type)
      if shared_memory_in_bytes > max_shared_memory_per_block:
        raise Exception("Invalid kernel code: using shared_memory_in_bytes %d > max_shared_memory_per_block %d" % (shared_memory_in_bytes, max_shared_memory_per_block))

      func = tvm.build(s, arg_bufs, tvm_target, name='template_op')

  assert(len(func.imported_modules) == 1)
  device_source = translate_code(func.imported_modules[0].get_source())

  if code_only:
    return device_source

  if lower_source and device_source:
    tune_slot_id = 0 if 'CUDA_VISIBLE_DEVICES' not in os.environ else int(os.environ['CUDA_VISIBLE_DEVICES'])
    exec_fd, _ = system_lock([tune_slot_id])
    gpu_id = 0
    ctx = tvm.context(tvm_target, gpu_id)
    tensors, outs = [], []
    for arg in arg_bufs:
      shape = [int(x) for x in arg.shape]
      is_output = arg.op.__class__ != tvm.tensor.PlaceholderOp
      from tvm._ffi.ndarray import empty
      td = empty(shape, arg.dtype, ctx)
      if is_output:
        outs.append(td)
      tensors.append(td)

    def timeout_handler():
      print("Error: Timeout during Kernel warmup")
      os._exit(1)

    my_timer = Timer(10, timeout_handler, [])
    my_timer.start()
    # Warmup
    func(*tensors)
    tvm.ndarray.gpu(gpu_id).sync()
    # Estimate
    t_start = time.time()
    func(*tensors)
    tvm.ndarray.gpu(gpu_id).sync()
    t_diff = time.time() - t_start
    my_timer.cancel()
    del my_timer

    num_runs = max(3, min(100, math.floor(1.0 / t_diff)))
    timeout_seconds = math.ceil((num_runs + 5) * t_diff)
    my_timer = Timer(timeout_seconds, timeout_handler, [])
    my_timer.start()
    timer_f = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    t = timer_f(*tensors).mean
    my_timer.cancel()
    exec_fd()

    gflops = compute_gflops(task.flop, t)
    print("[TVM-engine] Average time cost of %d runs = %g ms, %g gflops." % (num_runs, t * 1e3, gflops))

    with open(local_get_dir_file('result.txt'), 'w') as fp:
      fp.write(str(t))


if __name__ == '__main__':
    try:
      search_op_config()
    except SystemExit:
      sys.exit(0)
    except:
      import traceback
      traceback.print_exc()
