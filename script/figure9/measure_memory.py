from time import sleep
import os
import sys
import contextlib
import subprocess
import time
import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()


def analyze_log(log_path):
    if not os.path.exists(log_path):
        print(f"{log_path} does not exists")
    peak = 0
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if 'MiB' in line:
                try:
                    peak = max(peak, int(re.split(' ',line)[0]))
                except Exception as err:
                    pass
    return peak


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


patterns = ['coarse', 'coarse_int8', 'finegrained']
frameworks = ['jit', 'tvm', 'tvm-s', 'rammer', 'rammer-s', 'sparta', 'trt']
data = {}
model = args.model
for pattern in patterns:
    for framework in frameworks:
        path = '../figure8/{}_{}_{}'.format(model, pattern, framework)
        memory_usage = 0
        if os.path.exists(path):
            with pushd(path):
                print('Measure the memory for {} {} under {}'.format(model, pattern, framework))
                if os.path.exists('prepare_mem.sh'):
                    os.system('bash prepare_mem.sh')
                # here start the inference process at the same time and
                # measure the memory at the same time
                target_process = subprocess.Popen('bash run_mem.sh', shell=True)
            sleep(3) # wait the memory to be steady
            monitor_process = subprocess.Popen('bash 2080_mem.sh 0 > run.log', shell=True)
            target_process.wait()
            monitor_process.terminate()
            memory_usage = analyze_log('run.log')
        data['{}_{}_{}'.format(model, pattern, framework)] = memory_usage
print(data)
with open(f'{args.model}_data.json', 'w') as f:
    json.dump(data, f)