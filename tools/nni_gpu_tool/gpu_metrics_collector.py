# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import subprocess
import sys
import time
import traceback

from xml.dom import minidom

def check_ready_to_run():
    if sys.platform == 'win32':
        pgrep_output = subprocess.check_output(
            'wmic process where "CommandLine like \'%nni_gpu_tool.gpu_metrics_collector%\' and name like \'%python%\'" get processId')
        pidList = pgrep_output.decode("utf-8").strip().split()
        pidList.pop(0) # remove the key word 'ProcessId'
        pidList = list(map(int, pidList))
        pidList.remove(os.getpid())
        return not pidList
    else:
        pgrep_output = subprocess.check_output('pgrep -fxu "$(whoami)" \'python3 -m nni_gpu_tool.gpu_metrics_collector\'', shell=True)
        pidList = []
        for pid in pgrep_output.splitlines():
            pidList.append(int(pid))
        pidList.remove(os.getpid())
        return not pidList

def main(argv):
    metrics_output_dir = os.environ['METRIC_OUTPUT_DIR']
    if check_ready_to_run() == False:
        # GPU metrics collector is already running. Exit
        exit(2)
    cmd = 'nvidia-smi -q -x'.split()
    while(True):
        try:
            smi_output = subprocess.check_output(cmd)
        except Exception:
            traceback.print_exc()
            gen_empty_gpu_metric(metrics_output_dir)
            break
        parse_nvidia_smi_result(smi_output, metrics_output_dir)
        # TODO: change to sleep time configurable via arguments
        time.sleep(5)

def parse_nvidia_smi_result(smi, outputDir):
    try:
        old_umask = os.umask(0)
        xmldoc = minidom.parseString(smi)
        gpuList = xmldoc.getElementsByTagName('gpu')
        with open(os.path.join(outputDir, "gpu_metrics"), 'a') as outputFile:
            outPut = {}
            outPut["Timestamp"] = time.asctime(time.localtime())
            outPut["gpuCount"] = len(gpuList)
            outPut["gpuInfos"] = []
            for gpuIndex, gpu in enumerate(gpuList):
                gpuInfo = {}
                gpuInfo['index'] = gpuIndex
                gpuInfo['gpuUtil'] = gpu.getElementsByTagName('utilization')[0]\
                    .getElementsByTagName('gpu_util')[0]\
                    .childNodes[0].data.replace("%", "").strip()
                gpuInfo['gpuMemUtil'] = gpu.getElementsByTagName('utilization')[0]\
                    .getElementsByTagName('memory_util')[0]\
                    .childNodes[0].data.replace("%", "").strip()
                processes = gpu.getElementsByTagName('processes')
                runningProNumber = len(processes[0].getElementsByTagName('process_info'))
                gpuInfo['activeProcessNum'] = runningProNumber

                outPut["gpuInfos"].append(gpuInfo)
            print(outPut)
            outputFile.write("{}\n".format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush();
    except:
        # e_info = sys.exc_info()
        print('xmldoc paring error')
    finally:
        os.umask(old_umask)

def gen_empty_gpu_metric(outputDir):
    try:
        old_umask = os.umask(0)
        with open(os.path.join(outputDir, "gpu_metrics"), 'a') as outputFile:
            outPut = {}
            outPut["Timestamp"] = time.asctime(time.localtime())
            outPut["gpuCount"] = 0
            outPut["gpuInfos"] = []
            print(outPut)
            outputFile.write("{}\n".format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush()
    except Exception:
        traceback.print_exc()
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    main(sys.argv[1:])
