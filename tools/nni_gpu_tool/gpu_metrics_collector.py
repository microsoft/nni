#!/usr/bin/python
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import os
import subprocess
import sys
import time

from xml.dom import minidom

def check_ready_to_run():
    if sys.platform == 'win32':
        pgrep_output = subprocess.check_output('wmic process where "CommandLine like \'%nni_gpu_tool.gpu_metrics_collector%\' and name like \'%python%\'" get processId')
        pidList = pgrep_output.decode("utf-8").strip().split()
        pidList.pop(0) # remove the key word 'ProcessId'
        pidList = list(map(int, pidList))
        pidList.remove(os.getpid())
        return len(pidList) == 0
    else:
        pgrep_output =subprocess.check_output('pgrep -fx \'python3 -m nni_gpu_tool.gpu_metrics_collector\'', shell=True)
        pidList = []
        for pid in pgrep_output.splitlines():
            pidList.append(int(pid))
        pidList.remove(os.getpid())
        return len(pidList) == 0

def main(argv):
    metrics_output_dir = os.environ['METRIC_OUTPUT_DIR']
    if check_ready_to_run() == False:
        # GPU metrics collector is already running. Exit
        exit(2)
    with open(os.path.join(metrics_output_dir, "gpu_metrics"), "w") as outputFile:
        pass
    os.chmod(os.path.join(metrics_output_dir, "gpu_metrics"), 0o777)
    cmd = 'nvidia-smi -q -x'
    while(True):
        try:
            smi_output = subprocess.check_output(cmd, shell=True)
            parse_nvidia_smi_result(smi_output, metrics_output_dir)
        except:
            exception = sys.exc_info()
            for e in exception:
                print("job exporter error {}".format(e))
        # TODO: change to sleep time configurable via arguments
        time.sleep(5)

def parse_nvidia_smi_result(smi, outputDir):
    try:
        xmldoc = minidom.parseString(smi)
        gpuList = xmldoc.getElementsByTagName('gpu')
        with open(os.path.join(outputDir, "gpu_metrics"), 'a') as outputFile:
            outPut = {}
            outPut["Timestamp"] = time.asctime(time.localtime())
            outPut["gpuCount"] = len(gpuList)
            outPut["gpuInfos"] = []
            for gpuIndex, gpu in enumerate(gpuList):
                gpuInfo ={}
                gpuInfo['index'] = gpuIndex
                gpuInfo['gpuUtil'] = gpu.getElementsByTagName('utilization')[0].getElementsByTagName('gpu_util')[0].childNodes[0].data.replace("%", "").strip()
                gpuInfo['gpuMemUtil'] = gpu.getElementsByTagName('utilization')[0].getElementsByTagName('memory_util')[0].childNodes[0].data.replace("%", "").strip()
                processes = gpu.getElementsByTagName('processes')
                runningProNumber = len(processes[0].getElementsByTagName('process_info'))
                gpuInfo['activeProcessNum'] = runningProNumber

                outPut["gpuInfos"].append(gpuInfo)
            print(outPut)
            outputFile.write("{}\n".format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush();
    except :
        e_info = sys.exc_info()
        print('xmldoc paring error')


if __name__ == "__main__":
    main(sys.argv[1:])
