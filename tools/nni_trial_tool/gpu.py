# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import time
import traceback
from xml.dom import minidom


def collect_gpu_usage(node_id):
    cmd = 'nvidia-smi -q -x'.split()
    info = None
    try:
        smi_output = subprocess.check_output(cmd)
        info = parse_nvidia_smi_result(smi_output)
    except Exception:
        traceback.print_exc()
        info = gen_empty_gpu_metric()
    if node_id is not None:
        info["node"] = node_id
    return info


def parse_nvidia_smi_result(smi):
    output = {}
    try:
        xmldoc = minidom.parseString(smi)
        gpuList = xmldoc.getElementsByTagName('gpu')
        output["Timestamp"] = time.asctime(time.localtime())
        output["gpuCount"] = len(gpuList)
        output["gpuInfos"] = []
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

            output["gpuInfos"].append(gpuInfo)
    except Exception:
        # e_info = sys.exc_info()
        traceback.print_exc()
    return output


def gen_empty_gpu_metric():
    output = {}
    try:
        output["Timestamp"] = time.asctime(time.localtime())
        output["gpuCount"] = 0
        output["gpuInfos"] = []
    except Exception:
        traceback.print_exc()
    return output
