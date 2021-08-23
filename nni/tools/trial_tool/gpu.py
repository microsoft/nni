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
    return info


def parse_nvidia_smi_result(smi):
    try:
        output = {}
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

            gpuInfo['gpuType'] = gpu.getElementsByTagName('product_name')[0]\
                .childNodes[0].data
            memUsage = gpu.getElementsByTagName('fb_memory_usage')[0]
            gpuInfo['gpuMemTotal'] = memUsage.getElementsByTagName('total')[0]\
                .childNodes[0].data.replace("MiB", "").strip()
            gpuInfo['gpuMemUsed'] = memUsage.getElementsByTagName('used')[0]\
                .childNodes[0].data.replace("MiB", "").strip()
            gpuInfo['gpuMemFree'] = memUsage.getElementsByTagName('free')[0]\
                .childNodes[0].data.replace("MiB", "").strip()

            output["gpuInfos"].append(gpuInfo)
    except Exception:
        traceback.print_exc()
        output = {}
    return output


def gen_empty_gpu_metric():
    try:
        output = {}
        output["Timestamp"] = time.asctime(time.localtime())
        output["gpuCount"] = 0
        output["gpuInfos"] = []
    except Exception:
        traceback.print_exc()
        output = {}
    return output
