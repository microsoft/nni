# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Collect GPU utilization metrics, and debug info if ``--detail`` is specified.
Results are printed to stdout in JSON format.

See `ts/nni_manager/common/gpu_scheduler/collect_info` for details.
"""

# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import sys
from typing import Any, Literal

from pynvml import *

errors = set()

def main() -> None:
    info = collect('--detail' in sys.argv)
    if info:
        data = asdict(info, dict_factory=dict_factory)
        data['success'] = True
    else:
        data = {'success': False}
    if errors:
        data['failures'] = sorted(errors)
    print(json.dumps(data), flush=True)

def dict_factory(obj):
    ret = {}
    for k, v in obj:
        if k.startswith('_'):
            continue
        if v is None:
            continue
        words = k.split('_')
        camel_k = words[0] + ''.join(word.title() for word in words[1:])
        ret[camel_k] = v
    return ret

def collect(detail: bool) -> SystemInfo | None:
    try:
        nvmlInit()
    except Exception as e:
        errors.add(f'init: {e}')
        return None

    info = None
    try:
        info = SystemInfo(detail)
    except Exception as e:
        errors.add(f'unexpected: {e}')

    try:
        nvmlShutdown()
    except Exception as e:
        errors.add(f'shutdown: {e}')

    return info

@dataclass(init=False)
class SystemInfo:
    gpu_number: int = 0
    driver_version: str | None = None
    cuda_version: int | None = None
    gpus: list[GpuInfo]
    processes: list[ProcessInfo]

    def __init__(self, detail: bool):
        self.gpus = []
        self.processes = []

        try:
            self.gpu_number = nvmlDeviceGetCount()
        except Exception as e:
            errors.add(f'gpu_number: {e}')

        if detail:
            try:
                self.driver_version = nvmlSystemGetDriverVersion()
            except Exception as e:
                errors.add(f'driver_version: {e}')

            try:
                self.cuda_version = nvmlSystemGetCudaDriverVersion_v2()
            except Exception as e:
                errors.add(f'cuda_version: {e}')

        self.gpus = [GpuInfo(index, detail) for index in range(self.gpu_number)]

        procs = []
        for gpu in self.gpus:
            procs += gpu._procs
        self.processes = sorted(procs, key=(lambda proc: proc.pid))

@dataclass(init=False)
class GpuInfo:
    index: int
    model: str | None = None
    cuda_cores: int | None = None
    gpu_memory: int | None = None
    free_gpu_memory: int | None = None
    gpu_core_utilization: float | None = None
    gpu_memory_utilization: float | None = None
    _procs: list[ProcessInfo]

    def __init__(self, index: int, detail: bool):
        self.index = index
        self._procs = []

        try:
            device = nvmlDeviceGetHandleByIndex(self.index)
        except Exception as e:
            errors.add(f'device: {e}')
            return

        if detail:
            try:
                self.model = nvmlDeviceGetName(device)
            except Exception as e:
                errors.add(f'model: {e}')

            try:
                self.cuda_cores = nvmlDeviceGetNumGpuCores(device)
            except Exception as e:
                errors.add(f'cuda_cores: {e}')

            try:
                mem = nvmlDeviceGetMemoryInfo(device)
                self.gpu_memory = mem.total
                self.free_gpu_memory = mem.free
            except Exception as e:
                errors.add(f'gpu_memory: {e}')

        try:
            util = nvmlDeviceGetUtilizationRates(device)
            self.gpu_core_utilization = util.gpu / 100
            self.gpu_memory_utilization = util.memory / 100
        except Exception as e:
            errors.add(f'gpu_utilization: {e}')

        try:
            cprocs = nvmlDeviceGetComputeRunningProcesses_v3(device)
            gprocs = nvmlDeviceGetGraphicsRunningProcesses_v3(device)
            self._procs += [ProcessInfo(proc, self.index, 'compute', detail) for proc in cprocs]
            self._procs += [ProcessInfo(proc, self.index, 'graphics', detail) for proc in gprocs]
        except Exception as e:
            errors.add(f'process: {e}')

@dataclass(init=False)
class ProcessInfo:
    pid: int
    name: str | None = None
    gpu_index: int
    type: Literal['compute', 'graphics']
    used_gpu_memory: int | None

    def __init__(self, info: Any, gpu_index: int, type_: Literal['compute', 'graphics'], detail: bool):
        self.pid = info.pid
        if detail:
            self.name = nvmlSystemGetProcessName(self.pid)
        self.gpu_index = gpu_index
        self.type = type_
        self.used_gpu_memory = info.usedGpuMemory

if __name__ == '__main__':
    main()
