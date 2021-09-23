# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class Device(ABC):
    node_id: str
    status: Literal['idle', 'busy', 'unknown'] = 'idle'

    def __eq__(self, o) -> bool:
        if isinstance(self, type(o)):
            return self.node_id == o.node_id
        else:
            return False

    def __lt__(self, o) -> bool:
        return self.node_id < o.node_id

    def set_status(self, status):
        self.status = status

    def __repr__(self) -> str:
        return "{Abstract Device %s, Status %s}" % (self.node_id, self.status)

    @abstractmethod
    def device_repr(self) -> str:
        pass


@dataclass
class GPUDevice(Device):
    gpu_id: str = -1

    def __init__(self, node_id, gpu_id, status='idle'):
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.status = status

    def __eq__(self, o: Device) -> bool:
        if isinstance(o, GPUDevice):
            return self.node_id == o.node_id and self.gpu_id == o.gpu_id
        return False

    def __lt__(self, o: Device) -> bool:
        if self.node_id < o.node_id:
            return True
        elif self.node_id > o.node_id:
            return False
        else:
            if isinstance(o, GPUDevice):
                return self.gpu_id < o.gpu_id
            else:
                return True

    def __repr__(self) -> str:
        return "{Environment %s, GPU %d, Status %s}" % (self.node_id, self.gpu_id, self.status)

    def __hash__(self) -> int:
        return hash(self.node_id + '_' + str(self.gpu_id))

    def device_repr(self,):
        return f"cuda:{self.gpu_id}"


@dataclass
class CPUDevice(Device):
    def __init__(self, node_id):
        self.node_id = node_id
        self.device = 'cpu'

    def __repr__(self) -> str:
        return "{CPU Device, NodeID %s, Status %s}" % (self.node_id, self.status)

    def device_repr(self):
        return "cpu"
