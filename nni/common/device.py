# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class GPUDevice:
    node_id: str
    gpu_id: int
    status: Literal['idle', 'busy', 'unknown'] = 'idle'

    def __eq__(self, o) -> bool:
        if isinstance(o, GPUDevice):
            return self.node_id == o.node_id and self.gpu_id == o.gpu_id
        return False

    def __lt__(self, o) -> bool:
        if self.node_id < o.node_id:
            return True
        elif self.node_id > o.node_id:
            return False
        else:
            return self.gpu_id < o.gpu_id

    def __repr__(self) -> str:
        return "{Environment %s, GPU %d, Status %s}" % (self.node_id, self.gpu_id, self.status)

    def __hash__(self) -> int:
        return hash(self.node_id + '_' + str(self.gpu_id))

    def set_status(self, status):
        self.status = status

    def device_repr(self,):
        return f"cuda:{self.gpu_id}"
