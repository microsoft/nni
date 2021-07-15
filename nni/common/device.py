
from dataclasses import dataclass

@dataclass
class GPUDevice:
    host: str
    gpu_id: int
    status: str = 'free'

    def __eq__(self, o) -> bool:
        return self.host == o.host and self.gpu_id == o.gpu_id

    def __lt__(self, o) -> bool:
        if self.host < o.host:
            return True
        elif self.host > o.host:
            return False
        else:
            return self.gpu_id < o.gpu_id

    def __repr__(self) -> str:
        return "{Server-%s, GPU-%d, Status: %s}" % (self.host, self.gpu_id, self.status)

    def __hash__(self) -> int:
        return hash(self.host + '_' + self.gpu_id)

    def set_status(self, status):
        self.status = status
