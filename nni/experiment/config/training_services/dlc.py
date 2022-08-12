# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional

from typing_extensions import Literal

from ..training_service import TrainingServiceConfig

__all__ = ['DlcConfig']

@dataclass(init=False)
class DlcConfig(TrainingServiceConfig):
    platform: Literal['dlc'] = 'dlc'
    type: str = 'Worker'
    image: str # 'registry-vpc.{region}.aliyuncs.com/pai-dlc/tensorflow-training:1.15.0-cpu-py36-ubuntu18.04',
    job_type: str = 'TFJob'
    pod_count: int
    ecs_spec: str # e.g.,'ecs.c6.large'
    region: str
    workspace_id: str
    nas_data_source_id: str
    oss_data_source_id: Optional[str] = None
    access_key_id: str
    access_key_secret: str
    local_storage_mount_point: str
    container_storage_mount_point: str
