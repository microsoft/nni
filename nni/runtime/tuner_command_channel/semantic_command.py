# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Literal
from nni.utils import MetricType

@dataclass
class BasicCommand:
    command_type: str = ''
    trial_job_id: str = ''
    parameter_id: int = 0
    parameter_index: int = 0
    parameters: dict = {}
    parameter_source: str = ''

@dataclass
class ReportMetricData:
    command_type: str = 'ReportMetricData'
    trial_job_id: str = ''
    parameter_id: int = 0
    parameter_index: int = 0
    type: Literal['FINAL', 'PERIODICAL', 'REQUEST_PARAMETER'] = MetricType.PERIODICAL
    value: str = ''
    sequence: int = 0

@dataclass
class UpdateSearchSpace:
    command_type: str = 'UpdateSearchSpace'
    name: str = ''

@dataclass
class ImportData:
    command_type: str = 'ImportData'
    parameters: dict = {}
    value: str = ''

@dataclass
class TrialEnd:
    command_type: str = 'TrialEnd'
    trial_job_id: str = ''
    event: str = ''

@dataclass
class NewTrialJob(BasicCommand):
    command_type: str = 'NewTrialJob'
    placement_constraint: str = ''
    version_info: str = ''

@dataclass
class SendTrialJobParameter(BasicCommand):
    command_type: str = 'SendTrialJobParameter'

@dataclass
class NoMoreTrialJobs(BasicCommand):
    command_type: str = 'NoMoreTrialJobs'

@dataclass
class KillTrialJob:
    command_type: str = 'KillTrialJob'
    trial_job_id: str = 'null'
