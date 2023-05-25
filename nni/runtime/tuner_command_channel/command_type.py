# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, ClassVar

from nni.typehint import TrialMetric, SearchSpace
from nni.utils import MetricType

class CommandType(Enum):
    # in
    Initialize = 'IN'
    RequestTrialJobs = 'GE'
    ReportMetricData = 'ME'
    UpdateSearchSpace = 'SS'
    ImportData = 'FD'
    AddCustomizedTrialJob = 'AD'
    TrialEnd = 'EN'
    Terminate = 'TE'
    Ping = 'PI'

    # out
    Initialized = 'ID'
    NewTrialJob = 'TR'
    SendTrialJobParameter = 'SP'
    NoMoreTrialJobs = 'NO'
    KillTrialJob = 'KI'
    Error = 'ER'

class TunerIncomingCommand:
    # For type checking.
    command_type: ClassVar[CommandType]

# Only necessary commands to make NAS work.

@dataclass
class Initialize(TunerIncomingCommand):
    command_type: ClassVar[CommandType] = CommandType.Initialize
    search_space: SearchSpace

@dataclass
class RequestTrialJobs(TunerIncomingCommand):
    command_type: ClassVar[CommandType] = CommandType.RequestTrialJobs
    count: int

@dataclass
class UpdateSearchSpace(TunerIncomingCommand):
    command_type: ClassVar[CommandType] = CommandType.UpdateSearchSpace
    search_space: SearchSpace

@dataclass
class ReportMetricData(TunerIncomingCommand):
    command_type: ClassVar[CommandType] = CommandType.ReportMetricData
    parameter_id: int                      # Parameter ID.
    type: MetricType                       # Request parameter, periodical, or final.
    sequence: int                          # Sequence number of the metric.
    value: Optional[TrialMetric] = None    # The metric value. When type is NOT request parameter.
    trial_job_id: Optional[str] = None     # Only available when type is request parameter.
    parameter_index: Optional[int] = None  # Only available when type is request parameter.

@dataclass
class TrialEnd(TunerIncomingCommand):
    command_type: ClassVar[CommandType] = CommandType.TrialEnd
    trial_job_id: str         # The trial job id.
    parameter_ids: List[int]  # All parameter ids of the trial job.
    event: str                # The job's state

@dataclass
class Terminate(TunerIncomingCommand):
    command_type: ClassVar[CommandType] = CommandType.Terminate
