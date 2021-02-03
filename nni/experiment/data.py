from dataclasses import dataclass
import json
from typing import List


@dataclass
class TrialResult:
    """
    TrialResult stores the result information of a trial job.

    Attributes
    ----------
    parameter: dict
        Hyper parameters for this trial.
    value: serializable object, usually a number, or a dict with key "default" and other extra keys
        Final result.
    trialJobId: str
        Trial job id.
    """
    parameter: dict
    value: dict
    trialJobId: str

    def __init__(self, parameter: dict, value: str, trialJobId: str):
        self.parameter = parameter
        self.value = json.loads(value)
        self.trialJobId = trialJobId


@dataclass
class TrialMetricData:
    """
    TrialMetricData stores the metric data of a trial job.
    A trial job may have both intermediate metric and final metric.

    Attributes
    ----------
    timestamp: int
        Time stamp.
    trialJobId: str
        Trial job id.
    parameterId: int
        Parameter id.
    type: str
        Metric type, `PERIODICAL` for intermediate result and `FINAL` for final result.
    sequence: int
        Sequence number in this trial.
    data: serializable object, usually a number, or a dict with key "default" and other extra keys
        Metric data.
    """
    timestamp: int
    trialJobId: str
    parameterId: int
    type: str
    sequence: int
    data: dict

    def __init__(self, timestamp: int, trialJobId: str, parameterId: int, type: str, sequence: int, data: str): # pylint: disable=W0622
        self.timestamp = timestamp
        self.trialJobId = trialJobId
        self.parameterId = parameterId
        self.type = type
        self.sequence = sequence
        self.data = json.loads(json.loads(data))


@dataclass
class TrialHyperParameters:
    """
    TrialHyperParameters stores the hyper parameters of a trial job.

    Attributes
    ----------
    parameter_id: int
        Parameter id.
    parameter_source: str
        Parameter source.
    parameters: dict
        Hyper parameters.
    parameter_index: int
        Parameter index.
    """
    parameter_id: int
    parameter_source: str
    parameters: dict
    parameter_index: int


@dataclass
class TrialJob:
    """
    TrialJob stores the information of a trial job.

    Attributes
    ----------
    trialJobId: str
        Trial job id.
    status: str
        Job status.
    hyperParameters: list of `nni.experiment.TrialHyperParameters`
        See `nni.experiment.TrialHyperParameters`.
    logPath: str
        Log path.
    startTime: int
        Job start time (timestamp).
    endTime: int
        Job end time (timestamp).
    finalMetricData: list of `nni.experiment.TrialMetricData`
        See `nni.experiment.TrialMetricData`.
    stderrPath: str
        Stderr log path.
    sequenceId: int
        Sequence Id.
    """
    trialJobId: str
    status: str
    hyperParameters: List[TrialHyperParameters]
    logPath: str
    startTime: int
    endTime: int
    finalMetricData: List[TrialMetricData]
    stderrPath: str
    sequenceId: int

    def __init__(self, trialJobId: str, status: str, logPath: str, startTime: int, sequenceId: int,
                 endTime: int = -1, stderrPath: str = '', hyperParameters: List = [], finalMetricData: List = []):
        self.trialJobId = trialJobId
        self.status = status
        self.hyperParameters = [TrialHyperParameters(**json.loads(e)) for e in hyperParameters]
        self.logPath = logPath
        self.startTime = startTime
        self.endTime = endTime
        self.finalMetricData = [TrialMetricData(**e) for e in finalMetricData]
        self.stderrPath = stderrPath
        self.sequenceId = sequenceId
