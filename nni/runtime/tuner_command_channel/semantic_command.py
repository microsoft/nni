# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
import json
from typing import Dict, Literal

@dataclass
class BaseCommand:
    command_type: str

    def dump(self) -> str:
        command_dict = self.__dict__
        return json.dumps(command_dict)

    @staticmethod
    def load(command_json: str) -> 'BaseCommand':
        command_dict = json.loads(command_json)
        return BaseCommand(**command_dict)

    def to_info_dict(self) -> Dict[str, str]:
        return {"command_type": self.command_type}

@dataclass
class ReportMetricData(BaseCommand):
    type: Literal['FINAL', 'PERIODICAL', 'REQUEST_PARAMETER']
    value: str
    sequence: int

    @staticmethod
    def load(command_json: str) -> 'ReportMetricData':
        command_dict = json.loads(command_json)
        return ReportMetricData(**command_dict)

@dataclass
class UpdateSearchSpace(BaseCommand):
    name: str

    @staticmethod
    def load(command_json: str) -> 'UpdateSearchSpace':
        command_dict = json.loads(command_json)
        return UpdateSearchSpace(**command_dict)

@dataclass
class ImportData(BaseCommand):
    parameters: dict
    value: str

    @staticmethod
    def load(command_json: str) -> 'ImportData':
        command_dict = json.loads(command_json)
        return ImportData(**command_dict)

@dataclass
class TrialEnd(BaseCommand):
    trial_job_id: str
    event: str

    @staticmethod
    def load(command_json: str) -> 'TrialEnd':
        command_dict = json.loads(command_json)
        return TrialEnd(**command_dict)

@dataclass
class NewTrialJob(BaseCommand):
    trial_job_id: str
    parameter_id: int
    parameter_index: int
    parameters: dict
    parameter_source: str
    placement_constraint: str
    version_info: str

    @staticmethod
    def load(command_json: str) -> 'NewTrialJob':
        command_dict = json.loads(command_json)
        return NewTrialJob(**command_dict)

@dataclass
class SendTrialJobParameter(BaseCommand):
    trial_job_id: str
    parameter_id: int
    parameter_index: int
    parameters: dict
    parameter_source: str

    @staticmethod
    def load(command_json: str) -> 'SendTrialJobParameter':
        command_dict = json.loads(command_json)
        return SendTrialJobParameter(**command_dict)

@dataclass
class NoMoreTrialJobs(BaseCommand):
    trial_job_id: str
    parameter_id: int
    parameter_index: int
    parameters: dict
    parameter_source: str

    @staticmethod
    def load(command_json: str) -> 'NoMoreTrialJobs':
        command_dict = json.loads(command_json)
        return NoMoreTrialJobs(**command_dict)

@dataclass
class KillTrialJob(BaseCommand):
    trial_job_id: str

    @staticmethod
    def load(command_json: str) -> 'KillTrialJob':
        command_dict = json.loads(command_json)
        return KillTrialJob(**command_dict)