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
    def load(command_dict: dict) -> 'BaseCommand':
        return BaseCommand(**command_dict)

    def to_info_dict(self) -> Dict[str, str]:
        return {"command_type": self.command_type}

@dataclass
class ReportMetricData(BaseCommand):
    type: Literal['FINAL', 'PERIODICAL', 'REQUEST_PARAMETER']
    value: str
    sequence: int

    @staticmethod
    def load(command_dict: dict) -> 'ReportMetricData':
        return ReportMetricData(**command_dict)

@dataclass
class UpdateSearchSpace(BaseCommand):
    name: str

    @staticmethod
    def load(command_dict: dict) -> 'UpdateSearchSpace':
        return UpdateSearchSpace(**command_dict)

@dataclass
class ImportData(BaseCommand):
    parameters: dict
    value: str

    @staticmethod
    def load(command_dict: dict) -> 'ImportData':
        return ImportData(**command_dict)

@dataclass
class TrialEnd(BaseCommand):
    trial_job_id: str
    event: str

    @staticmethod
    def load(command_dict: dict) -> 'TrialEnd':
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
    def load(command_dict: dict) -> 'NewTrialJob':
        return NewTrialJob(**command_dict)

@dataclass
class SendTrialJobParameter(BaseCommand):
    trial_job_id: str
    parameter_id: int
    parameter_index: int
    parameters: dict
    parameter_source: str

    @staticmethod
    def load(command_dict: dict) -> 'SendTrialJobParameter':
        return SendTrialJobParameter(**command_dict)

@dataclass
class NoMoreTrialJobs(BaseCommand):
    trial_job_id: str
    parameter_id: int
    parameter_index: int
    parameters: dict
    parameter_source: str

    @staticmethod
    def load(command_dict: dict) -> 'NoMoreTrialJobs':
        return NoMoreTrialJobs(**command_dict)

@dataclass
class KillTrialJob(BaseCommand):
    trial_job_id: str

    @staticmethod
    def load(command_dict: dict) -> 'KillTrialJob':
        return KillTrialJob(**command_dict)