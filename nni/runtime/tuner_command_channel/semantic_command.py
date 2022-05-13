# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class BasicCommand:
    def __init__(self, trial_job_id, parameter_id, parameter_index, parameters, parameter_source):
        self.trial_job_id = trial_job_id
        self.parameter_id = parameter_id
        self.parameter_index = parameter_index
        self.parameters = parameters
        self.parameter_source = parameter_source

class ReportMetricData:
    def __init__(self, trial_job_id, parameter_id, parameter_index, type, value, sequence):
        self.trial_job_id = trial_job_id
        self.parameter_id = parameter_id
        self.parameter_index = parameter_index
        self.type = type
        self.value = value
        self.sequence = sequence

class UpdateSearchSpace:
     def __init__(self, name):
        self.name = name

class ImportData:
     def __init__(self, parameter, value):
        self.parameter = parameter
        self.value = value

class TrialEnd:
     def __init__(self, trial_job_id, event):
        self.trial_job_id = trial_job_id
        self.event = event

class NewTrialJob(BasicCommand):
    def __init__(self, trial_job_id, parameter_id, parameter_index, parameters, parameter_source, placement_constraint, version_info):
        self.placement_constraint = placement_constraint
        self.version_info = version_info
        super().__init__(trial_job_id, parameter_id, parameter_index, parameters, parameter_source)

class SendTrialJobParameter(BasicCommand):
     def __init__(self, trial_job_id, parameter_id, parameter_index, parameters, parameter_source):
        super().__init__(trial_job_id, parameter_id, parameter_index, parameters, parameter_source)

class NoMoreTrialJobs(BasicCommand):
    def __init__(self, trial_job_id, parameter_id, parameter_index, parameters, parameter_source):
        super().__init__(trial_job_id, parameter_id, parameter_index, parameters, parameter_source)

class KillTrialJob:
     def __init__(self, trial_job_id):
        self.trial_job_id = trial_job_id