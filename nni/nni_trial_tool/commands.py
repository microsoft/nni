# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class CommandType(Enum):
    Initialize = b'IN'
    RequestTrialJobs = b'GE'
    ReportMetricData = b'ME'
    ReportGpuInfo = b'GI'
    UpdateSearchSpace = b'SS'
    ImportData = b'FD'
    AddCustomizedTrialJob = b'AD'
    TrialEnd = b'EN'
    Terminate = b'TE'
    Ping = b'PI'

    Initialized = b'ID'
    NewTrialJob = b'TR'
    SendTrialJobParameter = b'SP'
    NoMoreTrialJobs = b'NO'
    KillTrialJob = b'KI'
    StdOut = b'SO'
    VersionCheck = b'VC'
