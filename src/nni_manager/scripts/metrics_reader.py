# ============================================================================================================================== #
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================================================================== #

import argparse
import errno
import json
import os
import re

METRICS_FILENAME = '.nni/metrics'
OFFSET_FILENAME  = '.nni/metrics_offset'
JOB_CODE_FILENAME = '.nni/code'
JOB_PID_FILENAME = '.nni/jobpid'
JOB_CODE_PATTERN = re.compile('^(\d+)\s+(\d+)$')

LEN_FIELD_SIZE = 6
MAGIC = 'ME'

class TrialMetricsReader():
    '''
    Read metrics data from a trial job
    '''
    def __init__(self, trial_job_dir):
        self.trial_job_dir = trial_job_dir
        self.offset_filename = os.path.join(trial_job_dir, OFFSET_FILENAME)
        self.metrics_filename = os.path.join(trial_job_dir, METRICS_FILENAME)
        self.jobcode_filename = os.path.join(trial_job_dir, JOB_CODE_FILENAME)
        self.jobpid_filemame = os.path.join(trial_job_dir, JOB_PID_FILENAME)

    def _metrics_file_is_empty(self):
        if not os.path.isfile(self.metrics_filename):
            return True
        statinfo = os.stat(self.metrics_filename)
        return statinfo.st_size == 0

    def _get_offset(self):
        offset = 0
        if os.path.isfile(self.offset_filename):
            with open(self.offset_filename, 'r') as f:
                offset = int(f.readline())
        return offset

    def _write_offset(self, offset):
        statinfo = os.stat(self.metrics_filename)
        if offset < 0 or offset > statinfo.st_size:
            raise ValueError('offset value is invalid: {}'.format(offset))

        with open(self.offset_filename, 'w') as f:
            f.write(str(offset)+'\n')

    def _read_all_available_records(self, offset):
        new_offset = offset
        metrics = []
        with open(self.metrics_filename, 'r') as f:
            f.seek(offset)
            while True:
                magic_string = f.read(len(MAGIC))
                # empty data means EOF
                if not magic_string:
                    break
                strdatalen = f.read(LEN_FIELD_SIZE)
                # empty data means EOF
                if not strdatalen:
                    raise ValueError("metric file {} format error after offset: {}.".format(self.metrics_filename, new_offset))
                datalen = int(strdatalen)
                data = f.read(datalen)

                if datalen > 0 and len(data) == datalen:
                    new_offset = f.tell()
                    metrics.append(data)
                else:
                    raise ValueError("metric file {} format error after offset: {}.".format(self.metrics_filename, new_offset))
        self._write_offset(new_offset)
        return metrics

    def _pid_exists(selft, pid):
        if pid < 0:
            return False
        if pid == 0:
            # According to "man 2 kill" PID 0 refers to every process
            # in the process group of the calling process.
            # On certain systems 0 is a valid PID but we have no way
            # to know that in a portable fashion.
            raise ValueError('invalid PID 0')
        try:
            os.kill(pid, 0)
        except OSError as err:
            if err.errno == errno.ESRCH:
                # ESRCH == No such process
                return False
            elif err.errno == errno.EPERM:
                # EPERM clearly means there's a process to deny access to
                return True
            else:
                # According to "man 2 kill" possible error values are
                # (EINVAL, EPERM, ESRCH)
                raise
        else:
            return True

    def read_trial_metrics(self):
        '''
        Read available metrics data for a trial
        '''
        if self._metrics_file_is_empty():
            return []

        offset = self._get_offset()
        return self._read_all_available_records(offset)

    def read_trial_status(self):
        if os.path.isfile(self.jobpid_filemame):
            with open(self.jobpid_filemame, 'r') as f:
                jobpid = int(f.readline())
                if self._pid_exists(jobpid):
                    return 'RUNNING' ,-1
                else:
                    return self._read_job_return_code()
        else:
            # raise ValueError('offset value is invalid: {}'.format(offset))
            return 'UNKNOWN' ,-1

    def _read_job_return_code(self):
        if os.path.isfile(self.jobcode_filename):
            with open(self.jobcode_filename, 'r') as f:
                job_return_code = f.readline()
                match = JOB_CODE_PATTERN.match(job_return_code)
                if(match):
                    return_code = int(match.group(1))
                    timestamp = int(match.group(2))
                    status = ''
                    if return_code == 0:
                        status = 'SUCCEEDED'
                    elif return_code > 128:
                        status = 'USER_CANCELED'
                    else:
                        status = 'FAILED'
                    return status, timestamp
                else:
                    raise ValueError('Job code file format incorrect')
        else:
            raise ValueError('job return code file doesnt exist: {}'.format(self.jobcode_filename))


def read_experiment_metrics(args):
    '''
    Read metrics data for specified trial jobs
    '''
    trial_job_ids = args.trial_job_ids.strip().split(',')
    trial_job_ids = [id.strip() for id in trial_job_ids]
    results = []
    for trial_job_id in trial_job_ids:
        result = {}
        try:
            trial_job_dir = os.path.join(args.experiment_dir, 'trials', trial_job_id)
            reader = TrialMetricsReader(trial_job_dir)
            result['jobId'] = trial_job_id
            result['metrics'] = reader.read_trial_metrics()
            result['jobStatus'], result['endTimestamp'] = reader.read_trial_status()
            results.append(result)
        except Exception:
            #TODO error logging to file
            pass
    print(json.dumps(results))




if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--experiment_dir", type=str, help="Root directory of experiment", required=True)
    PARSER.add_argument("--trial_job_ids", type=str, help="Trial job ids splited with ','", required=True)

    ARGS, UNKNOWN = PARSER.parse_known_args()
    read_experiment_metrics(ARGS)

