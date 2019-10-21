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

import nni
import subprocess
import logging

LOG = logging.getLogger('rocksdb-fillrandom')


def run(**parameters):
    '''Run rocksdb benchmark and return throughput'''
    bench_type = parameters['benchmarks']
    # recover args
    args = ["--{}={}".format(k, v) for k, v in parameters.items()]
    # subprocess communicate
    process = subprocess.Popen(['db_bench'] + args, stdout=subprocess.PIPE)
    out, err = process.communicate()
    # split into lines
    lines = out.decode("utf8").splitlines()

    match_lines = []
    for line in lines:
        # find the line with matched str
        if bench_type not in line:
            continue
        else:
            match_lines.append(line)
            break

    results = {}
    for line in match_lines:
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.split("op")[1]
        results[key] = float(value)

    return results[bench_type]


def generate_params(received_params):
    '''generate parameters based on received parameters'''
    params = {
        "benchmarks": "fillrandom",
        "threads": 1,
        "key_size": 20,
        "value_size": 100,
        "num": 13107200,
        "db": "/tmp/rockdb",
        "disable_wal": 1,
        "max_background_flushes": 1,
        "max_background_compactions": 4,
        "write_buffer_size": 67108864,
        "max_write_buffer_number": 16,
        "min_write_buffer_number_to_merge": 2,
        "level0_file_num_compaction_trigger": 2,
        "max_bytes_for_level_base": 268435456,
        "max_bytes_for_level_multiplier": 10,
        "target_file_size_base": 33554432,
        "target_file_size_multiplier": 1
    }

    for k, v in received_params.items():
        params[k] = int(v)

    return params


if __name__ == "__main__":
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_params(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        # run benchmark
        throughput = run(**PARAMS)
        # report throughput to nni
        nni.report_final_result(throughput)
    except Exception as exception:
        LOG.exception(exception)
        raise
