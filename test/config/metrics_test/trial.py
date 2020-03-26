# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import json
import argparse
import nni

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_metrics", action='store_true')
    args = parser.parse_args()

    if args.dict_metrics:
        result_file = 'expected_metrics_dict.json'
    else:
        result_file = 'expected_metrics.json'

    nni.get_next_parameter()
    with open(result_file, 'r') as f:
        m = json.load(f)
    for v in m['intermediate_result']:
        time.sleep(1)
        print('report_intermediate_result:', v)
        nni.report_intermediate_result(v)
    time.sleep(1)
    print('report_final_result:', m['final_result'])
    nni.report_final_result(m['final_result'])
    print('done')
