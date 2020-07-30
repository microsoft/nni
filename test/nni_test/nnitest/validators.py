# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os.path as osp
from os import remove
import subprocess
import json
import requests
import nnicli as nc
from utils import METRICS_URL


class ITValidator:
    def __call__(self, rest_endpoint, experiment_dir, nni_source_dir, **kwargs):
        pass

class ExportValidator(ITValidator):
    def __call__(self, rest_endpoint, experiment_dir, nni_source_dir, **kwargs):
        exp_id = osp.split(experiment_dir)[-1]
        proc1 = subprocess.run(["nnictl", "experiment", "export", exp_id, "-t", "csv", "-f", "report.csv"])
        assert proc1.returncode == 0, '`nnictl experiment export -t csv` failed with code %d' % proc1.returncode
        with open("report.csv", 'r') as f:
            print('Exported CSV file: \n')
            print(''.join(f.readlines()))
            print('\n\n')
        remove('report.csv')

        proc2 = subprocess.run(["nnictl", "experiment", "export", exp_id, "-t", "json", "-f", "report.json"])
        assert proc2.returncode == 0, '`nnictl experiment export -t json` failed with code %d' % proc2.returncode
        with open("report.json", 'r') as f:
            print('Exported JSON file: \n')
            print('\n'.join(f.readlines()))
            print('\n\n')
        remove('report.json')

class MetricsValidator(ITValidator):
    def __call__(self, rest_endpoint, experiment_dir, nni_source_dir, **kwargs):
        self.check_metrics(nni_source_dir, **kwargs)

    def check_metrics(self, nni_source_dir, **kwargs):
        expected_result_file = kwargs.get('expected_result_file', 'expected_metrics.json')
        with open(osp.join(nni_source_dir, 'test', 'config', 'metrics_test', expected_result_file), 'r') as f:
            expected_metrics = json.load(f)
        print('expected metrics:', expected_metrics)
        metrics = requests.get(METRICS_URL).json()
        print('RAW METRICS:', json.dumps(metrics, indent=4))
        intermediate_result, final_result = self.get_metric_results(metrics)

        assert intermediate_result and final_result
        for trialjob_id in intermediate_result:
            trial_final_result = final_result[trialjob_id]
            trial_intermediate_result = intermediate_result[trialjob_id]
            print('intermediate result:', trial_intermediate_result)
            print('final result:', trial_final_result)
            assert len(trial_final_result) == 1, 'there should be 1 final result'
            assert trial_final_result[0] == expected_metrics['final_result']
            # encode dict/number into json string to compare them in set
            assert set([json.dumps(x, sort_keys=True) for x in trial_intermediate_result]) \
                == set([json.dumps(x, sort_keys=True) for x in expected_metrics['intermediate_result']])

    def get_metric_results(self, metrics):
        intermediate_result = {}
        final_result = {}
        for metric in metrics:
            # metrics value are encoded by NNI SDK as json string,
            # here we decode the value by json.loads twice
            metric_value = json.loads(json.loads(metric['data']))
            if metric['type'] == 'PERIODICAL':
                if metric['trialJobId'] in intermediate_result:
                    intermediate_result[metric['trialJobId']].append(metric_value)
                else:
                    intermediate_result[metric['trialJobId']] = [metric_value]
            elif metric['type'] == 'FINAL':
                if metric['trialJobId'] in final_result:
                    final_result[metric['trialJobId']].append(metric_value)
                else:
                    final_result[metric['trialJobId']] = [metric_value]
        return intermediate_result, final_result
