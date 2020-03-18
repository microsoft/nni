# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os.path as osp
import json
import requests
from utils import GREEN, RED, CLEAR, STATUS_URL, TRIAL_JOBS_URL, METRICS_URL


class ITValidator:
    def __call__(self, api_root_url, experiment_dir, nni_source_dir):
        pass


class MetricsValidator(ITValidator):
    def __call__(self, api_root_url, experiment_dir, nni_source_dir):
        #print('VALIDATOR CALLED!!!')
        self.check_metrics(nni_source_dir)

    def check_metrics(self, nni_source_dir):
        with open(osp.join(nni_source_dir, 'test', 'config', 'metrics_test', 'expected_metrics.json'), 'r') as f:
            expected_metrics = json.load(f)
        print('expected metrics:', expected_metrics)
        metrics = requests.get(METRICS_URL).json()
        intermediate_result, final_result = self.get_metric_results(metrics)

        assert intermediate_result and final_result
        for trialjob_id in intermediate_result:
            trial_final_result = final_result[trialjob_id]
            trial_intermediate_result = intermediate_result[trialjob_id]
            print('intermediate result:', trial_intermediate_result)
            print('final result:', trial_final_result)
            assert len(trial_final_result) == 1, 'there should be 1 final result'
            assert trial_final_result[0] == expected_metrics['final_result']
            assert set(trial_intermediate_result) == set(expected_metrics['intermediate_result'])

    def get_metric_results(self, metrics):
        intermediate_result = {}
        final_result = {}
        for metric in metrics:
            metric_value = round(float(json.loads(metric['data'])), 2)
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
