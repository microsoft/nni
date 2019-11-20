# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json_tricks


def get_next_parameter():
    pass

def get_experiment_id():
    pass

def get_trial_id():
    pass

def get_sequence_id():
    pass

def send_metric(string):
    metric = json_tricks.loads(string)
    if metric['type'] == 'FINAL':
        print('Final result:', metric['value'])
    elif metric['type'] == 'PERIODICAL':
        print('Intermediate result:', metric['value'])
